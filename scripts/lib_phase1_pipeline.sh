#!/usr/bin/env bash
# =====================================================================
#  Shared Phase-1 fine-tune + evaluation pipeline for DINOv3 (HF) on NAVI.
#  ────────────────────────────────────────────────────────────────────
#  Phase 1 reproduces the "Projection Head + InfoNCE — Catastrophic
#  Collapse" experiment from main.tex §3.2:
#    * Backbone is FULLY FROZEN (no LoRA).
#    * A randomly-initialised ProjectionHead (embed_dim → mid → 256, L2-norm)
#      is added on top and trained with PLAIN InfoNCE (no Safe Radius).
#    * Expected signature of failure: contrastive loss → ln(N_neg + 1)
#      and a sharp Precision drop vs. zero-shot.
#
#  This file is meant to be `source`d from a thin entry-point script
#  (e.g. train_phase1.sh, train_phase1_middle.sh) which is solely
#  responsible for setting:
#
#      MODEL=small   (or middle)
#      EPOCHS=15     (or whatever was passed on the command line)
#
#  before sourcing.  Output layout: output/navi_<model>_phase1/
#
#  Pipeline (5 steps; no zero-shot duplication — those metrics already
#  exist in output/navi_<model>/eval_zeroshot/ from the LoRA pipeline):
#    [1] Verify Python deps
#    [2] Phase-1 training                    (frozen + ProjHead + plain InfoNCE)
#    [3] Final-ckpt feature extraction       (latest ckpt)
#    [4] Final-ckpt evaluation               (final metrics)
#    [5] Per-epoch evaluation                (training curve)
#
#  Step gating env vars:
#    START_STEP=1         (default) first step to run
#    END_STEP=5           (default) last step to run
#    EVAL_START_EPOCH=0   step 5: skip per-epoch eval below this index
#    EVAL_FORCE=0         step 5: 1 = re-run even if results exist
# =====================================================================
set -euo pipefail

: "${MODEL:?lib_phase1_pipeline.sh: MODEL must be set (small|middle) by caller}"
: "${EPOCHS:?lib_phase1_pipeline.sh: EPOCHS must be set by caller}"

case "$MODEL" in
  small|middle) ;;
  *)
    echo "[lib] Unknown MODEL '$MODEL'. Choose: small | middle"
    exit 1
    ;;
esac

START_STEP=${START_STEP:-1}
END_STEP=${END_STEP:-5}
EVAL_START_EPOCH=${EVAL_START_EPOCH:-0}
EVAL_FORCE=${EVAL_FORCE:-0}

run_step() {
    local s=$1
    [ "$s" -ge "$START_STEP" ] && [ "$s" -le "$END_STEP" ]
}

# ---------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)
cd "$PROJECT_ROOT"

WEIGHTS_DIR="dinov3_weights/dinov3-${MODEL}"
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "[error] Weights directory not found: $WEIGHTS_DIR"
    exit 1
fi

# Phase-1 hyper-params (override via env vars).
IMG_SIZE=${IMG_SIZE:-448}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
PROJ_DIM=${PROJ_DIM:-256}
NUM_WORKERS=${NUM_WORKERS:-4}

# NAVI-only data paths
TRAIN_PAIRS="finetune/navi_train_pairs.txt"
DATA_ROOT="datasets/navi_v1.5"
EVAL_W=1024
EVAL_H=1024

if [ -f "datasets/navi_test_pairs.txt" ]; then
    EVAL_PAIRS="datasets/navi_test_pairs.txt"
elif [ -f "datasets/navi_with_gt.txt" ]; then
    EVAL_PAIRS="datasets/navi_with_gt.txt"
else
    EVAL_PAIRS="$TRAIN_PAIRS"
    echo "[warn] No held-out NAVI test pairs found; falling back to training pairs."
fi

# Output layout: output/navi_<model>_phase1/
OUT_ROOT="output/navi_${MODEL}_phase1"
OUT_CKPT="${OUT_ROOT}/ckpt"
MNN_FT="${OUT_ROOT}/mnn_phase1"
EVAL_FT="${OUT_ROOT}/eval_phase1"
EVAL_PER_EPOCH="${OUT_ROOT}/eval_per_epoch"
mkdir -p "$OUT_ROOT"

echo "===================================================================="
echo "  DINOv3 Phase-1 (frozen + ProjHead + InfoNCE) pipeline  (NAVI)"
echo "  Model        : $MODEL  ($WEIGHTS_DIR)"
echo "  Train pairs  : $TRAIN_PAIRS"
echo "  Eval pairs   : $EVAL_PAIRS"
echo "  Output root  : $OUT_ROOT"
echo "  Epochs       : $EPOCHS"
echo "  Img size     : $IMG_SIZE"
echo "  Batch size   : $BATCH_SIZE"
echo "  LR           : $LR"
echo "  Proj dim     : $PROJ_DIM"
echo "  Num workers  : $NUM_WORKERS"
echo "  Steps        : ${START_STEP}..${END_STEP}"
if [ "$START_STEP" -le 5 ] && [ "$END_STEP" -ge 5 ]; then
echo "  Eval resume  : start_epoch=$EVAL_START_EPOCH  force=$EVAL_FORCE"
fi
echo "===================================================================="

# ---------------------------------------------------------------------
#  Auto-activate venv
# ---------------------------------------------------------------------
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -d "$VENV_DIR" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    echo "[env] Activated venv: $VENV_DIR"
else
    echo "[env] No local .venv found at $VENV_DIR (continuing with system Python)."
fi
PY=$(command -v python || command -v python3)
echo "[env] Python: $($PY --version 2>&1) ($PY)"

# ---------------------------------------------------------------------
#  Step 1 — verify deps
# ---------------------------------------------------------------------
if run_step 1; then
echo ""
echo "[1/5] Verifying Python deps..."
$PY - <<'PY'
import importlib, sys
need = ["torch", "torchvision", "transformers", "safetensors",
        "numpy", "cv2", "PIL"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  ok  {m:14s} {v}")
    except Exception as e:
        missing.append(m)
        print(f"  MISS {m:14s} ({e.__class__.__name__})")
if missing:
    print(f"\nMissing modules: {missing}")
    sys.exit(1)
PY
else
    echo ""
    echo "[1/5] SKIPPED (START_STEP=$START_STEP)"
fi

# ---------------------------------------------------------------------
#  Step 2 — Phase-1 training
#  (Saves checkpoint_epoch* every SAVE_EVERY=3 epochs + final epoch.)
# ---------------------------------------------------------------------
if run_step 2; then
echo ""
echo "[2/5] Phase-1 training → $OUT_CKPT"
$PY -m finetune.train_lora_hf \
    --phase1 \
    --proj_dim "$PROJ_DIM" \
    --weights_dir "$WEIGHTS_DIR" \
    --train_pairs "$TRAIN_PAIRS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT_CKPT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --img_size "$IMG_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --no_hard_negatives
else
    echo ""
    echo "[2/5] SKIPPED"
fi

CKPT="$OUT_CKPT/checkpoint_latest.pth"

# ---------------------------------------------------------------------
#  Step 3 — final-ckpt feature extraction + MNN matching
# ---------------------------------------------------------------------
if run_step 3; then
echo ""
echo "[3/5] Phase-1 feature extraction (latest ckpt) → $MNN_FT"
if [ ! -f "$CKPT" ]; then
    echo "  [error] checkpoint not found: $CKPT"
    exit 1
fi
$PY -m finetune.extract_and_match_hf \
    --weights_dir "$WEIGHTS_DIR" \
    --checkpoint "$CKPT" \
    --pairs "$EVAL_PAIRS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$MNN_FT" \
    --img_size "$IMG_SIZE" \
    --eval_resize "$EVAL_W" "$EVAL_H"
else
    echo ""
    echo "[3/5] SKIPPED"
fi

# ---------------------------------------------------------------------
#  Step 4 — final-ckpt evaluation
# ---------------------------------------------------------------------
if run_step 4; then
echo ""
echo "[4/5] Phase-1 evaluation (latest ckpt) → $EVAL_FT"
mkdir -p "$EVAL_FT"
$PY evaluate/evaluate_csv_essential.py \
    --input_pairs "$EVAL_PAIRS" \
    --input_dir "$DATA_ROOT" \
    --input_csv_dir "$MNN_FT" \
    --output_dir "$EVAL_FT" \
    --resize "$EVAL_W" "$EVAL_H" || \
    echo "  (evaluate script returned non-zero — inspect $EVAL_FT)"
else
    echo ""
    echo "[4/5] SKIPPED"
fi

# ---------------------------------------------------------------------
#  Step 5 — per-epoch evaluation (training curve)
# ---------------------------------------------------------------------
SUMMARY_TSV="$EVAL_PER_EPOCH/summary.tsv"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if run_step 5; then
echo ""
echo "[5/5] Per-epoch evaluation → $EVAL_PER_EPOCH"
echo "      EVAL_START_EPOCH=$EVAL_START_EPOCH  EVAL_FORCE=$EVAL_FORCE"
mkdir -p "$EVAL_PER_EPOCH"

shopt -s nullglob
CKPT_LIST=( "$OUT_CKPT"/checkpoint_epoch*.pth )
shopt -u nullglob

if [ ${#CKPT_LIST[@]} -eq 0 ]; then
    echo "  [warn] No checkpoint_epoch*.pth found in $OUT_CKPT — skipping per-epoch eval."
else
    echo "  Found ${#CKPT_LIST[@]} epoch checkpoints."
    for CKPT_E in "${CKPT_LIST[@]}"; do
        FNAME=$(basename "$CKPT_E")
        EPOCH_TAG="${FNAME#checkpoint_epoch}"
        EPOCH_TAG="${EPOCH_TAG%.pth}"
        EPOCH_NUM=$((10#$EPOCH_TAG))
        EP_DIR="$EVAL_PER_EPOCH/epoch${EPOCH_TAG}"
        EP_EVAL_DIR="$EP_DIR/eval"
        RES_JSON="$EP_EVAL_DIR/evaluation_results.json"
        RES_TXT="$EP_EVAL_DIR/evaluation_results.txt"

        if [ "$EPOCH_NUM" -lt "$EVAL_START_EPOCH" ]; then
            echo ""
            echo "  ── epoch ${EPOCH_TAG} ── SKIP (< EVAL_START_EPOCH=$EVAL_START_EPOCH)"
            continue
        fi
        if [ "$EVAL_FORCE" != "1" ] && { [ -f "$RES_JSON" ] || [ -f "$RES_TXT" ]; }; then
            DONE_FILE="$RES_JSON"; [ -f "$RES_JSON" ] || DONE_FILE="$RES_TXT"
            echo ""
            echo "  ── epoch ${EPOCH_TAG} ── SKIP (already evaluated: $DONE_FILE)"
            continue
        fi

        mkdir -p "$EP_EVAL_DIR"

        echo ""
        echo "  ── epoch ${EPOCH_TAG} ──────────────────────────────"
        echo "  ckpt : $CKPT_E"
        echo "  out  : $EP_DIR"

        $PY -m finetune.extract_and_match_hf \
            --weights_dir "$WEIGHTS_DIR" \
            --checkpoint "$CKPT_E" \
            --pairs "$EVAL_PAIRS" \
            --data_root "$DATA_ROOT" \
            --output_dir "$EP_DIR" \
            --img_size "$IMG_SIZE" \
            --eval_resize "$EVAL_W" "$EVAL_H" || {
                echo "  [warn] extract failed for epoch ${EPOCH_TAG}, skipping eval."
                continue
            }

        $PY evaluate/evaluate_csv_essential.py \
            --input_pairs "$EVAL_PAIRS" \
            --input_dir "$DATA_ROOT" \
            --input_csv_dir "$EP_DIR" \
            --output_dir "$EP_EVAL_DIR" \
            --resize "$EVAL_W" "$EVAL_H" || {
                echo "  [warn] evaluate returned non-zero for epoch ${EPOCH_TAG}"
            }
    done

    # Rebuild summary.tsv via the shared helper.
    echo ""
    echo "  Rebuilding summary.tsv from existing per-epoch results..."
    $PY "$SCRIPT_DIR/build_eval_summary.py" "$EVAL_PER_EPOCH" "$SUMMARY_TSV" || {
        echo "  [warn] build_eval_summary.py exited non-zero (no usable results yet?)"
    }
fi
else
    echo ""
    echo "[5/5] SKIPPED"
fi

echo ""
echo "===================================================================="
echo "  DONE.  Phase-1 ($MODEL)"
echo "  Phase-1 final  : $EVAL_FT/evaluation_results.json"
echo "  Per-epoch      : $SUMMARY_TSV"
echo "  Training log   : $OUT_CKPT/training_log.json"
echo "  (Compare against zero-shot at output/navi_${MODEL}/eval_zeroshot/)"
echo "===================================================================="
