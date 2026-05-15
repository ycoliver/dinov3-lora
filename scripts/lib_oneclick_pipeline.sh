#!/usr/bin/env bash
# =====================================================================
#  Shared LoRA fine-tune + evaluation pipeline for DINOv3 (HF) on NAVI.
#  ────────────────────────────────────────────────────────────────────
#  This file is meant to be `source`d from a thin entry-point script
#  (e.g. train_oneclick.sh, train_oneclick_middle.sh) which is solely
#  responsible for setting:
#
#      MODEL=small   (or middle)
#      EPOCHS=15     (or whatever was passed on the command line)
#
#  before sourcing.  Keep the model-specific entry-points small so that
#  future changes to the pipeline only need to be made here.
#
#  Pipeline (7 steps, all artefacts under output/navi_<model>/):
#    [1] Verify Python deps
#    [2] Zero-shot feature extraction + MNN matching     (baseline)
#    [3] Zero-shot evaluation                            (baseline metrics)
#    [4] LoRA fine-tune                                  (training)
#    [5] Fine-tuned feature extraction (latest ckpt)     (post-train)
#    [6] Fine-tuned evaluation (latest ckpt)             (final metrics)
#    [7] Per-epoch evaluation                            (training curve)
#
#  Note: training saves a checkpoint every SAVE_EVERY (=3) epochs
#  (controlled in finetune/config.py), so step 7 evaluates roughly
#  one curve point every 3 epochs by default.
#
#  Step gating env vars:
#    START_STEP=1         (default) first step to run
#    END_STEP=7           (default) last step to run
#    EVAL_START_EPOCH=0   step 7: skip per-epoch eval below this index
#    EVAL_FORCE=0         step 7: 1 = re-run even if results exist
#                         (default: skip any epoch whose
#                          evaluation_results.json already exists, with
#                          legacy .txt also accepted as "done")
# =====================================================================
set -euo pipefail

# ---------------------------------------------------------------------
#  Required from the entry-point script
# ---------------------------------------------------------------------
: "${MODEL:?lib_oneclick_pipeline.sh: MODEL must be set (small|middle) by caller}"
: "${EPOCHS:?lib_oneclick_pipeline.sh: EPOCHS must be set by caller}"

case "$MODEL" in
  small|middle) ;;
  *)
    echo "[lib] Unknown MODEL '$MODEL'. Choose: small | middle"
    exit 1
    ;;
esac

# Step gating: which steps to run. Default = full pipeline (1..7).
START_STEP=${START_STEP:-1}
END_STEP=${END_STEP:-7}

# Per-epoch eval (STEP 7) resume controls.
EVAL_START_EPOCH=${EVAL_START_EPOCH:-0}     # numeric, e.g. 0, 5, 12
EVAL_FORCE=${EVAL_FORCE:-0}                 # 1 = re-run even if results exist

run_step() {
    local s=$1
    [ "$s" -ge "$START_STEP" ] && [ "$s" -le "$END_STEP" ]
}

# ---------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)
cd "$PROJECT_ROOT"

# dinov3_weights/dinov3-small/   (≈ 86M)
# dinov3_weights/dinov3-middle/  (≈ 300M, 1.21G safetensors)
WEIGHTS_DIR="dinov3_weights/dinov3-${MODEL}"
if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "[error] Weights directory not found: $WEIGHTS_DIR"
    echo "        Expected layout:"
    echo "          dinov3_weights/dinov3-small/"
    echo "          dinov3_weights/dinov3-middle/"
    exit 1
fi

# Hyper-params (shared by small and middle).
# Each can be overridden from the command line, e.g.
#   BATCH_SIZE=16 LR=2e-4 LORA_RANK=8 LORA_ALPHA=16 \
#       bash scripts/train_oneclick_middle.sh 15
IMG_SIZE=${IMG_SIZE:-448}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-1e-4}
LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-8.0}

# NAVI-only data paths
TRAIN_PAIRS="finetune/navi_train_pairs.txt"
DATA_ROOT="datasets/navi_v1.5"
EVAL_W=1024
EVAL_H=1024

# Eval pairs: prefer the held-out test split; fall back with a warning.
if [ -f "datasets/navi_test_pairs.txt" ]; then
    EVAL_PAIRS="datasets/navi_test_pairs.txt"
elif [ -f "datasets/navi_with_gt.txt" ]; then
    EVAL_PAIRS="datasets/navi_with_gt.txt"
else
    EVAL_PAIRS="$TRAIN_PAIRS"
    echo "[warn] No held-out NAVI test pairs found; falling back to training pairs."
    echo "[warn] Run 'bash scripts/build_navi_pairs.sh' first for a clean eval."
fi

# Output layout: output/navi_<model>/
OUT_ROOT="output/navi_${MODEL}"
OUT_LORA="${OUT_ROOT}/lora_ckpt"
MNN_ZS="${OUT_ROOT}/mnn_zeroshot"
MNN_FT="${OUT_ROOT}/mnn_finetuned"
EVAL_ZS="${OUT_ROOT}/eval_zeroshot"
EVAL_FT="${OUT_ROOT}/eval_finetuned"
EVAL_PER_EPOCH="${OUT_ROOT}/eval_per_epoch"
mkdir -p "$OUT_ROOT"

echo "===================================================================="
echo "  DINOv3 LoRA-HF one-click pipeline  (NAVI)"
echo "  Model        : $MODEL  ($WEIGHTS_DIR)"
echo "  Train pairs  : $TRAIN_PAIRS"
echo "  Eval pairs   : $EVAL_PAIRS"
echo "  Data root    : $DATA_ROOT"
echo "  Output root  : $OUT_ROOT"
echo "  Epochs       : $EPOCHS"
echo "  Img size     : $IMG_SIZE"
echo "  Batch size   : $BATCH_SIZE"
echo "  LR           : $LR"
echo "  LoRA rank    : $LORA_RANK"
echo "  LoRA alpha   : $LORA_ALPHA"
echo "  Steps        : ${START_STEP}..${END_STEP}"
if [ "$START_STEP" -le 7 ] && [ "$END_STEP" -ge 7 ]; then
echo "  Eval resume  : start_epoch=$EVAL_START_EPOCH  force=$EVAL_FORCE"
fi
echo "===================================================================="

# ---------------------------------------------------------------------
#  Auto-activate project-local .venv (built by scripts/setup_env.sh)
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
echo "[1/7] Verifying Python deps..."
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
    print("Run: bash scripts/setup_env.sh")
    sys.exit(1)
PY
else
    echo ""
    echo "[1/7] SKIPPED (START_STEP=$START_STEP)"
fi

# ---------------------------------------------------------------------
#  Step 2 — zero-shot extraction + MNN matching
# ---------------------------------------------------------------------
if run_step 2; then
echo ""
echo "[2/7] Zero-shot feature extraction + MNN matching → $MNN_ZS"
$PY -m finetune.extract_and_match_hf \
    --weights_dir "$WEIGHTS_DIR" \
    --pairs "$EVAL_PAIRS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$MNN_ZS" \
    --img_size "$IMG_SIZE" \
    --eval_resize "$EVAL_W" "$EVAL_H"
else
    echo ""
    echo "[2/7] SKIPPED"
fi

# ---------------------------------------------------------------------
#  Step 3 — zero-shot evaluation
# ---------------------------------------------------------------------
if run_step 3; then
echo ""
echo "[3/7] Zero-shot evaluation → $EVAL_ZS"
mkdir -p "$EVAL_ZS"
$PY evaluate/evaluate_csv_essential.py \
    --input_pairs "$EVAL_PAIRS" \
    --input_dir "$DATA_ROOT" \
    --input_csv_dir "$MNN_ZS" \
    --output_dir "$EVAL_ZS" \
    --resize "$EVAL_W" "$EVAL_H" || \
    echo "  (evaluate script returned non-zero — inspect $EVAL_ZS)"
else
    echo ""
    echo "[3/7] SKIPPED"
fi

# ---------------------------------------------------------------------
#  Step 4 — LoRA fine-tune
#  Checkpoint frequency is controlled by SAVE_EVERY in finetune/config.py
#  (default = 3 → produces checkpoint_epoch002, 005, 008, ... and the final epoch).
# ---------------------------------------------------------------------
if run_step 4; then
echo ""
echo "[4/7] LoRA fine-tuning → $OUT_LORA"
$PY -m finetune.train_lora_hf \
    --weights_dir "$WEIGHTS_DIR" \
    --train_pairs "$TRAIN_PAIRS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT_LORA" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --img_size "$IMG_SIZE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA"
else
    echo ""
    echo "[4/7] SKIPPED"
fi

CKPT="$OUT_LORA/checkpoint_latest.pth"

# ---------------------------------------------------------------------
#  Step 5 — fine-tuned extraction + MNN matching
# ---------------------------------------------------------------------
if run_step 5; then
echo ""
echo "[5/7] Fine-tuned feature extraction (latest ckpt) → $MNN_FT"
if [ ! -f "$CKPT" ]; then
    echo "  [error] checkpoint not found: $CKPT"
    echo "          Run step 4 first, or check $OUT_LORA"
    exit 1
fi
$PY -m finetune.extract_and_match_hf \
    --weights_dir "$WEIGHTS_DIR" \
    --checkpoint "$CKPT" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --pairs "$EVAL_PAIRS" \
    --data_root "$DATA_ROOT" \
    --output_dir "$MNN_FT" \
    --img_size "$IMG_SIZE" \
    --eval_resize "$EVAL_W" "$EVAL_H"
else
    echo ""
    echo "[5/7] SKIPPED"
fi

# ---------------------------------------------------------------------
#  Step 6 — fine-tuned evaluation (using checkpoint_latest.pth)
# ---------------------------------------------------------------------
if run_step 6; then
echo ""
echo "[6/7] Fine-tuned evaluation (latest ckpt) → $EVAL_FT"
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
    echo "[6/7] SKIPPED"
fi

# ---------------------------------------------------------------------
#  Step 7 — per-epoch evaluation
#
#  Iterate over every checkpoint_epoch*.pth produced during training,
#  run extract+evaluate for each, then summarise into one table:
#
#    output/navi_<model>/eval_per_epoch/
#      ├── epoch002/                 ← extracted MNN csvs
#      │   └── eval/                 ← evaluation_results.json (+ .txt)
#      ├── epoch005/...
#      └── summary.tsv               ← one row per epoch
#
#  Because training only saves every SAVE_EVERY (=3) epochs, this naturally
#  evaluates one point per ~3 epochs (plus the final epoch).
#
#  Resume behaviour:
#    * EVAL_START_EPOCH=N skips checkpoints whose epoch index < N.
#    * Without EVAL_FORCE=1, any epoch whose `eval/evaluation_results.json`
#      (or the legacy `.txt`) already exists is skipped, so an interrupted
#      run can be resumed simply by re-invoking with START_STEP=7.
# ---------------------------------------------------------------------
SUMMARY_TSV="$EVAL_PER_EPOCH/summary.tsv"
if run_step 7; then
echo ""
echo "[7/7] Per-epoch evaluation → $EVAL_PER_EPOCH"
echo "      EVAL_START_EPOCH=$EVAL_START_EPOCH  EVAL_FORCE=$EVAL_FORCE"
mkdir -p "$EVAL_PER_EPOCH"

if [ ! -f "$SUMMARY_TSV" ]; then
    echo -e "epoch\tauc@5\tauc@10\tauc@20\tprecision\trecall" > "$SUMMARY_TSV"
fi

shopt -s nullglob
CKPT_LIST=( "$OUT_LORA"/checkpoint_epoch*.pth )
shopt -u nullglob

if [ ${#CKPT_LIST[@]} -eq 0 ]; then
    echo "  [warn] No checkpoint_epoch*.pth found in $OUT_LORA — skipping per-epoch eval."
else
    echo "  Found ${#CKPT_LIST[@]} epoch checkpoints."
    for CKPT_E in "${CKPT_LIST[@]}"; do
        FNAME=$(basename "$CKPT_E")
        EPOCH_TAG="${FNAME#checkpoint_epoch}"
        EPOCH_TAG="${EPOCH_TAG%.pth}"
        EPOCH_NUM=$((10#$EPOCH_TAG))
        EP_DIR="$EVAL_PER_EPOCH/epoch${EPOCH_TAG}"
        EP_EVAL_DIR="$EP_DIR/eval"
        # Use evaluation_results.json as the "already done" marker; falling back
        # to .txt for backward-compatibility with older runs (so old results,
        # which only emitted .txt, are still treated as completed).
        RES_JSON="$EP_EVAL_DIR/evaluation_results.json"
        RES_TXT="$EP_EVAL_DIR/evaluation_results.txt"

        # ── Resume gate 1: skip epochs below the requested start. ──
        if [ "$EPOCH_NUM" -lt "$EVAL_START_EPOCH" ]; then
            echo ""
            echo "  ── epoch ${EPOCH_TAG} ── SKIP (< EVAL_START_EPOCH=$EVAL_START_EPOCH)"
            continue
        fi

        # ── Resume gate 2: skip already-finished epochs (unless forced). ──
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

        # 7a. extract + MNN match for this epoch
        $PY -m finetune.extract_and_match_hf \
            --weights_dir "$WEIGHTS_DIR" \
            --checkpoint "$CKPT_E" \
            --lora_rank "$LORA_RANK" \
            --lora_alpha "$LORA_ALPHA" \
            --pairs "$EVAL_PAIRS" \
            --data_root "$DATA_ROOT" \
            --output_dir "$EP_DIR" \
            --img_size "$IMG_SIZE" \
            --eval_resize "$EVAL_W" "$EVAL_H" || {
                echo "  [warn] extract failed for epoch ${EPOCH_TAG}, skipping eval."
                continue
            }

        # 7b. evaluate
        $PY evaluate/evaluate_csv_essential.py \
            --input_pairs "$EVAL_PAIRS" \
            --input_dir "$DATA_ROOT" \
            --input_csv_dir "$EP_DIR" \
            --output_dir "$EP_EVAL_DIR" \
            --resize "$EVAL_W" "$EVAL_H" || {
                echo "  [warn] evaluate returned non-zero for epoch ${EPOCH_TAG}"
            }
    done

    # Rebuild summary.tsv from every existing per-epoch result file.
    echo ""
    echo "  Rebuilding summary.tsv from existing per-epoch results..."
    echo -e "epoch\tauc@5\tauc@10\tauc@20\tprecision\trecall" > "$SUMMARY_TSV"
    shopt -s nullglob
    # Prefer .json results (newer); pick up legacy .txt-only epochs as well.
    declare -A SEEN_EPOCHS=()
    EP_RES_FILES=()
    for EP_RES in "$EVAL_PER_EPOCH"/epoch*/eval/evaluation_results.json; do
        EP_TAG=$(basename "$(dirname "$(dirname "$EP_RES")")")
        SEEN_EPOCHS["$EP_TAG"]=1
        EP_RES_FILES+=( "$EP_RES" )
    done
    for EP_RES in "$EVAL_PER_EPOCH"/epoch*/eval/evaluation_results.txt; do
        EP_TAG=$(basename "$(dirname "$(dirname "$EP_RES")")")
        if [ -z "${SEEN_EPOCHS[$EP_TAG]:-}" ]; then
            EP_RES_FILES+=( "$EP_RES" )
        fi
    done
    for EP_RES in "${EP_RES_FILES[@]}"; do
        EP_TAG=$(basename "$(dirname "$(dirname "$EP_RES")")")
        EP_TAG="${EP_TAG#epoch}"
        $PY - "$EP_RES" "$EP_TAG" "$SUMMARY_TSV" <<'PY'
import re, sys
res_path, epoch_tag, summary_path = sys.argv[1:4]
text = open(res_path, "r", errors="ignore").read()

def grab(pattern, default="-"):
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else default

auc5  = grab(r"AUC@5[^0-9\-]*([0-9.]+)")
auc10 = grab(r"AUC@10[^0-9\-]*([0-9.]+)")
auc20 = grab(r"AUC@20[^0-9\-]*([0-9.]+)")
prec  = grab(r"Precision[^0-9\-]*([0-9.]+)")
rec   = grab(r"Recall[^0-9\-]*([0-9.]+)")

with open(summary_path, "a") as f:
    f.write(f"{epoch_tag}\t{auc5}\t{auc10}\t{auc20}\t{prec}\t{rec}\n")
PY
    done
    shopt -u nullglob

    echo ""
    echo "  Per-epoch summary written to: $SUMMARY_TSV"
    echo "  ──────── summary preview ────────"
    column -t -s $'\t' "$SUMMARY_TSV" || cat "$SUMMARY_TSV"
fi
else
    echo ""
    echo "[7/7] SKIPPED"
fi

echo ""
echo "===================================================================="
echo "  DONE.  ($MODEL)"
echo "  Zero-shot results : $EVAL_ZS/evaluation_results.json"
echo "  Fine-tuned results: $EVAL_FT/evaluation_results.json"
echo "  Per-epoch summary : $SUMMARY_TSV"
echo "===================================================================="
