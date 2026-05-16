#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 一键运行诊断分析（论文 Layer 1 / 2 / 3 / 4 的定量+定性证据）。
#
# 包含 4 个子任务：
#   1) per_epoch  → presentation/plot_per_epoch.py
#                   读取 output/navi_*/eval_per_epoch/summary.tsv
#                   画 Precision / AUC@10 / AUC@20 三联图
#                   叠加 zero-shot baseline 横线
#   2) pca        → presentation/pca_visualizer.py
#                   PCA(1024→3) → RGB 贴回原图
#                   Zero-Shot vs LoRA 同色彩空间对比
#   3) features   → presentation/diagnostics_features.py
#                   intra-image cos / effective rank /
#                   positive-pair cos / neighbour dominance
#   4) layer4     → presentation/diagnostics_layer4.py
#                   Δprecision vs Δpose-error 散点 + 相关系数
#
# 默认路径（可被环境变量覆盖）:
#   PROJECT_ROOT            默认 /root/autodl-tmp/cv-project
#   SMALL_WEIGHTS_DIR       $PROJECT_ROOT/dinov3_weights/dinov3-small
#   MIDDLE_WEIGHTS_DIR      $PROJECT_ROOT/dinov3_weights/dinov3-middle
#   SMALL_CKPT              $PROJECT_ROOT/output/navi_small/lora_ckpt/checkpoint_latest.pth
#   MIDDLE_CKPT             $PROJECT_ROOT/output/navi_middle/lora_ckpt/checkpoint_latest.pth
#   IMAGE_DIR               $PROJECT_ROOT/datasets/navi_v1.5
#   PAIRS_FILE              $PROJECT_ROOT/datasets/navi_test_pairs.txt
#                           (用于 PCA / features 自动挑选样本图；优先于 IMAGE_DIR 递归查找)
#   PCA_IMAGE               (默认: 从 PAIRS_FILE 第一行解析第一张可用图)
#   LAST_EPOCH_TAG          默认 014  (即 epoch014/eval/evaluation_pairs.csv)
#
# 用法示例:
#   bash scripts/run_diagnostics.sh                           # 全部任务，small + middle
#   bash scripts/run_diagnostics.sh --model small             # 只 small
#   bash scripts/run_diagnostics.sh --task pca --model middle # 只 PCA / 只 middle
#   bash scripts/run_diagnostics.sh --task per_epoch          # 只画曲线
#   PCA_IMAGE=/some/path.jpg bash scripts/run_diagnostics.sh --task pca
#
# 退出码:
#   0 = 成功；非 0 = 任一子任务失败
# ─────────────────────────────────────────────────────────────────────
set -e
set -u
set -o pipefail

# ─── Defaults ──────────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/cv-project}"
SMALL_WEIGHTS_DIR="${SMALL_WEIGHTS_DIR:-$PROJECT_ROOT/dinov3_weights/dinov3-small}"
MIDDLE_WEIGHTS_DIR="${MIDDLE_WEIGHTS_DIR:-$PROJECT_ROOT/dinov3_weights/dinov3-middle}"
SMALL_CKPT="${SMALL_CKPT:-$PROJECT_ROOT/output/navi_small/lora_ckpt/checkpoint_latest.pth}"
MIDDLE_CKPT="${MIDDLE_CKPT:-$PROJECT_ROOT/output/navi_middle/lora_ckpt/checkpoint_latest.pth}"
IMAGE_DIR="${IMAGE_DIR:-$PROJECT_ROOT/datasets/navi_v1.5}"
PAIRS_FILE="${PAIRS_FILE:-$PROJECT_ROOT/datasets/navi_test_pairs.txt}"
PCA_IMAGE="${PCA_IMAGE:-}"
LAST_EPOCH_TAG="${LAST_EPOCH_TAG:-014}"
NUM_IMAGES="${NUM_IMAGES:-30}"
IMG_SIZE="${IMG_SIZE:-448}"
SEED="${SEED:-0}"

OUT_BASE="$PROJECT_ROOT/presentation/result"
SMALL_OUT="$OUT_BASE/diag_small"
MIDDLE_OUT="$OUT_BASE/diag_middle"

# ─── CLI ───────────────────────────────────────────────────────────
TASK="all"        # all | per_epoch | pca | features | layer4
MODEL="both"      # both | small | middle

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)  TASK="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --image) PCA_IMAGE="$2"; shift 2 ;;
        --num_images) NUM_IMAGES="$2"; shift 2 ;;
        --epoch_tag) LAST_EPOCH_TAG="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$TASK" != "all" && "$TASK" != "per_epoch" && "$TASK" != "pca" \
      && "$TASK" != "features" && "$TASK" != "layer4" ]]; then
    echo "[error] --task must be one of: all|per_epoch|pca|features|layer4" >&2
    exit 2
fi
if [[ "$MODEL" != "both" && "$MODEL" != "small" && "$MODEL" != "middle" ]]; then
    echo "[error] --model must be one of: both|small|middle" >&2
    exit 2
fi

# ─── helpers ───────────────────────────────────────────────────────
log()    { echo -e "\033[1;36m[run]\033[0m $*"; }
ok()     { echo -e "\033[1;32m[ok]\033[0m  $*"; }
warn()   { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
abort()  { echo -e "\033[1;31m[abort]\033[0m $*" >&2; exit 1; }

require_file() {
    local p="$1" name="$2"
    if [[ ! -e "$p" ]]; then
        warn "$name not found: $p  (will skip dependent steps)"
        return 1
    fi
    return 0
}

# Resolve a usable image path, in priority order:
#   1) explicit --image / $PCA_IMAGE
#   2) first valid path appearing in $PAIRS_FILE  (joined with $IMAGE_DIR if relative)
#   3) recursive find under $IMAGE_DIR
resolve_pca_image() {
    if [[ -n "$PCA_IMAGE" ]]; then
        [[ -f "$PCA_IMAGE" ]] || abort "PCA_IMAGE not found: $PCA_IMAGE"
        echo "$PCA_IMAGE"; return
    fi

    # ── (2) try pairs file ────────────────────────────────────────
    if [[ -f "$PAIRS_FILE" ]]; then
        # Each non-empty line is whitespace-separated; take the first 2 fields
        # (img_a, img_b) of the first non-comment line. Fields can be relative.
        local cand
        cand=$(awk 'NF && $1 !~ /^#/ {print $1; print $2; exit}' "$PAIRS_FILE" 2>/dev/null)
        while IFS= read -r p; do
            [[ -z "$p" ]] && continue
            if [[ -f "$p" ]]; then echo "$p"; return; fi
            if [[ -f "$IMAGE_DIR/$p" ]]; then echo "$IMAGE_DIR/$p"; return; fi
            # Some pairs files store paths relative to PROJECT_ROOT
            if [[ -f "$PROJECT_ROOT/$p" ]]; then echo "$PROJECT_ROOT/$p"; return; fi
        done <<< "$cand"
    fi

    # ── (3) recursive find ────────────────────────────────────────
    local img
    img=$(find "$IMAGE_DIR" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) 2>/dev/null | head -n 1)
    [[ -n "$img" ]] || abort "No image found via PAIRS_FILE=$PAIRS_FILE or under IMAGE_DIR=$IMAGE_DIR. Pass --image <path>."
    echo "$img"
}

cd "$PROJECT_ROOT" || abort "PROJECT_ROOT does not exist: $PROJECT_ROOT"
mkdir -p "$SMALL_OUT" "$MIDDLE_OUT"

# ─── Task 1: per-epoch curves ──────────────────────────────────────
run_per_epoch() {
    log "TASK 1/4: per-epoch curves"
    local args=()
    if [[ "$MODEL" == "both" || "$MODEL" == "small" ]]; then
        args+=("$PROJECT_ROOT/output/navi_small")
    fi
    if [[ "$MODEL" == "both" || "$MODEL" == "middle" ]]; then
        args+=("$PROJECT_ROOT/output/navi_middle")
    fi

    if [[ ${#args[@]} -eq 0 ]]; then
        warn "no model selected for per_epoch"; return 0
    fi

    # Validate summary.tsv presence
    local any=0
    for r in "${args[@]}"; do
        if [[ -f "$r/eval_per_epoch/summary.tsv" ]]; then
            any=1
        else
            warn "summary.tsv missing for $r — run scripts/build_eval_summary.py first"
        fi
    done
    [[ $any -eq 1 ]] || { warn "skip per_epoch: no summary.tsv found"; return 0; }

    if [[ "$MODEL" == "both" ]]; then
        python presentation/plot_per_epoch.py "${args[@]}" \
            --labels small middle \
            --out "$OUT_BASE/per_epoch_small_vs_middle.png" \
            --title "LoRA per-epoch metrics — small vs middle"
        ok "per-epoch (both) → $OUT_BASE/per_epoch_small_vs_middle.png"
    else
        local label="$MODEL"
        local out="$OUT_BASE/per_epoch_${label}.png"
        python presentation/plot_per_epoch.py "${args[@]}" \
            --labels "$label" --out "$out" \
            --title "LoRA per-epoch metrics — $label"
        ok "per-epoch ($label) → $out"
    fi
}

# ─── Task 2: PCA visualizer ────────────────────────────────────────
run_pca_one() {
    local model="$1" weights="$2" ckpt="$3" out_dir="$4"
    local img; img=$(resolve_pca_image)
    require_file "$weights" "weights_dir [$model]" || return 0

    local ckpt_arg=()
    if require_file "$ckpt" "lora ckpt [$model]"; then
        ckpt_arg=(--checkpoint "$ckpt")
    fi

    log "PCA [$model]  image=$img"
    python presentation/pca_visualizer.py \
        --model "$model" \
        --weights_dir "$weights" \
        --image "$img" \
        --img_size "$IMG_SIZE" \
        --shared_pca \
        --out_dir "$out_dir" \
        "${ckpt_arg[@]}"
    ok "PCA [$model] → $out_dir"
}

run_pca() {
    log "TASK 2/4: PCA visualization"
    [[ "$MODEL" == "both" || "$MODEL" == "small"  ]] && \
        run_pca_one small  "$SMALL_WEIGHTS_DIR"  "$SMALL_CKPT"  "$SMALL_OUT"
    [[ "$MODEL" == "both" || "$MODEL" == "middle" ]] && \
        run_pca_one middle "$MIDDLE_WEIGHTS_DIR" "$MIDDLE_CKPT" "$MIDDLE_OUT"
}

# ─── Task 3: feature-space diagnostics ─────────────────────────────
run_features_one() {
    local model="$1" weights="$2" ckpt="$3" out_dir="$4"
    require_file "$weights"   "weights_dir [$model]" || return 0
    require_file "$IMAGE_DIR" "IMAGE_DIR"            || return 0

    local ckpt_arg=()
    if require_file "$ckpt" "lora ckpt [$model]"; then
        ckpt_arg=(--checkpoint "$ckpt")
    fi

    log "features [$model]  num_images=$NUM_IMAGES"
    python presentation/diagnostics_features.py \
        --model "$model" \
        --weights_dir "$weights" \
        --image_dir "$IMAGE_DIR" \
        --num_images "$NUM_IMAGES" \
        --img_size "$IMG_SIZE" \
        --seed "$SEED" \
        --out_dir "$out_dir" \
        "${ckpt_arg[@]}"
    ok "features [$model] → $out_dir"
}

run_features() {
    log "TASK 3/4: feature-space diagnostics"
    [[ "$MODEL" == "both" || "$MODEL" == "small"  ]] && \
        run_features_one small  "$SMALL_WEIGHTS_DIR"  "$SMALL_CKPT"  "$SMALL_OUT"
    [[ "$MODEL" == "both" || "$MODEL" == "middle" ]] && \
        run_features_one middle "$MIDDLE_WEIGHTS_DIR" "$MIDDLE_CKPT" "$MIDDLE_OUT"
}

# ─── Task 4: Layer 4 scatter ───────────────────────────────────────
run_layer4_one() {
    local model="$1" out_dir="$2"
    local zs_csv="$PROJECT_ROOT/output/navi_${model}/eval_zeroshot/evaluation_pairs.csv"
    local lora_csv="$PROJECT_ROOT/output/navi_${model}/eval_per_epoch/epoch${LAST_EPOCH_TAG}/eval/evaluation_pairs.csv"

    require_file "$zs_csv"   "zero-shot pairs csv [$model]" || return 0
    require_file "$lora_csv" "LoRA epoch${LAST_EPOCH_TAG} pairs csv [$model]" || return 0

    log "layer4 [$model]  epoch=$LAST_EPOCH_TAG"
    python presentation/diagnostics_layer4.py \
        --zs_csv "$zs_csv" \
        --lora_csv "$lora_csv" \
        --label "$model" \
        --out_dir "$out_dir"
    ok "layer4 [$model] → $out_dir"
}

run_layer4() {
    log "TASK 4/4: Layer-4 non-differentiable-gap scatter"
    [[ "$MODEL" == "both" || "$MODEL" == "small"  ]] && run_layer4_one small  "$SMALL_OUT"
    [[ "$MODEL" == "both" || "$MODEL" == "middle" ]] && run_layer4_one middle "$MIDDLE_OUT"
}

# ─── Dispatch ──────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Diagnostics runner"
echo "  PROJECT_ROOT = $PROJECT_ROOT"
echo "  TASK         = $TASK"
echo "  MODEL        = $MODEL"
echo "═══════════════════════════════════════════════════════════════"

case "$TASK" in
    per_epoch) run_per_epoch ;;
    pca)       run_pca ;;
    features)  run_features ;;
    layer4)    run_layer4 ;;
    all)
        run_per_epoch
        run_pca
        run_features
        run_layer4
        ;;
esac

ok "all requested diagnostics finished. Outputs under $OUT_BASE"
