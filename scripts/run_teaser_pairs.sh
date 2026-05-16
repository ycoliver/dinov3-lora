#!/usr/bin/env bash
# ----------------------------------------------------------------------
# scripts/run_teaser_pairs.sh
#
# Batch-render SuperGlue-style matching teaser figures for the first
# few NAVI test pairs. By default each pair produces TWO PNGs under
# output/teaser_pairs/ : one with the zero-shot DINOv3 baseline and one
# with the LoRA-finetuned checkpoint, ready to be placed side-by-side
# in the paper.
#
# Usage (run from the repo root, e.g. /root/autodl-tmp/cv-project):
#
#     bash scripts/run_teaser_pairs.sh                 # baseline + LoRA (default)
#     bash scripts/run_teaser_pairs.sh --lora_only     # only LoRA
#     bash scripts/run_teaser_pairs.sh --also_zeroshot # explicit; same as default
#
# Optional environment overrides:
#     MODEL        : 'small' | 'middle'      (default: small)
#     DATA_ROOT    : NAVI image root         (default: datasets/navi_v1.5)
#     WEIGHTS_DIR  : HF DINOv3 ckpt dir      (default: dinov3_weights/dinov3-${MODEL})
#     CKPT         : LoRA checkpoint path    (default: output/navi_${MODEL}/lora_ckpt/checkpoint_latest.pth)
#     OUT_DIR      : output directory        (default: output/teaser_pairs)
#     IMG_SIZE     : model input resolution  (default: 448)
#     TOPK         : keep top-K MNN matches  (default: 60)
# ----------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-small}"
DATA_ROOT="${DATA_ROOT:-datasets/navi_v1.5}"
WEIGHTS_DIR="${WEIGHTS_DIR:-dinov3_weights/dinov3-${MODEL}}"
CKPT="${CKPT:-output/navi_${MODEL}/lora_ckpt/checkpoint_latest.pth}"
OUT_DIR="${OUT_DIR:-output/teaser_pairs}"
IMG_SIZE="${IMG_SIZE:-448}"
# If OUT_DIR is a relative path, anchor it to the repo root so it lands at
# <repo>/output/teaser_pairs (e.g. /root/autodl-tmp/cv-project/output/teaser_pairs).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
if [[ "$OUT_DIR" != /* ]]; then
    OUT_DIR="$PROJECT_ROOT/$OUT_DIR"
fi
TOPK="${TOPK:-60}"

echo "[teaser] MODEL=$MODEL"
echo "[teaser] DATA_ROOT=$DATA_ROOT"
echo "[teaser] WEIGHTS_DIR=$WEIGHTS_DIR"
echo "[teaser] CKPT=$CKPT"
echo "[teaser] OUT_DIR=$OUT_DIR"

# By default we render BOTH zero-shot baseline and LoRA-finetuned
# matches, so each pair yields two PNGs (teaser_zs_*.png + teaser_lora_*.png)
# that can be placed side-by-side in the paper.
# Pass --lora_only to disable the zero-shot baseline.
ALSO_ZS=1
for arg in "$@"; do
    if [[ "$arg" == "--also_zeroshot" ]]; then
        ALSO_ZS=1
    elif [[ "$arg" == "--lora_only" ]]; then
        ALSO_ZS=0
    fi
done

mkdir -p "$OUT_DIR"

# Sanity checks
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "[teaser] ERROR: DATA_ROOT '$DATA_ROOT' does not exist." >&2
    echo "[teaser] Hint: override it, e.g.  DATA_ROOT=/abs/path/to/navi_v1.5 bash $0" >&2
    exit 1
fi
if [[ ! -d "$WEIGHTS_DIR" ]] || [[ ! -f "$WEIGHTS_DIR/config.json" ]]; then
    echo "[teaser] ERROR: WEIGHTS_DIR '$WEIGHTS_DIR' is not a valid HF model dir" >&2
    echo "          (expected a directory containing config.json, like" >&2
    echo "           dinov3_weights/dinov3-small/  or  dinov3_weights/dinov3-middle/)." >&2
    echo "[teaser] Hint: WEIGHTS_DIR=dinov3_weights/dinov3-small bash $0" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "[teaser] ERROR: checkpoint '$CKPT' not found." >&2
    echo "[teaser] Hint: CKPT=output/navi_${MODEL}/lora_ckpt/checkpoint_latest.pth bash $0" >&2
    exit 1
fi

# (tag, image_a_rel, image_b_rel)
PAIRS=(
  "circo_fish_holder_004_020|circo_fish_toothbrush_holder_14995988/multiview-11-ipad_5/images/004.jpg|circo_fish_toothbrush_holder_14995988/multiview-11-ipad_5/images/020.jpg"
  "dino_5_020_049|dino_5/multiview-13-canon_t4i/images/020.jpg|dino_5/multiview-13-canon_t4i/images/049.jpg"
  "dollhouse_sink_002_010|3d_dollhouse_sink/multiview-01-pixel_5/images/002.jpg|3d_dollhouse_sink/multiview-01-pixel_5/images/010.jpg"
  "dollhouse_sink_005_009|3d_dollhouse_sink/multiview-01-pixel_5/images/005.jpg|3d_dollhouse_sink/multiview-01-pixel_5/images/009.jpg"
  "circo_fish_pixel7_027_042|circo_fish_toothbrush_holder_14995988/multiview-15-pixel_7/images/027.jpg|circo_fish_toothbrush_holder_14995988/multiview-15-pixel_7/images/042.jpg"
)

run_one () {
    local tag="$1"
    local img_a="$2"
    local img_b="$3"
    local mode="$4"        # "lora" | "zs"
    local extra_args="$5"  # extra CLI args (e.g. checkpoint)

    local a_path="$DATA_ROOT/$img_a"
    local b_path="$DATA_ROOT/$img_b"
    local out_path="$OUT_DIR/teaser_${mode}_${tag}.png"

    if [[ ! -f "$a_path" || ! -f "$b_path" ]]; then
        echo "[teaser] skip $tag ($mode): image not found"
        echo "         A: $a_path"
        echo "         B: $b_path"
        return 0
    fi

    echo "------------------------------------------------------------"
    echo "[teaser] $mode | $tag"
    echo "         A: $a_path"
    echo "         B: $b_path"
    echo "         -> $out_path"

    # shellcheck disable=SC2086
    python -m presentation.plot_match_teaser \
        --weights_dir "$WEIGHTS_DIR" \
        --image_a "$a_path" \
        --image_b "$b_path" \
        --img_size "$IMG_SIZE" \
        --topk "$TOPK" \
        --title "$tag ($mode)" \
        --output "$out_path" \
        $extra_args
}

for spec in "${PAIRS[@]}"; do
    IFS='|' read -r tag img_a img_b <<< "$spec"
    run_one "$tag" "$img_a" "$img_b" "lora" "--checkpoint $CKPT"
    if [[ "$ALSO_ZS" -eq 1 ]]; then
        run_one "$tag" "$img_a" "$img_b" "zs" ""
    fi
done

echo "============================================================"
echo "[teaser] done. PNGs are in: $OUT_DIR"
ls -1 "$OUT_DIR" | sed 's/^/    /'
