#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_navi_pairs.sh
#
# One-shot generator for NAVI train + test pairs.
#
# Why this script:
#   We must ensure NO image is shared between train and test pairs (otherwise
#   the model would have already seen those views during fine-tuning, leading
#   to inflated metrics). NAVI v1.5's annotations.json does NOT contain
#   per-image split labels, so we rely on a deterministic SCENE-LEVEL split:
#   `generate_train_pairs.py` shuffles all multiview_* scenes with a fixed
#   seed and reserves the last 15% as test scenes. Because every image of a
#   scene goes entirely into one side, train/test are image-disjoint by
#   construction.
#
# Outputs:
#   finetune/navi_train_pairs.txt   ← pairs from the ~85% train scenes
#   datasets/navi_test_pairs.txt    ← pairs from the ~15% held-out scenes
#
# After running, the script also performs an explicit overlap check at the end.
# ─────────────────────────────────────────────────────────────────────────────
set -e

DATA_ROOT="${1:-full_dataset/navi_v1.5}"
TRAIN_OUT="${2:-finetune/navi_train_pairs.txt}"
TEST_OUT="${3:-datasets/navi_test_pairs.txt}"

# Per-scene pair limits (final totals are determined by the scene-level split,
# not by an arbitrary cap; NAVI itself does not specify a "test set size").
TRAIN_PER_SCENE=20
TEST_PER_SCENE=20

mkdir -p "$(dirname "$TRAIN_OUT")"
mkdir -p "$(dirname "$TEST_OUT")"

echo "════════════════════════════════════════════════════════════════"
echo "  Building NAVI train/test pairs (scene-level disjoint split)"
echo "════════════════════════════════════════════════════════════════"
echo "  data_root : $DATA_ROOT"
echo "  train_out : $TRAIN_OUT"
echo "  test_out  : $TEST_OUT"
echo

echo "[1/3] Generating TRAIN pairs (scene-level split, train scenes)..."
python -m finetune.generate_train_pairs \
    --data_root "$DATA_ROOT" \
    --split train \
    --output "$TRAIN_OUT" \
    --max_pairs_per_scene "$TRAIN_PER_SCENE" \
    --min_angle 10 --max_angle 90 \
    --seed 42

echo
echo "[2/3] Generating TEST pairs (scene-level split, held-out scenes)..."
python -m finetune.generate_train_pairs \
    --data_root "$DATA_ROOT" \
    --split test \
    --output "$TEST_OUT" \
    --max_pairs_per_scene "$TEST_PER_SCENE" \
    --min_angle 10 --max_angle 90 \
    --seed 42

echo
echo "[3/3] Verifying train/test image disjointness..."
python - <<PY
train_imgs = set()
test_imgs  = set()

with open("$TRAIN_OUT") as f:
    for line in f:
        toks = line.strip().split()
        if len(toks) >= 2:
            train_imgs.add(toks[0]); train_imgs.add(toks[1])

with open("$TEST_OUT") as f:
    for line in f:
        toks = line.strip().split()
        if len(toks) >= 2:
            test_imgs.add(toks[0]); test_imgs.add(toks[1])

overlap = train_imgs & test_imgs
print(f"  train images : {len(train_imgs)}")
print(f"  test  images : {len(test_imgs)}")
print(f"  overlap      : {len(overlap)}")

if overlap:
    print("  ✗  LEAK DETECTED — there are shared images between train and test!")
    for s in list(overlap)[:5]:
        print("    ", s)
    raise SystemExit(1)
else:
    print("  ✓  no overlap — train/test are image-disjoint")
PY

echo
echo "════════════════════════════════════════════════════════════════"
echo "  DONE."
echo "    TRAIN : $TRAIN_OUT"
echo "    TEST  : $TEST_OUT"
echo "════════════════════════════════════════════════════════════════"
