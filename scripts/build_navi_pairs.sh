#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_navi_pairs.sh
#
# One-shot generator for NAVI train + test pairs.
#
# Why this script:
#   We must ensure NO image is shared between train and test pairs (otherwise
#   the model would have already seen those views during fine-tuning, leading
#   to inflated metrics). NAVI v1.5 already labels every image with a
#   'split' field (train / val / test) inside annotations.json, so we just
#   reuse that authoritative labeling.
#
# Outputs:
#   finetune/navi_train_pairs.txt   ← only images with split=='train'
#   datasets/navi_test_pairs.txt    ← only images with split=='test'
#
# After running, train and test sets are image-disjoint by construction;
# the script also performs an explicit overlap check at the end.
# ─────────────────────────────────────────────────────────────────────────────
set -e

DATA_ROOT="${1:-full_dataset/navi_v1.5}"
TRAIN_OUT="${2:-finetune/navi_train_pairs.txt}"
TEST_OUT="${3:-datasets/navi_test_pairs.txt}"

# Target sizes
TRAIN_PER_SCENE=20      # ~5000+ train pairs
TEST_PER_SCENE=12       # ~3000 test pairs
TEST_TOTAL_CAP=3000     # hard cap

mkdir -p "$(dirname "$TRAIN_OUT")"
mkdir -p "$(dirname "$TEST_OUT")"

echo "════════════════════════════════════════════════════════════════"
echo "  Building NAVI train/test pairs (image-disjoint by NAVI split)"
echo "════════════════════════════════════════════════════════════════"
echo "  data_root : $DATA_ROOT"
echo "  train_out : $TRAIN_OUT"
echo "  test_out  : $TEST_OUT"
echo

echo "[1/3] Generating TRAIN pairs (split='train')..."
python -m finetune.generate_train_pairs \
    --data_root "$DATA_ROOT" \
    --split train \
    --output "$TRAIN_OUT" \
    --max_pairs_per_scene "$TRAIN_PER_SCENE" \
    --min_angle 10 --max_angle 90 \
    --seed 42

echo
echo "[2/3] Generating TEST pairs (split='test', cap=${TEST_TOTAL_CAP})..."
python -m finetune.generate_train_pairs \
    --data_root "$DATA_ROOT" \
    --split test \
    --output "$TEST_OUT" \
    --max_pairs_per_scene "$TEST_PER_SCENE" \
    --max_total_pairs "$TEST_TOTAL_CAP" \
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
