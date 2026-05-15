#!/usr/bin/env bash
# =====================================================================
#  One-click LoRA fine-tune + evaluation pipeline for DINOv3 (HF)
#  ────────────────────────────────────────────────────────────────────
#  This is the SMALL-model entry point.
#  For the middle model (≈300M / 1.21G), use:
#      bash scripts/train_oneclick_middle.sh [EPOCHS]
#
#  Dataset : NAVI v1.5
#  Model   : 'small'  (~86M, dinov3_weights/dinov3-small/)
#  Output  : output/navi_small/
#
#  Usage:
#    bash scripts/train_oneclick.sh                      # 15 epochs (default)
#    bash scripts/train_oneclick.sh 30                   # 30 epochs
#
#  Skip earlier steps (resume the pipeline mid-way):
#    START_STEP=4 bash scripts/train_oneclick.sh                 # start from training
#    START_STEP=5 bash scripts/train_oneclick.sh                 # eval latest ckpt only
#    START_STEP=7 bash scripts/train_oneclick.sh                 # only per-epoch eval
#    START_STEP=4 END_STEP=4 bash scripts/train_oneclick.sh      # train only
#
#  Per-epoch eval resume controls (STEP 7):
#    EVAL_START_EPOCH=9 START_STEP=7 bash scripts/train_oneclick.sh
#    EVAL_FORCE=1       START_STEP=7 bash scripts/train_oneclick.sh
#    (default: skip any epoch whose evaluation_results.json already exists)
#
#  Hyper-param overrides (any combination, shown with their pipeline defaults):
#    BATCH_SIZE=8   LR=1e-4   LORA_RANK=4   LORA_ALPHA=8.0   IMG_SIZE=448 \
#        bash scripts/train_oneclick.sh 15
#  These can also be set inline below (see the block right after MODEL=...).
#
#  Pipeline implementation lives in scripts/lib_oneclick_pipeline.sh
# =====================================================================

EPOCHS=${1:-15}
MODEL=small

# ---------------------------------------------------------------------
#  Hyper-param tweak zone (uncomment + edit to override pipeline defaults)
#  ---------------------------------------------------------------------
#  Pipeline defaults (defined in lib_oneclick_pipeline.sh):
#    IMG_SIZE=448
#    BATCH_SIZE=8
#    LR=1e-4
#    LORA_RANK=4
#    LORA_ALPHA=8.0
#
#  Either uncomment below, or pass as env vars on the command line.
#  Env vars passed on the command line take precedence over both this
#  block and the lib defaults.
# ---------------------------------------------------------------------
# export IMG_SIZE=448
# export BATCH_SIZE=8
# export LR=1e-4
# export LORA_RANK=4
# export LORA_ALPHA=8.0

# Source the shared pipeline. It picks up MODEL / EPOCHS from this scope
# and honours START_STEP / END_STEP / EVAL_START_EPOCH / EVAL_FORCE env vars.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib_oneclick_pipeline.sh
source "$SCRIPT_DIR/lib_oneclick_pipeline.sh"
