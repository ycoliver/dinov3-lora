#!/usr/bin/env bash
# =====================================================================
#  One-click LoRA fine-tune + evaluation pipeline for DINOv3 (HF)
#  ────────────────────────────────────────────────────────────────────
#  This is the MIDDLE-model entry point.
#  For the small model (~86M), use:
#      bash scripts/train_oneclick.sh [EPOCHS]
#
#  Dataset : NAVI v1.5
#  Model   : 'middle'  (≈300M / 1.21G safetensors,
#                       dinov3_weights/dinov3-middle/)
#  Output  : output/navi_middle/
#
#  Usage:
#    bash scripts/train_oneclick_middle.sh               # 15 epochs (default)
#    bash scripts/train_oneclick_middle.sh 30            # 30 epochs
#
#  Skip earlier steps (resume the pipeline mid-way):
#    START_STEP=4 bash scripts/train_oneclick_middle.sh              # start from training
#    START_STEP=5 bash scripts/train_oneclick_middle.sh              # eval latest ckpt only
#    START_STEP=7 bash scripts/train_oneclick_middle.sh              # only per-epoch eval
#    START_STEP=4 END_STEP=4 bash scripts/train_oneclick_middle.sh   # train only
#
#  Per-epoch eval resume controls (STEP 7):
#    EVAL_START_EPOCH=9 START_STEP=7 bash scripts/train_oneclick_middle.sh
#    EVAL_FORCE=1       START_STEP=7 bash scripts/train_oneclick_middle.sh
#    (default: skip any epoch whose evaluation_results.json already exists)
#
#  Pipeline implementation lives in scripts/lib_oneclick_pipeline.sh
# =====================================================================

EPOCHS=${1:-15}
MODEL=middle

# Source the shared pipeline. It picks up MODEL / EPOCHS from this scope
# and honours START_STEP / END_STEP / EVAL_START_EPOCH / EVAL_FORCE env vars.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib_oneclick_pipeline.sh
source "$SCRIPT_DIR/lib_oneclick_pipeline.sh"
