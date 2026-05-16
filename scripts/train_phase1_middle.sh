#!/usr/bin/env bash
# =====================================================================
#  Phase-1 entry point — MIDDLE model.
#  ────────────────────────────────────────────────────────────────────
#  Reproduces main.tex §3.2 "Projection Head + InfoNCE — Catastrophic
#  Collapse" on the middle (≈300M / 1.21G) DINOv3 backbone.
#
#  Output : output/navi_middle_phase1/
#  Compare against the LoRA-middle zero-shot baseline already in:
#           output/navi_middle/eval_zeroshot/
#
#  Usage:
#    bash scripts/train_phase1_middle.sh            # 15 epochs (default)
#    bash scripts/train_phase1_middle.sh 30         # 30 epochs
#
#  Hyper-param overrides (env vars; defaults are pipeline defaults):
#    BATCH_SIZE=4 LR=1e-4 PROJ_DIM=256 IMG_SIZE=448 \
#        bash scripts/train_phase1_middle.sh 15
#
#  Resume / step-gating:
#    START_STEP=5 bash scripts/train_phase1_middle.sh        # only per-epoch eval
#    EVAL_START_EPOCH=9 START_STEP=5 bash scripts/train_phase1_middle.sh
# =====================================================================

EPOCHS=${1:-15}
MODEL=middle

# Middle model is ~3-4× heavier; default to a smaller batch unless overridden.
export BATCH_SIZE=${BATCH_SIZE:-4}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib_phase1_pipeline.sh
source "$SCRIPT_DIR/lib_phase1_pipeline.sh"
