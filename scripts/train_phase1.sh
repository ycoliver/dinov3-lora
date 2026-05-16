#!/usr/bin/env bash
# =====================================================================
#  Phase-1 entry point — SMALL model.
#  ────────────────────────────────────────────────────────────────────
#  Reproduces main.tex §3.2 "Projection Head + InfoNCE — Catastrophic
#  Collapse": frozen DINOv3-small backbone + random ProjectionHead
#  trained with PLAIN InfoNCE (no Safe Radius). Expected outcome:
#  contrastive loss → ln(N_neg + 1), Precision collapses vs. zero-shot.
#
#  Output : output/navi_small_phase1/
#  Compare against the LoRA-small zero-shot baseline already in:
#           output/navi_small/eval_zeroshot/
#
#  Usage:
#    bash scripts/train_phase1.sh                   # 15 epochs (default)
#    bash scripts/train_phase1.sh 30                # 30 epochs
#
#  Resume / step-gating:
#    START_STEP=2 bash scripts/train_phase1.sh           # skip dep check
#    START_STEP=5 bash scripts/train_phase1.sh           # only per-epoch eval
#    EVAL_START_EPOCH=5 START_STEP=5 bash scripts/train_phase1.sh
#    EVAL_FORCE=1       START_STEP=5 bash scripts/train_phase1.sh
#
#  Hyper-param overrides (env vars):
#    BATCH_SIZE=8  LR=1e-4  PROJ_DIM=256  IMG_SIZE=448 \
#        bash scripts/train_phase1.sh 15
# =====================================================================

EPOCHS=${1:-15}
MODEL=small

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib_phase1_pipeline.sh
source "$SCRIPT_DIR/lib_phase1_pipeline.sh"
