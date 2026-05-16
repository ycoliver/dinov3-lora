#!/usr/bin/env python3
"""
Plot the Phase-1 contrastive-loss convergence curve.

Reproduces the left half of Figure 2 in main.tex (§3.2):
the contrastive loss collapses toward the theoretical random-guessing
limit  L*  =  ln(N_negatives + 1), proving that the feature space has
collapsed to a single point on the hypersphere.

For HardInfoNCE+SafeRadius (the "Phase 3" recipe), L* is replaced by
ln(K + 1) where K = max(int(N * hard_neg_ratio), 1). Pass --K manually
in that case.

Usage (defaults assume the pipeline output layout):
    python presentation/plot_loss_convergence.py output/navi_small_phase1/ckpt
    python presentation/plot_loss_convergence.py output/navi_middle_phase1/ckpt

    # Override the asymptote (e.g. compare to ln(129) = 4.86 used in main.tex)
    python presentation/plot_loss_convergence.py output/navi_small_phase1/ckpt --K 128

    # Plot small + middle on the same axes
    python presentation/plot_loss_convergence.py \
        output/navi_small_phase1/ckpt output/navi_middle_phase1/ckpt \
        --labels small middle \
        --output presentation/figures/loss_convergence.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_log(ckpt_dir: Path) -> Path:
    """Locate training_log.json inside a checkpoint directory."""
    p = ckpt_dir / "training_log.json"
    if p.exists():
        return p
    # Some pipelines drop the log next to the ckpts under a sibling dir.
    candidates = list(ckpt_dir.rglob("training_log.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"training_log.json not found under {ckpt_dir}")


def load_curve(log_path: Path) -> dict:
    log = json.loads(log_path.read_text())
    if not isinstance(log, list) or not log:
        raise ValueError(f"Empty / malformed log: {log_path}")
    epochs = [r["epoch"] for r in log]
    contrastive = [r.get("contrastive", r.get("loss", float("nan"))) for r in log]
    total = [r.get("loss", float("nan")) for r in log]
    return {
        "epochs": epochs,
        "contrastive": contrastive,
        "total": total,
        "config": log[0].get("args", {}),  # not always present
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot Phase-1 loss convergence vs. ln(K+1) baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "ckpt_dirs",
        nargs="+",
        type=Path,
        help="One or more directories containing training_log.json "
             "(typically output/navi_<model>_phase1/ckpt).",
    )
    p.add_argument(
        "--labels", nargs="*", default=None,
        help="Curve labels (one per ckpt_dir). Defaults to the dir names.",
    )
    p.add_argument(
        "--K", type=int, default=None,
        help="Number of negatives per anchor used by InfoNCE. The asymptote "
             "drawn is ln(K+1). If omitted, falls back to ln(N+1) where N is "
             "automatically inferred (or 4.86 ≈ ln(129) as a safe fallback "
             "matching main.tex Phase 1).",
    )
    p.add_argument(
        "--output", "-o", type=Path,
        default=Path("presentation/figures/loss_convergence.png"),
        help="Output PNG path.",
    )
    p.add_argument(
        "--title", type=str,
        default="Phase 1: Projection Head + InfoNCE — Loss Convergence",
    )
    p.add_argument(
        "--ymin", type=float, default=None,
        help="Optional lower y-limit (defaults to auto).",
    )
    p.add_argument(
        "--ymax", type=float, default=None,
        help="Optional upper y-limit (defaults to auto).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.labels and len(args.labels) != len(args.ckpt_dirs):
        raise SystemExit(
            f"--labels has {len(args.labels)} items but {len(args.ckpt_dirs)} ckpt_dirs"
        )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    last_max_epoch = 0
    final_losses = []

    for i, ckpt_dir in enumerate(args.ckpt_dirs):
        log_path = find_log(ckpt_dir)
        curve = load_curve(log_path)
        label = (
            args.labels[i] if args.labels
            else ckpt_dir.parent.name + "/" + ckpt_dir.name
        )
        color = palette[i % len(palette)]

        ax.plot(
            curve["epochs"], curve["contrastive"],
            marker="o", markersize=4, linewidth=1.6,
            color=color, label=label,
        )
        last_max_epoch = max(last_max_epoch, max(curve["epochs"]))
        final_losses.append((label, curve["contrastive"][-1]))
        print(f"[{label}] log = {log_path}")
        print(f"  epochs    : {curve['epochs'][0]} … {curve['epochs'][-1]}")
        print(f"  contrast. : {curve['contrastive'][0]:.4f} → {curve['contrastive'][-1]:.4f}")

    # ── Asymptote: theoretical random-guessing limit. ─────────────────
    if args.K is not None:
        asymptote = math.log(args.K + 1)
        asym_lbl = rf"$\ln(K+1) = \ln({args.K + 1}) \approx {asymptote:.3f}$"
    else:
        # Default to ln(129) ≈ 4.86 — the value cited in main.tex Phase 1.
        asymptote = math.log(129)
        asym_lbl = rf"$\ln(129) \approx {asymptote:.3f}$  (collapse limit, K=128)"

    ax.axhline(
        asymptote, color="#888888", linestyle="--", linewidth=1.4,
        label=asym_lbl,
    )

    # Annotate how close we got to the collapse limit.
    for label, final in final_losses:
        gap = final - asymptote
        print(f"  → {label}: final − asymptote = {gap:+.4f}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive (InfoNCE) loss")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(args.ymin, args.ymax)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
