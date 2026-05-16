"""Per-epoch evaluation curve（论文 Fig.\\ref{fig:scannet_chronological} 的离散版）。

读取 ``scripts/build_eval_summary.py`` 产出的 ``summary.tsv``，画出
Precision、AUC@10、AUC@20 三条折线，并以 zero-shot baseline 作为水平参考线。

由于现在每 3 个 epoch 评估一次（epoch 2/5/8/11/14），曲线点之间的间隔
就是 3 个 epoch；脚本不假定连续，只按 summary.tsv 里实际出现的 epoch 标签
画点。

CLI 例：
  # 单模型
  python presentation/plot_per_epoch.py output/navi_small \
      --out presentation/result/navi_small_per_epoch.png

  # 双模型对比（small vs middle）
  python presentation/plot_per_epoch.py output/navi_small output/navi_middle \
      --labels small middle \
      --out presentation/result/navi_small_vs_middle_per_epoch.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = ("precision", "auc@10", "auc@20")
COLORS = {"small": "#1f77b4", "middle": "#d62728"}
DEFAULT_FALLBACK_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _read_summary(path: Path) -> tuple[dict | None, list[tuple[int, dict]]]:
    """Parse a summary.tsv file.

    Returns:
        baseline_metrics: dict | None
        epochs: list of (epoch_int, metrics_dict) sorted by epoch_int
    """
    if not path.exists():
        raise FileNotFoundError(f"summary.tsv not found: {path}")

    baseline = None
    rows: list[tuple[int, dict]] = []

    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row["epoch"].strip()

            def _val(k: str):
                s = row.get(k, "-").strip()
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return None

            metrics = {k: _val(k) for k in METRICS}

            if label == "baseline":
                baseline = metrics
            else:
                # Try to parse as int (e.g. "002", "5")
                try:
                    ep = int(label)
                except ValueError:
                    continue
                rows.append((ep, metrics))

    rows.sort(key=lambda r: r[0])
    return baseline, rows


def _resolve_summary(model_root_or_tsv: Path) -> Path:
    p = model_root_or_tsv.expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        # 1) <root>/eval_per_epoch/summary.tsv
        cand = p / "eval_per_epoch" / "summary.tsv"
        if cand.exists():
            return cand
        # 2) <root>/summary.tsv (already an eval_per_epoch dir)
        cand = p / "summary.tsv"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate summary.tsv from {model_root_or_tsv}")


def plot(curves: list[tuple[str, dict | None, list[tuple[int, dict]]]],
         out_path: Path,
         title: str | None = None):
    """curves: list of (label, baseline_metrics, [(epoch, metrics), ...])."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.style.use("default")

    for ax, metric in zip(axes, METRICS):
        for i, (label, base, rows) in enumerate(curves):
            if not rows:
                continue
            xs = [ep for ep, _ in rows]
            ys = [m.get(metric) for _, m in rows]

            color = COLORS.get(label, DEFAULT_FALLBACK_COLORS[i % len(DEFAULT_FALLBACK_COLORS)])

            # Drop None points for plotting only
            xs_plot = [x for x, y in zip(xs, ys) if y is not None]
            ys_plot = [y for y in ys if y is not None]

            ax.plot(xs_plot, ys_plot, marker='o', linewidth=2.2, markersize=7,
                    color=color, label=f"LoRA-{label}")

            if base is not None and base.get(metric) is not None:
                ax.axhline(y=base[metric], color=color, linestyle='--',
                           linewidth=1.5, alpha=0.7,
                           label=f"Zero-Shot {label}: {base[metric]:.4g}")

        ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(metric.upper(), fontsize=13, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=9, loc='best')

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] wrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot per-epoch eval metrics (Precision / AUC@10 / AUC@20) "
                    "with Zero-Shot baseline horizontal reference.")
    p.add_argument("inputs", nargs="+",
                   help="One or more model roots (containing eval_per_epoch/summary.tsv) "
                        "or summary.tsv paths directly.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Display labels per input. Default: directory name "
                        "(navi_small -> small, navi_middle -> middle).")
    p.add_argument("--out", default=None,
                   help="Output PNG. Default: <first_input>/eval_per_epoch/per_epoch.png")
    p.add_argument("--title", default=None)
    return p.parse_args()


def _default_label(p: Path) -> str:
    name = p.expanduser().resolve().name
    if name.startswith("navi_"):
        return name[len("navi_"):]
    if name == "eval_per_epoch":
        return p.parent.name
    return name


def main():
    args = parse_args()

    inputs = [Path(s) for s in args.inputs]
    labels = (args.labels if args.labels
              else [_default_label(p) for p in inputs])
    if len(labels) != len(inputs):
        sys.exit(f"[error] {len(labels)} labels for {len(inputs)} inputs.")

    curves = []
    for label, inp in zip(labels, inputs):
        tsv = _resolve_summary(inp)
        base, rows = _read_summary(tsv)
        n_pts = len([r for r in rows if any(v is not None for v in r[1].values())])
        print(f"[read] {label}: {tsv}  -> {n_pts} epoch point(s)"
              + ("  (baseline ✓)" if base else ""))
        curves.append((label, base, rows))

    out = Path(args.out) if args.out else (inputs[0].expanduser().resolve()
                                           / "eval_per_epoch" / "per_epoch.png")

    title = args.title or "Per-Epoch Evaluation vs Zero-Shot Baseline"
    plot(curves, out, title=title)


if __name__ == "__main__":
    main()
