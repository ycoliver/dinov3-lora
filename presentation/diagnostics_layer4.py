"""Layer-4 不可微 gap 的定量证据（论文 §5.4）。

论文 Layer 4 的论点是：
    cosine 相似度 → MNN → USAC → pose error
存在三重不可微链，因此 cosine 改善 **未必** 传递到 pose 改善。

这个脚本用 Zero-Shot 与 LoRA 两轮评估产出的 per-pair 结果（即
``evaluation_pairs.csv``），按 ``pair_id`` 配对后做散点：

    X = ΔPrecision   = precision_lora - precision_zs    (匹配质量改善, ↑ 好)
    Y = ΔPoseError   = pose_error_zs - pose_error_lora  (位姿改善, ↑ 好)

如果存在传递性，散点应集中在第一象限并相关；如果三重 gap 成立，散点
应呈现"匹配改善了但 pose 没改善"的发散云。

同时计算并打印 Pearson / Spearman 相关，作为定量结论。

CLI 例:
  python presentation/diagnostics_layer4.py \
      --zs_csv  output/navi_small/eval_zeroshot/evaluation_pairs.csv \
      --lora_csv output/navi_small/eval_per_epoch/epoch014/eval/evaluation_pairs.csv \
      --label small \
      --out_dir presentation/result/diag_small
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _read_pair_csv(path: Path) -> dict:
    """Returns dict[pair_id] -> {precision, num_matches, error_R, error_t, pose_error}."""
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("pair_id")
            if not pid:
                continue
            try:
                out[pid] = {
                    "precision": float(row["precision"]) if row.get("precision") not in (None, "") else float("nan"),
                    "num_matches": int(row["num_matches"]) if row.get("num_matches") not in (None, "") else 0,
                    "error_R": float(row["error_R"]) if row.get("error_R") not in (None, "") else float("nan"),
                    "error_t": float(row["error_t"]) if row.get("error_t") not in (None, "") else float("nan"),
                    "pose_error": float(row["pose_error"]) if row.get("pose_error") not in (None, "") else float("nan"),
                }
            except ValueError:
                continue
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx.astype(float), ry.astype(float))


def _cap(arr: np.ndarray, cap: float) -> np.ndarray:
    """Clip ±inf and outliers to cap (degree)."""
    a = np.where(np.isfinite(arr), arr, cap)
    return np.clip(a, -cap, cap)


def main():
    ap = argparse.ArgumentParser(description="Layer-4 non-differentiable-gap scatter plot.")
    ap.add_argument("--zs_csv", required=True, help="Zero-Shot evaluation_pairs.csv")
    ap.add_argument("--lora_csv", required=True, help="LoRA   evaluation_pairs.csv (e.g. last epoch)")
    ap.add_argument("--label", default="model", help="Used in figure titles & filenames.")
    ap.add_argument("--out_dir", default="presentation/result/diag")
    ap.add_argument("--max_pose_err", type=float, default=180.0,
                    help="Cap pose-error magnitude (degrees) to suppress ±inf outliers in scatter.")
    args = ap.parse_args()

    zs = _read_pair_csv(Path(args.zs_csv))
    lo = _read_pair_csv(Path(args.lora_csv))
    common = sorted(set(zs.keys()) & set(lo.keys()))
    if not common:
        sys.exit(f"[error] no common pair_id between\n  {args.zs_csv}\n  {args.lora_csv}")

    print(f"[data] zero-shot pairs={len(zs)}  lora pairs={len(lo)}  common={len(common)}")

    d_prec = np.array([lo[p]["precision"] - zs[p]["precision"] for p in common])
    pose_zs = _cap(np.array([zs[p]["pose_error"] for p in common]), args.max_pose_err)
    pose_lo = _cap(np.array([lo[p]["pose_error"] for p in common]), args.max_pose_err)
    d_pose = pose_zs - pose_lo  # 正值 = LoRA 比 ZS 的 pose 误差更小

    pearson = _pearson(d_prec, d_pose)
    spearman = _spearman(d_prec, d_pose)
    n_pos_x = int((d_prec > 0).sum())
    n_pos_y = int((d_pose > 0).sum())
    n_both = int(((d_prec > 0) & (d_pose > 0)).sum())
    n_x_no_y = int(((d_prec > 0) & (d_pose <= 0)).sum())

    print(f"[stat] N pairs                       = {len(common)}")
    print(f"[stat] Δprecision > 0                = {n_pos_x}  ({n_pos_x/len(common):.1%})")
    print(f"[stat] Δpose     > 0                 = {n_pos_y}  ({n_pos_y/len(common):.1%})")
    print(f"[stat] both improved                 = {n_both}")
    print(f"[stat] precision↑ but pose did not   = {n_x_no_y}")
    print(f"[stat] Pearson  corr(Δprec, Δpose)  = {pearson:.3f}")
    print(f"[stat] Spearman corr(Δprec, Δpose)  = {spearman:.3f}")

    # ── Plot ─────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
    ax.axvline(0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
    sc = ax.scatter(d_prec, d_pose, s=14, alpha=0.55, c="#1f77b4",
                    edgecolors="none")

    # quadrant counts annotation
    txt = (
        f"N = {len(common)}\n"
        f"Pearson  = {pearson:.3f}\n"
        f"Spearman = {spearman:.3f}\n"
        f"both improved: {n_both}\n"
        f"prec↑ but pose↓: {n_x_no_y}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel("Δ Precision  (LoRA − Zero-Shot)", fontweight='bold')
    ax.set_ylabel("Δ Pose-error  (Zero-Shot − LoRA)  [deg]", fontweight='bold')
    ax.set_title(f"Layer-4 indicator ({args.label}): cosine-side gain → pose-side gain?",
                 fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    out = out_dir / f"layer4_scatter_{args.label}.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] {out}")

    # secondary plot: Δpose distribution conditioned on Δprecision sign
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-args.max_pose_err, args.max_pose_err, 60)
    ax.hist(d_pose[d_prec > 0], bins=bins, alpha=0.6, color="#2ca02c",
            label=f"pairs with Δprec > 0  (n={n_pos_x})", density=True)
    ax.hist(d_pose[d_prec <= 0], bins=bins, alpha=0.6, color="#d62728",
            label=f"pairs with Δprec ≤ 0  (n={len(common)-n_pos_x})", density=True)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel("Δ Pose-error  (Zero-Shot − LoRA) [deg]   →  positive = improved")
    ax.set_ylabel("density")
    ax.set_title(f"Pose Δ distribution conditioned on Δprecision sign ({args.label})",
                 fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5); ax.legend()
    fig.tight_layout()
    out2 = out_dir / f"layer4_pose_hist_{args.label}.png"
    fig.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] {out2}")


if __name__ == "__main__":
    main()
