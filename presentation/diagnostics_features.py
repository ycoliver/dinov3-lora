"""轻量化诊断指标（论文 Layer 1 / Layer 2 / Layer 3 的定量证据）。

针对一组测试图像，分别用 Zero-Shot 与 LoRA-finetuned 的 backbone 提取
patch features，然后计算并可视化：

  1) Layer 2 - Manifold collapse:
       * mean intra-image pairwise cosine    (越高 → 越塌)
       * effective rank (entropy of PCA spectrum, normalised)  (越低 → 越塌)
       * intra-image pairwise cosine 直方图   (zero-shot vs lora 重叠)

  2) Layer 3 - Positive-pair viewpoint similarity:
       同一张图做小幅 random crop + scale 形成两份视图，把可几何对应的
       patch pair 的 cosine 直方图画出来：
           * Zero-Shot 的分布告诉我们"原生几何不变性"已经多强
           * LoRA 后的分布若整体更靠近 1.0 → "正样本被过分拉拢" (Layer 3)

  3) Layer 1 - Semantic-not-geometric proxy:
       计算 "top-1 NN 是不是空间相邻 patch" 的比例 -> 数值高表示模型
       几乎只能靠空间相邻性区分 patch（即缺乏几何独特性）。

CLI 例:
  python presentation/diagnostics_features.py \
      --model small \
      --weights_dir dinov3_weights/dinov3-small \
      --checkpoint output/navi_small/lora_ckpt/checkpoint_latest.pth \
      --image_dir datasets/test/navi_resized \
      --num_images 30 \
      --out_dir presentation/result/diag_small
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.model_hf import DINOv3HFBackbone
from finetune.lora_hf import inject_lora_hf, DEFAULT_HF_TARGETS


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────
# Backbone (Zero-Shot or LoRA)
# ─────────────────────────────────────────────────────────────────────
def build_backbone(weights_dir: str,
                   checkpoint: str | None,
                   lora_rank: int,
                   lora_alpha: float,
                   lora_targets: tuple[str, ...],
                   device: torch.device) -> DINOv3HFBackbone:
    backbone = DINOv3HFBackbone(weights_dir=weights_dir)
    if checkpoint:
        inject_lora_hf(
            backbone.hf_model, rank=lora_rank, alpha=lora_alpha,
            target_modules=lora_targets,
        )
        ckpt = torch.load(checkpoint, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        sd_clean = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                sd_clean[k[len("backbone."):]] = v
            else:
                sd_clean[k] = v
        backbone.load_state_dict(sd_clean, strict=False)
        print(f"[load] LoRA ckpt: {checkpoint}  (epoch {ckpt.get('epoch', '?')})")
    return backbone.to(device).eval()


# ─────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────
def _normalize() -> T.Normalize:
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_image_tensor(path: str, size: int) -> torch.Tensor:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return _normalize()(t)


def load_pair_random_crops(path: str, size: int,
                           min_scale: float = 0.7,
                           max_scale: float = 0.95,
                           rng: random.Random | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Two random-cropped + resized views of the same image, plus a
    `(size, size, 2)` int32 array mapping each pixel of view-A back to
    its (x, y) location in view-B. Pixels with no correspondence have
    value -1.

    The returned correspondence is **dense pixel-level**; we'll later
    aggregate it onto the patch grid.
    """
    rng = rng or random.Random()
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]

    def _rand_crop():
        s = rng.uniform(min_scale, max_scale)
        ch, cw = int(H0 * s), int(W0 * s)
        y = rng.randint(0, H0 - ch)
        x = rng.randint(0, W0 - cw)
        return x, y, cw, ch

    def _to_tensor(crop):
        c = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(c).permute(2, 0, 1).float() / 255.0
        return _normalize()(t)

    xa, ya, wa, ha = _rand_crop()
    xb, yb, wb, hb = _rand_crop()
    crop_a = img_rgb[ya:ya + ha, xa:xa + wa]
    crop_b = img_rgb[yb:yb + hb, xb:xb + wb]

    # Build pixel correspondence: for each pixel (u, v) in view-A (after
    # resize to size x size), back-project to original image, then forward-
    # project into view-B (also size x size).
    us = np.arange(size).astype(np.float32)
    vs = np.arange(size).astype(np.float32)
    U, V = np.meshgrid(us, vs)  # (size, size)
    # view-A → original
    Ox = xa + U * (wa / size)
    Oy = ya + V * (ha / size)
    # original → view-B
    Bx = (Ox - xb) * (size / wb)
    By = (Oy - yb) * (size / hb)
    valid = (Bx >= 0) & (Bx < size) & (By >= 0) & (By < size)
    corr = np.full((size, size, 2), -1, dtype=np.int32)
    corr[valid, 0] = Bx[valid].astype(np.int32)
    corr[valid, 1] = By[valid].astype(np.int32)

    return _to_tensor(crop_a), _to_tensor(crop_b), corr


# ─────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract(backbone: DINOv3HFBackbone, x: torch.Tensor,
            device: torch.device) -> torch.Tensor:
    """Returns (N, D) L2-normed patch features."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    feats = backbone(x.to(device))[0]
    return feats.float().cpu()


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────
def intra_image_cosines(feats: torch.Tensor) -> np.ndarray:
    """Off-diagonal entries of the (N, N) similarity matrix."""
    sim = feats @ feats.T  # already L2-normed
    n = sim.size(0)
    mask = ~torch.eye(n, dtype=torch.bool)
    return sim[mask].numpy()


def effective_rank(feats: torch.Tensor) -> float:
    """Spectrum entropy of the centered feature matrix, normalised by
    log(D). Range ≈ [0, 1]; 1 = isotropic, 0 = rank-1 (collapsed)."""
    F0 = feats - feats.mean(dim=0, keepdim=True)
    # Use SVD on the (N, D) matrix (D >= N usually fine)
    s = torch.linalg.svdvals(F0)
    p = (s ** 2)
    p = p / (p.sum() + 1e-12)
    p = p.clamp(min=1e-12)
    H = -(p * p.log()).sum().item()
    return H / float(np.log(feats.size(1)))


def positive_pair_cosines(feats_a: torch.Tensor, feats_b: torch.Tensor,
                          corr_pixel: np.ndarray, patch_size: int,
                          h_p: int, w_p: int) -> np.ndarray:
    """Aggregate dense pixel correspondences onto patch grid; for every
    correspondent (patch_a, patch_b) compute cos-sim."""
    # Down-sample dense pixel correspondence to patch level by taking
    # the centre pixel of each patch in view-A.
    centres = (np.arange(h_p) * patch_size + patch_size // 2).astype(np.int32)
    cy, cx = np.meshgrid(centres, centres, indexing="ij")  # (h_p, w_p)
    # centre pixel in view-A → corresponding pixel in view-B
    bx = corr_pixel[cy, cx, 0]
    by = corr_pixel[cy, cx, 1]
    valid = (bx >= 0) & (by >= 0)

    # patch index (linear) on view-B
    pb_x = (bx[valid] // patch_size).clip(0, w_p - 1)
    pb_y = (by[valid] // patch_size).clip(0, h_p - 1)
    pb_idx = pb_y * w_p + pb_x

    pa_idx = np.arange(h_p * w_p).reshape(h_p, w_p)[valid]

    a = feats_a[pa_idx]
    b = feats_b[pb_idx]
    # both already L2-normed
    return (a * b).sum(dim=-1).numpy()


def neighbour_dominance(feats: torch.Tensor, h_p: int, w_p: int,
                        radius: int = 1) -> float:
    """Fraction of patches whose top-1 NN (excluding self) is a spatial
    neighbour within `radius`. High value (close to 1) → model relies
    almost entirely on spatial proximity to discriminate patches, i.e.
    feature geometric distinctiveness is weak."""
    sim = feats @ feats.T
    sim.fill_diagonal_(-1.0)
    nn = sim.argmax(dim=1).numpy()  # (N,)

    n = h_p * w_p
    rows = np.arange(n) // w_p
    cols = np.arange(n) % w_p
    nn_r = nn // w_p
    nn_c = nn % w_p

    is_neighbour = (np.abs(rows - nn_r) <= radius) & (np.abs(cols - nn_c) <= radius)
    return float(is_neighbour.mean())


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────
def collect_metrics(backbone: DINOv3HFBackbone,
                    image_paths: list[str],
                    img_size: int,
                    device: torch.device,
                    rng: random.Random) -> dict:
    """Collect all diagnostic stats over a list of images."""
    h_p = w_p = img_size // backbone.patch_size

    intra_cos_all = []
    eff_rank_all = []
    pos_cos_all = []
    neigh_dom_all = []

    for i, p in enumerate(image_paths):
        # 1) full image: intra-image cosine + effective rank + neighbour-dominance
        x = load_image_tensor(p, img_size)
        feats = extract(backbone, x, device)  # (N, D)
        intra_cos_all.append(intra_image_cosines(feats))
        eff_rank_all.append(effective_rank(feats))
        neigh_dom_all.append(neighbour_dominance(feats, h_p, w_p))

        # 2) random crop pair: positive-pair cosine
        try:
            ta, tb, corr = load_pair_random_crops(p, img_size, rng=rng)
        except Exception as e:
            print(f"[warn] crop-pair failed on {p}: {e}", file=sys.stderr)
            continue
        fa = extract(backbone, ta, device)
        fb = extract(backbone, tb, device)
        pos = positive_pair_cosines(fa, fb, corr, backbone.patch_size, h_p, w_p)
        if pos.size > 0:
            pos_cos_all.append(pos)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(image_paths)}] images processed")

    intra = np.concatenate(intra_cos_all) if intra_cos_all else np.array([])
    pos = np.concatenate(pos_cos_all) if pos_cos_all else np.array([])
    return {
        "intra_cos": intra,
        "pos_cos": pos,
        "eff_rank": np.array(eff_rank_all),
        "neigh_dom": np.array(neigh_dom_all),
        "n_images": len(image_paths),
    }


def summarise(stats: dict) -> dict:
    intra = stats["intra_cos"]
    pos = stats["pos_cos"]
    return {
        "n_images": stats["n_images"],
        "mean_intra_cos": float(intra.mean()) if intra.size else float("nan"),
        "median_intra_cos": float(np.median(intra)) if intra.size else float("nan"),
        "mean_eff_rank": float(stats["eff_rank"].mean()) if stats["eff_rank"].size else float("nan"),
        "mean_pos_cos": float(pos.mean()) if pos.size else float("nan"),
        "median_pos_cos": float(np.median(pos)) if pos.size else float("nan"),
        "mean_neigh_dominance": float(stats["neigh_dom"].mean()) if stats["neigh_dom"].size else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def plot_histograms(stats_zs: dict, stats_lora: dict, out_dir: Path,
                    model_label: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) intra-image pairwise cosine
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(stats_zs["intra_cos"], bins=bins, alpha=0.55, density=True,
            color="#2ca02c", label=f"Zero-Shot  μ={stats_zs['intra_cos'].mean():.3f}")
    if stats_lora is not None:
        ax.hist(stats_lora["intra_cos"], bins=bins, alpha=0.55, density=True,
                color="#d62728", label=f"LoRA       μ={stats_lora['intra_cos'].mean():.3f}")
    ax.set_xlabel("Intra-image patch pairwise cosine", fontweight='bold')
    ax.set_ylabel("density", fontweight='bold')
    ax.set_title(f"Manifold-collapse indicator ({model_label})  ↑ = more collapsed",
                 fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend()
    out = out_dir / f"hist_intra_cos_{model_label}.png"
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[ok] {out}")

    # 2) positive-pair cosine
    if stats_zs["pos_cos"].size:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0.0, 1.0, 60)
        ax.hist(stats_zs["pos_cos"], bins=bins, alpha=0.55, density=True,
                color="#2ca02c", label=f"Zero-Shot  μ={stats_zs['pos_cos'].mean():.3f}")
        if stats_lora is not None and stats_lora["pos_cos"].size:
            ax.hist(stats_lora["pos_cos"], bins=bins, alpha=0.55, density=True,
                    color="#d62728", label=f"LoRA       μ={stats_lora['pos_cos'].mean():.3f}")
        ax.set_xlabel("Positive-pair cosine (same 3D point, two crops)", fontweight='bold')
        ax.set_ylabel("density", fontweight='bold')
        ax.set_title(f"Layer-3 indicator ({model_label})  →1.0 = positives over-pulled",
                     fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6); ax.legend()
        out = out_dir / f"hist_pos_cos_{model_label}.png"
        fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
        print(f"[ok] {out}")


def plot_summary_bars(summary_zs: dict, summary_lora: dict | None,
                      out_dir: Path, model_label: str):
    metrics = ["mean_intra_cos", "mean_eff_rank",
               "mean_pos_cos", "mean_neigh_dominance"]
    titles = ["intra-image\ncos (↓ better)",
              "effective rank\n(↑ better)",
              "positive-pair\ncos",
              "neighbour\ndominance (↓ better)"]

    zs_vals = [summary_zs[m] for m in metrics]
    lo_vals = [summary_lora[m] for m in metrics] if summary_lora else None

    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, zs_vals, width, color="#2ca02c", label="Zero-Shot")
    if lo_vals is not None:
        ax.bar(x + width / 2, lo_vals, width, color="#d62728", label="LoRA")

    for i, v in enumerate(zs_vals):
        ax.text(i - width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    if lo_vals is not None:
        for i, v in enumerate(lo_vals):
            ax.text(i + width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(titles, fontsize=10)
    ax.set_title(f"Feature-space diagnostics ({model_label})",
                 fontweight='bold', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.5, axis="y")
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"bars_summary_{model_label}.png"
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"[ok] {out}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def gather_image_paths(image_dir: Path, n: int, rng: random.Random) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [str(p) for p in image_dir.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    rng.shuffle(paths)
    return paths[:n]


def parse_args():
    p = argparse.ArgumentParser(
        description="Lightweight feature-space diagnostics: intra-image cos, "
                    "effective rank, positive-pair cos, neighbour dominance.")
    p.add_argument("--model", choices=["small", "middle"], default="small",
                   help="Used as a label in output filenames.")
    p.add_argument("--weights_dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--num_images", type=int, default=30)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--out_dir", default="presentation/result/diag")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_targets", nargs="+", default=list(DEFAULT_HF_TARGETS))
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_paths = gather_image_paths(Path(args.image_dir), args.num_images, rng)
    print(f"[data] using {len(image_paths)} images from {args.image_dir}")

    # --- Zero-Shot ---
    print(f"\n[zs] building Zero-Shot {args.model} ...")
    zs = build_backbone(args.weights_dir, None, 0, 0.0, (), device)
    rng_zs = random.Random(args.seed)
    stats_zs = collect_metrics(zs, image_paths, args.img_size, device, rng_zs)
    summary_zs = summarise(stats_zs)
    print(f"[zs] summary: {summary_zs}")
    del zs
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- LoRA ---
    stats_lora = summary_lora = None
    if args.checkpoint:
        print(f"\n[lora] building LoRA {args.model} ...")
        lora = build_backbone(args.weights_dir, args.checkpoint,
                              args.lora_rank, args.lora_alpha,
                              tuple(args.lora_targets), device)
        rng_lora = random.Random(args.seed)  # IMPORTANT: same crops for fair comparison
        stats_lora = collect_metrics(lora, image_paths, args.img_size, device, rng_lora)
        summary_lora = summarise(stats_lora)
        print(f"[lora] summary: {summary_lora}")
        del lora
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Outputs ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_histograms(stats_zs, stats_lora, out_dir, args.model)
    plot_summary_bars(summary_zs, summary_lora, out_dir, args.model)

    # Save raw summary as TSV for the report.
    tsv = out_dir / f"summary_{args.model}.tsv"
    cols = ["metric", "zero_shot"] + (["lora"] if summary_lora else []) \
           + (["delta"] if summary_lora else [])
    with tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for k in ("n_images", "mean_intra_cos", "median_intra_cos",
                  "mean_eff_rank", "mean_pos_cos", "median_pos_cos",
                  "mean_neigh_dominance"):
            row = [k, f"{summary_zs[k]:.6g}"]
            if summary_lora:
                row.append(f"{summary_lora[k]:.6g}")
                if isinstance(summary_zs[k], (int, float)) and isinstance(summary_lora[k], (int, float)):
                    row.append(f"{summary_lora[k] - summary_zs[k]:+.6g}")
                else:
                    row.append("-")
            f.write("\t".join(row) + "\n")
    print(f"[ok] {tsv}")


if __name__ == "__main__":
    main()
