"""
Generate a SuperGlue-style "teaser" matching figure for the report
by re-using the existing HF DINOv3 + LoRA-HF + MNN pipeline.

Loads ONE NAVI image pair, extracts patch features (zero-shot or
LoRA), computes mutual-nearest-neighbour matches, and saves a PNG
with the two images placed side-by-side and the top-K matches drawn
as connecting lines (green = high cosine, red = low cosine).

This script is **read-only** w.r.t. the rest of the codebase:
it imports `finetune.extract_and_match_hf` and uses its
`build_model`, `load_and_preprocess`, `mutual_nearest_neighbors`
helpers without modification.

Example
-------
# zero-shot ViT-S/16
python -m presentation.plot_match_teaser \
    --weights_dir dinov3_weights \
    --image_a datasets/navi_resized/<scene>/<view_a>.jpg \
    --image_b datasets/navi_resized/<scene>/<view_b>.jpg \
    --img_size 448 \
    --output presentation/result/teaser_matching_zs.png

# LoRA ViT-S/16 (recommended for the paper teaser)
python -m presentation.plot_match_teaser \
    --weights_dir dinov3_weights \
    --checkpoint output/navi_small/lora_ckpt/checkpoint_latest.pth \
    --image_a datasets/navi_resized/<scene>/<view_a>.jpg \
    --image_b datasets/navi_resized/<scene>/<view_b>.jpg \
    --img_size 448 \
    --output presentation/result/teaser_matching_lora.png \
    --topk 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Make sibling package importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.extract_and_match_hf import (   # noqa: E402
    build_model,
    load_and_preprocess,
    mutual_nearest_neighbors,
)
from finetune.lora_hf import DEFAULT_HF_TARGETS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_dir", type=str, default="dinov3_weights")
    p.add_argument("--checkpoint", type=str, default="",
                   help="LoRA-HF checkpoint (empty -> zero-shot).")
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_targets", type=str, nargs="+",
                   default=list(DEFAULT_HF_TARGETS))
    p.add_argument("--image_a", type=str, required=True)
    p.add_argument("--image_b", type=str, required=True)
    p.add_argument("--img_size", type=int, default=448,
                   help="Model input resolution (multiple of patch_size).")
    p.add_argument("--output", type=str, required=True,
                   help="Output PNG path.")
    p.add_argument("--topk", type=int, default=60,
                   help="Keep top-K MNN matches by cosine score.")
    p.add_argument("--score_threshold", type=float, default=0.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title", type=str, default=None,
                   help="Optional title above the figure.")
    return p.parse_args()


def patch_centres_in_image(h_patches: int, w_patches: int,
                            patch_size: int, img_size: int,
                            target_h: int, target_w: int) -> np.ndarray:
    """Return (N,2) (x,y) patch-centre pixel coords expressed in the
    target (display) image coordinate system."""
    ys = np.arange(h_patches) * patch_size + patch_size // 2
    xs = np.arange(w_patches) * patch_size + patch_size // 2
    sx = target_w / img_size
    sy = target_h / img_size
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel() * sx, gy.ravel() * sy], axis=1)


def load_display_image(path: str, max_side: int = 800) -> np.ndarray:
    """Load an RGB image for display, resized so max(H,W) <= max_side."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    return rgb


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[teaser] device={device}")

    # ---- Build (zero-shot or LoRA) model via the existing helper. ----
    model = build_model(args, device)
    patch_size = model.patch_size
    if args.img_size % patch_size != 0:
        raise ValueError(
            f"img_size ({args.img_size}) not divisible by "
            f"patch_size ({patch_size})"
        )
    h_patches = args.img_size // patch_size
    w_patches = args.img_size // patch_size

    # ---- Run the model on the two images. ----
    img_a, _ = load_and_preprocess(args.image_a, args.img_size)
    img_b, _ = load_and_preprocess(args.image_b, args.img_size)
    desc_a = model(img_a.unsqueeze(0).to(device))[0]
    desc_b = model(img_b.unsqueeze(0).to(device))[0]
    idx_a, idx_b, scores = mutual_nearest_neighbors(
        desc_a, desc_b, threshold=args.score_threshold,
    )
    print(f"[teaser] raw MNN matches = {len(idx_a)}")

    # ---- Keep top-K by score. ----
    if args.topk > 0 and len(scores) > args.topk:
        order = np.argsort(-scores)[: args.topk]
        idx_a, idx_b, scores = idx_a[order], idx_b[order], scores[order]
    print(f"[teaser] kept = {len(idx_a)} (top-{args.topk})")

    # ---- Load images for display (preserve aspect ratio). ----
    disp_a = load_display_image(args.image_a)
    disp_b = load_display_image(args.image_b)
    H_a, W_a = disp_a.shape[:2]
    H_b, W_b = disp_b.shape[:2]
    H = max(H_a, H_b)

    # Pad shorter image vertically so they sit on the same baseline.
    pad_a = np.full((H, W_a, 3), 255, dtype=np.uint8)
    pad_a[:H_a] = disp_a
    pad_b = np.full((H, W_b, 3), 255, dtype=np.uint8)
    pad_b[:H_b] = disp_b
    canvas = np.concatenate([pad_a, pad_b], axis=1)

    # ---- Patch-centre coordinates in display space. ----
    pts_a = patch_centres_in_image(
        h_patches, w_patches, patch_size, args.img_size, H_a, W_a
    )
    pts_b = patch_centres_in_image(
        h_patches, w_patches, patch_size, args.img_size, H_b, W_b
    )
    pts_b_shifted = pts_b.copy()
    pts_b_shifted[:, 0] += W_a   # right image is offset by W_a

    # ---- Draw. ----
    fig_w = (W_a + W_b) / 100.0
    fig_h = H / 100.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.set_axis_off()

    # Colour-code by cosine score: green (high) → yellow → red (low).
    if len(scores) > 0:
        s_min, s_max = float(scores.min()), float(scores.max())
        s_rng = max(s_max - s_min, 1e-6)
        cmap = cm.get_cmap("RdYlGn")
        for a_i, b_i, sc in zip(idx_a, idx_b, scores):
            xa, ya = pts_a[a_i]
            xb, yb = pts_b_shifted[b_i]
            colour = cmap((sc - s_min) / s_rng)
            ax.plot([xa, xb], [ya, yb], color=colour,
                    linewidth=0.7, alpha=0.85)
            ax.scatter([xa, xb], [ya, yb], s=4,
                       color=colour, edgecolors="none")

    if args.title:
        ax.set_title(args.title, fontsize=10)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(f"[teaser] saved -> {out_path}")


if __name__ == "__main__":
    main()
