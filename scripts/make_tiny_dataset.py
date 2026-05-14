#!/usr/bin/env python3
"""
Generate a *tiny* synthetic dataset to smoke-test the LoRA-HF pipeline
end-to-end (extract -> match -> train -> evaluate) without downloading
the real NAVI dataset.

What this produces under `datasets/tiny/`:
  images/
    pair{i}/a.jpg
    pair{i}/b.jpg
  pairs.txt          # 38-token-per-line file, used as BOTH
                     #   train_pairs and eval_pairs

Each pair (a, b) is the same textured image rendered twice with a small
camera translation (so the epipolar geometry is well-defined and lots of
patch-level correspondences are produced).

Why a single file for both train & eval:
  - The training dataset only consumes columns 0..37.
  - The evaluation script also splits each line by whitespace and
    requires exactly 38 columns; the column meaning is identical.
  - Using the same pairs lets us verify the whole pipeline in <5 min
    without a real dataset.

Usage:
    python scripts/make_tiny_dataset.py \
        --out datasets/tiny --num_pairs 8 --img_size 480
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------
# Synthetic image: a high-frequency checkerboard + colour gradient + a
# few coloured circles, so that DINOv3 patch features have actual
# discriminative content.
# ---------------------------------------------------------------------
def make_textured_image(H: int, W: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # 1) Smooth colour gradient (so each region looks different)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = (xx / W * 255).astype(np.uint8)
    g = (yy / H * 255).astype(np.uint8)
    b = (((xx + yy) / (H + W)) * 255).astype(np.uint8)
    img = np.stack([b, g, r], axis=2)  # BGR for cv2

    # 2) Checkerboard high-frequency texture
    cell = 24
    cb = (((xx // cell).astype(int) + (yy // cell).astype(int)) % 2) * 60
    img = np.clip(img.astype(np.int32) + cb[..., None], 0, 255).astype(np.uint8)

    # 3) A few coloured shapes for distinctive landmarks
    for _ in range(20):
        cx = int(rng.integers(20, W - 20))
        cy = int(rng.integers(20, H - 20))
        radius = int(rng.integers(8, 24))
        colour = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.circle(img, (cx, cy), radius, colour, thickness=-1)

    for _ in range(10):
        x0 = int(rng.integers(0, W - 40))
        y0 = int(rng.integers(0, H - 40))
        x1 = int(x0 + rng.integers(20, 60))
        y1 = int(y0 + rng.integers(20, 60))
        colour = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, thickness=2)

    return img


def synth_intrinsics(H: int, W: int) -> np.ndarray:
    """A reasonable pinhole intrinsics for a (W, H) image."""
    f = 1.2 * max(H, W)
    K = np.array([
        [f, 0.0, W / 2.0],
        [0.0, f, H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


def synth_pose(seed: int) -> np.ndarray:
    """
    Identity-ish pose with a small lateral translation so that the
    epipolar geometry is non-degenerate but the two views still see
    almost the same content. This guarantees plenty of cross-view
    patch correspondences.
    """
    rng = np.random.default_rng(seed + 9999)
    # Small rotation (a few degrees about y-axis)
    theta = float(rng.uniform(-0.05, 0.05))  # ~3 degrees max
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)
    # Small translation (mostly along x)
    t = np.array([
        float(rng.uniform(0.05, 0.15)),
        float(rng.uniform(-0.02, 0.02)),
        float(rng.uniform(-0.02, 0.02)),
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def write_jpg(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def serialise_pair_line(name0: str, name1: str, K0, K1, T) -> str:
    tokens = [name0, name1, "0", "0"]
    tokens.extend(f"{v:.6f}" for v in K0.flatten())
    tokens.extend(f"{v:.6f}" for v in K1.flatten())
    tokens.extend(f"{v:.6f}" for v in T.flatten())
    assert len(tokens) == 38, f"expected 38 tokens, got {len(tokens)}"
    return " ".join(tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="datasets/tiny")
    ap.add_argument("--num_pairs", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=480,
                    help="Synthetic image size (square).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    H = W = args.img_size
    K = synth_intrinsics(H, W)

    pairs_lines: list[str] = []
    print(f"Generating {args.num_pairs} synthetic pairs at {W}x{H} into "
          f"{out_root}/")
    for i in range(args.num_pairs):
        # Same texture for both views in this pair → guarantees lots of
        # correspondences after epipolar matching.
        img = make_textured_image(H, W, seed=args.seed + i)
        rel_a = f"images/pair{i:03d}/a.jpg"
        rel_b = f"images/pair{i:03d}/b.jpg"
        write_jpg(out_root / rel_a, img)
        write_jpg(out_root / rel_b, img)

        T = synth_pose(seed=args.seed + i)
        pairs_lines.append(serialise_pair_line(rel_a, rel_b, K, K, T))

    pairs_path = out_root / "pairs.txt"
    pairs_path.write_text("\n".join(pairs_lines) + "\n")
    print(f"  wrote {pairs_path} ({len(pairs_lines)} lines, 38 tokens each)")
    print("Done.")


if __name__ == "__main__":
    main()
