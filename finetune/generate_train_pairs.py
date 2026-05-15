"""
Generate train/val/test pairs from the full NAVI v1.5 dataset.

Reads annotations.json from each object/scene directory, extracts camera
parameters, computes relative poses, and outputs a pairs file in the same
38-token format used by evaluate_csv_essential.py:

    path_A path_B rot_A rot_B K_A[9] K_B[9] T_AB[16]

The --split argument selects which images participate in pair generation.
NAVI's annotations.json already labels each image as 'train' / 'val' / 'test',
and we only enumerate pairs whose BOTH endpoints belong to the requested split.
Therefore train/val/test pairs are guaranteed to be image-disjoint, with no
risk of data leakage between sets.

Usage:
    # Training pairs (default)
    python -m finetune.generate_train_pairs \
        --data_root full_dataset/navi_v1.5 \
        --split train \
        --output finetune/navi_train_pairs.txt \
        --max_pairs_per_scene 20 \
        --min_angle 10 --max_angle 90

    # Test pairs (target ~3000 pairs total)
    python -m finetune.generate_train_pairs \
        --data_root full_dataset/navi_v1.5 \
        --split test \
        --output datasets/navi_test_pairs.txt \
        --max_pairs_per_scene 12 \
        --min_angle 10 --max_angle 90
"""

from __future__ import annotations

import argparse
import json
import itertools
import math
import random
from pathlib import Path

import numpy as np


def quat_to_rotmat(q: list[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
    return R


def build_extrinsic(q: list[float], t: list[float]) -> np.ndarray:
    """Build 4x4 world-to-camera extrinsic matrix from quaternion and translation."""
    R = quat_to_rotmat(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def build_intrinsic(focal_length: float, image_size: list[int]) -> np.ndarray:
    """Build 3x3 intrinsic matrix. Assumes principal point at image centre."""
    h, w = image_size
    K = np.array([
        [focal_length, 0, w / 2.0],
        [0, focal_length, h / 2.0],
        [0, 0, 1],
    ])
    return K


def angular_distance(R: np.ndarray) -> float:
    """Compute angular distance of a rotation matrix in degrees."""
    trace = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return math.degrees(math.acos(trace))


def compute_relative_pose(T_w2c_0: np.ndarray, T_w2c_1: np.ndarray) -> np.ndarray:
    """Compute T_0to1: transforms points from camera 0 frame to camera 1 frame."""
    # T_0to1 = T_w2c_1 @ inv(T_w2c_0)
    T_0to1 = T_w2c_1 @ np.linalg.inv(T_w2c_0)
    return T_0to1


def process_scene(scene_dir: Path, max_pairs: int, min_angle: float, max_angle: float,
                  split: str = "train") -> list[str]:
    """Process one scene directory and return formatted pair lines.

    Only images whose annotation 'split' field equals `split` are used.
    Pairs are enumerated within that subset, so different splits never share
    an image (i.e., no data leakage between train / val / test).
    """
    ann_path = scene_dir / "annotations.json"
    if not ann_path.exists():
        return []

    with open(ann_path, "r") as f:
        annotations = json.load(f)

    # Filter to requested split only
    split_anns = [a for a in annotations if a.get("split", "train") == split]
    if len(split_anns) < 2:
        return []

    # Build camera data for each image
    cameras = {}
    for ann in split_anns:
        fname = ann["filename"]
        cam = ann["camera"]
        T_w2c = build_extrinsic(cam["q"], cam["t"])
        K = build_intrinsic(cam["focal_length"], ann["image_size"])
        
        # Construct the relative path: object_id/scene_name/images/filename
        obj_id = ann["object_id"]
        scene_name = ann["scene_name"]
        rel_path = f"{obj_id}/{scene_name}/images/{fname}"
        
        cameras[rel_path] = {
            "T_w2c": T_w2c,
            "K": K,
        }

    # Enumerate all pairs
    paths = list(cameras.keys())
    all_pairs = list(itertools.combinations(paths, 2))
    
    # Filter by angular distance
    valid_pairs = []
    for p0, p1 in all_pairs:
        T_0to1 = compute_relative_pose(cameras[p0]["T_w2c"], cameras[p1]["T_w2c"])
        R_rel = T_0to1[:3, :3]
        angle = angular_distance(R_rel)
        if min_angle <= angle <= max_angle:
            valid_pairs.append((p0, p1, T_0to1))

    # Subsample if too many
    if len(valid_pairs) > max_pairs:
        random.shuffle(valid_pairs)
        valid_pairs = valid_pairs[:max_pairs]

    # Format output lines
    lines = []
    for p0, p1, T_0to1 in valid_pairs:
        K0 = cameras[p0]["K"]
        K1 = cameras[p1]["K"]
        
        # Format: path_A path_B rot_A rot_B K_A[9] K_B[9] T_AB[16]
        tokens = [p0, p1, "0", "0"]
        tokens.extend([f"{v}" for v in K0.flatten()])
        tokens.extend([f"{v}" for v in K1.flatten()])
        tokens.extend([f"{v}" for v in T_0to1.flatten()])
        
        lines.append(" ".join(tokens))

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate training pairs from NAVI v1.5 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", type=str, default="full_dataset/navi_v1.5",
                        help="Root directory of the NAVI dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Which NAVI split to enumerate pairs from")
    parser.add_argument("--output", type=str, default="finetune/navi_train_pairs.txt",
                        help="Output pairs file")
    parser.add_argument("--max_pairs_per_scene", type=int, default=20,
                        help="Maximum pairs per scene")
    parser.add_argument("--min_angle", type=float, default=10.0,
                        help="Minimum angular distance between cameras (degrees)")
    parser.add_argument("--max_angle", type=float, default=90.0,
                        help="Maximum angular distance between cameras (degrees)")
    parser.add_argument("--max_total_pairs", type=int, default=0,
                        help="If >0, randomly subsample to this many pairs across all scenes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    all_lines = []
    n_scenes = 0

    # Iterate over all objects
    for obj_dir in sorted(data_root.iterdir()):
        if not obj_dir.is_dir() or obj_dir.name == "custom_splits":
            continue
        
        # Iterate over all scenes within this object
        for scene_dir in sorted(obj_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not scene_dir.name.startswith("multiview"):
                continue  # Skip video/wild_set directories for training
            
            lines = process_scene(
                scene_dir,
                max_pairs=args.max_pairs_per_scene,
                min_angle=args.min_angle,
                max_angle=args.max_angle,
                split=args.split,
            )
            if lines:
                n_scenes += 1
                all_lines.extend(lines)
                print(f"  {obj_dir.name}/{scene_dir.name}: {len(lines)} pairs")

    # Shuffle the final list
    random.shuffle(all_lines)

    # Optionally cap the total number of pairs (e.g. to get ~3000 test pairs)
    if args.max_total_pairs > 0 and len(all_lines) > args.max_total_pairs:
        all_lines = all_lines[: args.max_total_pairs]
        print(f"  -> Subsampled to {args.max_total_pairs} pairs")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"\nTotal: {len(all_lines)} {args.split} pairs from {n_scenes} scenes")
    print(f"Saved to: {output_path}")

    # Validate format
    with open(output_path, "r") as f:
        sample = f.readline().strip().split()
        assert len(sample) == 38, f"Expected 38 tokens per line, got {len(sample)}"
    print("Format validation: OK (38 tokens per line)")


if __name__ == "__main__":
    main()
