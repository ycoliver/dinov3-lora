"""
Generate train/val/test pairs from the full NAVI v1.5 dataset.

Reads annotations.json from each object/scene directory, extracts camera
parameters, computes relative poses, and outputs a pairs file in the same
38-token format used by evaluate_csv_essential.py:

    path_A path_B rot_A rot_B K_A[9] K_B[9] T_AB[16]

The --split argument selects whether to emit pairs from the train or test
SCENE partition. NAVI v1.5 does not ship per-image split labels in
annotations.json (only object/scene directories), so we instead perform a
deterministic SCENE-LEVEL split: a fixed seed shuffles the list of all
multiview_* scenes, and the last `--test_scene_ratio` fraction (default
15%) is reserved for testing. Because every image in a scene goes entirely
into one side of the split, train and test are guaranteed image-disjoint
with zero leakage.

Usage:
    # Training pairs (default)
    python -m finetune.generate_train_pairs \
        --data_root full_dataset/navi_v1.5 \
        --split train \
        --output finetune/navi_train_pairs.txt \
        --max_pairs_per_scene 20 \
        --min_angle 10 --max_angle 90

    # Test pairs (no cap → use whatever the held-out scenes give us)
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


def process_scene(scene_dir: Path, max_pairs: int, min_angle: float,
                  max_angle: float) -> list[str]:
    """Process one scene directory and return formatted pair lines.

    All images inside the scene are used; train/test image-disjointness is
    enforced at the SCENE level by the caller (`main`), not at the image
    level (NAVI annotations.json has no per-image split label).
    """
    ann_path = scene_dir / "annotations.json"
    if not ann_path.exists():
        return []

    with open(ann_path, "r") as f:
        annotations = json.load(f)

    if len(annotations) < 2:
        return []

    # Build camera data for each image
    cameras = {}
    for ann in annotations:
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
    parser.add_argument("--data_root", type=str, default="datasets/navi/navi_v1.5",
                        help="Root directory of the NAVI dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Which scene-level split to enumerate pairs from. "
                             "All multiview_* scenes are deterministically split "
                             "into train/test using --test_scene_ratio (no per-image "
                             "split labels are read from annotations.json).")
    parser.add_argument("--test_scene_ratio", type=float, default=0.15,
                        help="Fraction of multiview scenes reserved for testing. "
                             "The same shuffle seed is used for train and test, "
                             "so the two splits are image-disjoint by construction.")
    parser.add_argument("--split_seed", type=int, default=12345,
                        help="Seed used ONLY for the train/test scene shuffling. "
                             "Keep this fixed across both runs (train and test) "
                             "or train/test will overlap.")
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

    # ── Step 1. Collect ALL multiview_* scenes across all objects ────────
    all_scenes: list[Path] = []
    for obj_dir in sorted(data_root.iterdir()):
        if not obj_dir.is_dir() or obj_dir.name == "custom_splits":
            continue
        for scene_dir in sorted(obj_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if not scene_dir.name.startswith("multiview"):
                continue  # Skip video/wild_set directories
            if not (scene_dir / "annotations.json").exists():
                continue
            all_scenes.append(scene_dir)

    if not all_scenes:
        raise SystemExit(
            f"[error] No multiview_* scenes found under {data_root}. "
            f"Did you extract the NAVI tarball correctly?"
        )

    # ── Step 2. Deterministically split scenes into train / test ─────────
    # IMPORTANT: use a *separate* fixed seed (independent from --seed) so
    # both `--split train` and `--split test` runs see exactly the same
    # shuffle and therefore produce disjoint scene sets.
    scene_rng = random.Random(args.split_seed)
    shuffled = list(all_scenes)
    scene_rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * args.test_scene_ratio)))
    test_scenes = set(shuffled[-n_test:])
    train_scenes = set(shuffled[:-n_test])

    if args.split == "train":
        chosen_scenes = sorted(train_scenes, key=lambda p: str(p))
    else:
        chosen_scenes = sorted(test_scenes, key=lambda p: str(p))

    print(f"[split] total scenes={len(all_scenes)}  "
          f"train={len(train_scenes)}  test={len(test_scenes)}  "
          f"-> using {len(chosen_scenes)} scenes for split='{args.split}'")

    # ── Step 3. Generate pairs from chosen scenes ────────────────────────
    all_lines = []
    n_scenes = 0
    for scene_dir in chosen_scenes:
        lines = process_scene(
            scene_dir,
            max_pairs=args.max_pairs_per_scene,
            min_angle=args.min_angle,
            max_angle=args.max_angle,
        )
        if lines:
            n_scenes += 1
            all_lines.extend(lines)
            print(f"  {scene_dir.parent.name}/{scene_dir.name}: {len(lines)} pairs")

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
    if all_lines:
        with open(output_path, "r") as f:
            sample = f.readline().strip().split()
            assert len(sample) == 38, f"Expected 38 tokens per line, got {len(sample)}"
        print("Format validation: OK (38 tokens per line)")
    else:
        print("[warn] No pairs generated for this split — output file is empty.")


if __name__ == "__main__":
    main()
