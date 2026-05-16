"""
Extract dense features from a (zero-shot or LoRA-fine-tuned) HF DINOv3
backbone and write MNN matching results in lmz's CSV format so they can
be evaluated directly by `evaluate/evaluate_csv_essential.py`.

Counterpart of `extract_and_match.py`, but loads the model via
`DINOv3HFBackbone` (HuggingFace safetensors) and supports the LoRA-HF
checkpoint format produced by `train_lora_hf.py`. Setting `--checkpoint`
to an empty string runs **zero-shot** evaluation directly from the HF
weights — useful for the baseline comparison.

CRITICAL: keypoint coordinates in the CSV are written in the
**evaluation resize coordinate system** (default 640x480 in lmz's
pipeline), not the model's internal resolution.

CSV format:
    left_idx, right_idx, x1, y1, x2, y2, score

Usage:
    # Zero-shot (no fine-tune)
    python -m finetune.extract_and_match_hf \
        --weights_dir dinov3_weights \
        --pairs evaluate/navi/evaluation_pairs.csv \
        --data_root datasets/navi_resized \
        --output_dir mnn_matching_hf/navi \
        --img_size 448 \
        --eval_resize 640 480

    # Fine-tuned (LoRA-HF checkpoint)
    python -m finetune.extract_and_match_hf \
        --weights_dir dinov3_weights \
        --checkpoint finetune_output_lora_hf/checkpoint_latest.pth \
        --pairs evaluate/navi/evaluation_pairs.csv \
        --data_root datasets/navi_resized \
        --output_dir mnn_matching_lora_hf/navi
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

sys.stdout.reconfigure(line_buffering=True)

from .model_hf import DINOv3HFBackbone
from .lora_hf import inject_lora_hf, DEFAULT_HF_TARGETS
from .model import ProjectionHead   # for phase1 ckpts


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(
        description="HF DINOv3 feature extraction + MNN matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights_dir", type=str, default="dinov3_weights")
    p.add_argument("--checkpoint", type=str, default="",
                   help="LoRA-HF checkpoint (empty → zero-shot).")
    p.add_argument("--lora_rank", type=int, default=4,
                   help="Must match the rank used at training time.")
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_targets", type=str, nargs="+",
                   default=list(DEFAULT_HF_TARGETS))
    p.add_argument("--pairs", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--eval_resize", type=int, nargs=2, default=[640, 480],
                   help="Evaluation coord system (W H). Must equal the "
                        "evaluate script's --resize.")
    p.add_argument("--score_threshold", type=float, default=0.0)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
#  MNN matching
# ═════════════════════════════════════════════════════════════════════

def mutual_nearest_neighbors(
    desc_a: torch.Tensor,
    desc_b: torch.Tensor,
    threshold: float = 0.0,
):
    """
    Mutual Nearest Neighbour with cosine similarity. Inputs must be
    L2-normalised, shape `(N, D)` / `(M, D)`.
    """
    sim = desc_a @ desc_b.t()
    nn_b = sim.argmax(dim=1)
    nn_a = sim.argmax(dim=0)
    idx_a = torch.arange(desc_a.shape[0], device=desc_a.device)
    mutual = nn_a[nn_b] == idx_a
    scores = sim[idx_a, nn_b]
    valid = mutual & (scores >= threshold)
    return (
        idx_a[valid].cpu().numpy(),
        nn_b[valid].cpu().numpy(),
        scores[valid].cpu().numpy(),
    )


# ═════════════════════════════════════════════════════════════════════
#  Image preprocessing
# ═════════════════════════════════════════════════════════════════════

def load_and_preprocess(image_path: str, img_size: int):
    """Load an image, resize to `img_size x img_size`, ImageNet-normalise."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img_rgb), img.shape[:2]


def patch_coords_eval(h_patches, w_patches, patch_size, img_size,
                      eval_w, eval_h):
    """
    Pixel coordinates of patch centres expressed in the evaluation
    coordinate system (e.g. 640x480). Returns `(N, 2)` `(x, y)`.
    """
    ys = np.arange(h_patches) * patch_size + patch_size // 2
    xs = np.arange(w_patches) * patch_size + patch_size // 2
    sx = eval_w / img_size
    sy = eval_h / img_size
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack(
        [gx.ravel() * sx, gy.ravel() * sy], axis=1,
    )


# ═════════════════════════════════════════════════════════════════════
#  Pair-id naming (matches lmz's evaluate convention)
# ═════════════════════════════════════════════════════════════════════

def image_output_id(name: str) -> str:
    path = Path(name)
    scene = next((p for p in path.parts if p.startswith("scene")), None)
    if scene is not None:
        return f"{scene}_{path.stem}"
    parent_parts = [p for p in path.parts[:-1] if p not in ("", ".")]
    if parent_parts:
        return "{}_{}".format("_".join(parent_parts), path.stem)
    return path.stem


def pair_output_id(name0: str, name1: str) -> str:
    return f"{image_output_id(name0)}_{image_output_id(name1)}"


# ═════════════════════════════════════════════════════════════════════
#  Pairs file parser — supports both .txt (38-token) and .csv formats
# ═════════════════════════════════════════════════════════════════════

def parse_pairs(path: str):
    """
    Returns a list of (name_a, name_b) tuples.

    Accepts:
      * 38-token whitespace-separated .txt files (lmz format)
      * `evaluation_pairs.csv` style files where the first two columns
        are the image paths.
    """
    out = []
    with open(path, "r") as f:
        first = f.readline()
        f.seek(0)

        if "," in first and "left" in first.lower():
            # CSV with header
            reader = csv.DictReader(f)
            for row in reader:
                # try common column names
                a = row.get("left") or row.get("name0") or row.get("image0")
                b = row.get("right") or row.get("name1") or row.get("image1")
                if a and b:
                    out.append((a, b))
        else:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue
                out.append((tokens[0], tokens[1]))
    return out


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def build_model(args, device):
    """Build the HF DINOv3 backbone, optionally apply LoRA + ckpt.

    Returns either:
      - the bare backbone (whose `forward` yields L2-normalised patch tokens), or
      - a small wrapper module that runs `proj_head(backbone(x))` for Phase 1
        checkpoints (frozen backbone + ProjectionHead).
    The returned module exposes `.patch_size`, `.embed_dim`, `.get_patch_coords`
    so the rest of the script doesn't care which one it is.
    """
    backbone = DINOv3HFBackbone(weights_dir=args.weights_dir)

    # ---- Zero-shot fast-path: nothing to load. ----
    if not args.checkpoint:
        return backbone.to(device).eval()

    # ---- Detect checkpoint type (phase1 vs LoRA). ----
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    is_phase1 = bool(ckpt.get("phase1", False))
    if not is_phase1:
        # Some older ckpts don't carry the explicit flag but DO carry the
        # phase1_config block produced by `args.phase1`.
        is_phase1 = isinstance(ckpt.get("phase1_config"), dict)

    sd = ckpt.get("model_state_dict", ckpt)

    if is_phase1:
        proj_dim = (
            ckpt.get("phase1_config", {}).get("proj_dim")
            or ckpt.get("args", {}).get("proj_dim")
            or 256
        )
        proj_head = ProjectionHead(
            in_dim=backbone.embed_dim, proj_dim=proj_dim,
        )
        # The training wrapper saves keys as `backbone.hf_model.*` and
        # `proj_head.*`. Split them and load each into the right submodule.
        bb_sd = {}
        ph_sd = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                bb_sd[k[len("backbone."):]] = v
            elif k.startswith("proj_head."):
                ph_sd[k[len("proj_head."):]] = v
            else:
                # ignore unknown keys (e.g. optimiser leftovers)
                pass
        backbone.load_state_dict(bb_sd, strict=False)
        ph_missing, ph_unexpected = proj_head.load_state_dict(ph_sd, strict=False)
        if ph_missing or ph_unexpected:
            print(f"[load] ProjectionHead missing={len(ph_missing)} "
                  f"unexpected={len(ph_unexpected)}")
        print(
            f"[load] Loaded Phase-1 checkpoint "
            f"(epoch {ckpt.get('epoch', '?')}, proj_dim={proj_dim})"
        )

        class _Phase1Model(torch.nn.Module):
            def __init__(self, backbone, proj_head, proj_dim):
                super().__init__()
                self.backbone = backbone
                self.proj_head = proj_head
                self.patch_size = backbone.patch_size
                self.embed_dim = proj_dim
            def forward(self, x):
                with torch.no_grad():
                    feats = self.backbone(x)
                return self.proj_head(feats)
            @torch.no_grad()
            def get_patch_coords(self, H, W):
                return self.backbone.get_patch_coords(H, W)

        return _Phase1Model(backbone, proj_head, proj_dim).to(device).eval()

    # ---- LoRA path (existing behaviour). ----
    inject_lora_hf(
        backbone.hf_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_modules=tuple(args.lora_targets),
    )
    sd_clean = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            sd_clean[k[len("backbone."):]] = v
        else:
            sd_clean[k] = v
    missing, unexpected = backbone.load_state_dict(sd_clean, strict=False)
    if unexpected:
        print(f"[load] Unexpected keys ({len(unexpected)}): "
              f"{unexpected[:5]}...")
    print(
        f"[load] Loaded LoRA-HF checkpoint "
        f"(epoch {ckpt.get('epoch', '?')})"
    )
    return backbone.to(device).eval()


def main():
    args = get_args()
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_w, eval_h = args.eval_resize
    print(f"Model input size: {args.img_size}x{args.img_size}")
    print(f"Eval coordinate system: {eval_w}x{eval_h}")

    # Build model
    print("Building model...")
    model = build_model(args, device)
    patch_size = model.patch_size
    if args.img_size % patch_size != 0:
        raise ValueError(
            f"img_size ({args.img_size}) not divisible by patch_size "
            f"({patch_size})"
        )
    h_patches = args.img_size // patch_size
    w_patches = args.img_size // patch_size
    coords_eval = patch_coords_eval(
        h_patches, w_patches, patch_size, args.img_size, eval_w, eval_h,
    )

    # Pairs
    pairs = parse_pairs(args.pairs)
    print(f"Processing {len(pairs)} pairs...")
    print(
        f"Patch coord range: x=[{coords_eval[:,0].min():.1f}, "
        f"{coords_eval[:,0].max():.1f}], "
        f"y=[{coords_eval[:,1].min():.1f}, {coords_eval[:,1].max():.1f}]"
    )

    t0 = time.time()
    skipped = 0
    for i, (name0, name1) in enumerate(pairs):
        pid = pair_output_id(name0, name1)
        path0 = str(Path(args.data_root) / name0)
        path1 = str(Path(args.data_root) / name1)
        try:
            img_a, _ = load_and_preprocess(path0, args.img_size)
            img_b, _ = load_and_preprocess(path1, args.img_size)
        except FileNotFoundError:
            skipped += 1
            continue

        with torch.no_grad():
            desc_a = model(img_a.unsqueeze(0).to(device))[0]   # (N, D)
            desc_b = model(img_b.unsqueeze(0).to(device))[0]
        idx_a, idx_b, scores = mutual_nearest_neighbors(
            desc_a, desc_b, threshold=args.score_threshold,
        )

        csv_path = out_dir / f"{pid}_matches.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["left_idx", "right_idx", "x1", "y1", "x2", "y2", "score"]
            )
            for a, b, s in zip(idx_a, idx_b, scores):
                x1, y1 = coords_eval[a]
                x2, y2 = coords_eval[b]
                writer.writerow([
                    int(a), int(b),
                    f"{x1:.1f}", f"{y1:.1f}",
                    f"{x2:.1f}", f"{y2:.1f}",
                    f"{s}",
                ])

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(pairs)}] {elapsed:.1f}s | "
                f"{pid} | {len(idx_a)} matches"
            )

    elapsed = time.time() - t0
    print(f"\nDone! {len(pairs)} pairs in {elapsed:.1f}s (skipped {skipped})")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
