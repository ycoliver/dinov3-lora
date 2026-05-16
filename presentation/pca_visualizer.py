"""PCA 特征降维可视化（论文 Layer 1 / Layer 2 的定性证据）。

把 DINOv3 输出的 patch token (B, N, D) 用 PCA 降到 3 维，归一化到 [0,1]
当 RGB 贴回原图。直观对比：
  * Zero-Shot：色彩丰富、语义结构清晰  -> 流形未坍塌
  * LoRA 微调后：色彩变均匀          -> 流形被对比损失撕裂/塌缩

切换到 HuggingFace 后端 (与训练/评估管线一致)。同时支持 ViT-S 与 ViT-L
两种 backbone（``--model {small,middle}``），并可加载 LoRA checkpoint。

CLI 例：
  # 1) 仅 Zero-Shot small
  python presentation/pca_visualizer.py \
      --model small \
      --weights_dir dinov3_weights/dinov3-small \
      --image datasets/test/navi_resized/xxx.jpg

  # 2) Zero-Shot + LoRA 对照（同张图，4 张子图：原图 / ZS / LoRA / Δ）
  python presentation/pca_visualizer.py \
      --model small \
      --weights_dir dinov3_weights/dinov3-small \
      --checkpoint output/navi_small/lora_ckpt/checkpoint_latest.pth \
      --image datasets/test/navi_resized/xxx.jpg \
      --out_dir presentation/result/diag_small
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ensure repo root on path so we can import `finetune.*`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.model_hf import DINOv3HFBackbone
from finetune.lora_hf import inject_lora_hf, DEFAULT_HF_TARGETS


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────
def load_image(path: str, size: int = 448):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img_tensor = transform(img_rgb)

    # Also build a display version of the resized image for plotting.
    img_disp = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return img_disp, img_tensor


# ─────────────────────────────────────────────────────────────────────
# Model loading (Zero-Shot or LoRA)
# ─────────────────────────────────────────────────────────────────────
def build_backbone(weights_dir: str,
                   checkpoint: str | None,
                   lora_rank: int,
                   lora_alpha: float,
                   lora_targets: tuple[str, ...],
                   device: torch.device) -> DINOv3HFBackbone:
    """Build HF DINOv3 backbone. If ``checkpoint`` is given, inject LoRA
    and load the state-dict (mirroring extract_and_match_hf.build_model)."""
    backbone = DINOv3HFBackbone(weights_dir=weights_dir)
    if checkpoint:
        inject_lora_hf(
            backbone.hf_model,
            rank=lora_rank,
            alpha=lora_alpha,
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
        missing, unexpected = backbone.load_state_dict(sd_clean, strict=False)
        if unexpected:
            print(f"[load] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        ep = ckpt.get("epoch", "?")
        print(f"[load] Loaded LoRA checkpoint from {checkpoint}  (epoch {ep})")
    return backbone.to(device).eval()


@torch.no_grad()
def extract_patch_features(backbone: DINOv3HFBackbone,
                           img_tensor: torch.Tensor,
                           device: torch.device) -> np.ndarray:
    """Returns L2-normalised (N, D) patch features as numpy."""
    feats = backbone(img_tensor.unsqueeze(0).to(device))[0]  # (N, D), already L2-normed
    return feats.cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────
# PCA → RGB
# ─────────────────────────────────────────────────────────────────────
def pca_to_rgb(features: np.ndarray, h_patches: int, w_patches: int,
               pca: PCA | None = None) -> tuple[np.ndarray, PCA]:
    """Reduce (N, D) → (N, 3) and normalise per-channel to [0, 1].

    If ``pca`` is given, reuse its fitted projection (so two images share
    a colour space and become directly comparable).
    """
    if pca is None:
        pca = PCA(n_components=3)
        proj = pca.fit_transform(features)
    else:
        proj = pca.transform(features)

    for c in range(3):
        col = proj[:, c]
        proj[:, c] = (col - col.min()) / (col.max() - col.min() + 1e-8)

    return proj.reshape(h_patches, w_patches, 3), pca


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def _resize_pca(pca_img: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(pca_img, (w, h), interpolation=cv2.INTER_NEAREST)


def plot_single(orig_img: np.ndarray, pca_img: np.ndarray,
                title: str, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    h, w = orig_img.shape[:2]
    axes[1].imshow(_resize_pca(pca_img, h, w))
    axes[1].set_title(title, fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


def plot_pair(orig_img: np.ndarray,
              pca_zs: np.ndarray, pca_lora: np.ndarray,
              title_zs: str, title_lora: str,
              output_path: Path):
    """3 columns: original / ZS / LoRA — both PCAs on the same colour space."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(orig_img); axes[0].set_title("Original Image", fontsize=14, fontweight='bold'); axes[0].axis('off')

    h, w = orig_img.shape[:2]
    axes[1].imshow(_resize_pca(pca_zs, h, w))
    axes[1].set_title(title_zs, fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(_resize_pca(pca_lora, h, w))
    axes[2].set_title(title_lora, fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="PCA visualisation of DINOv3 patch features (Zero-Shot vs LoRA).")
    p.add_argument("--model", choices=["small", "middle"], default="small",
                   help="Used only to label the output files.")
    p.add_argument("--weights_dir", required=True,
                   help="HF weights directory, e.g. dinov3_weights/dinov3-small.")
    p.add_argument("--checkpoint", default=None,
                   help="Optional LoRA .pth (e.g. output/navi_small/lora_ckpt/checkpoint_latest.pth). "
                        "If omitted, only Zero-Shot PCA is produced.")
    p.add_argument("--image", required=True,
                   help="Path to a single image to visualise.")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--out_dir", default="presentation/result",
                   help="Output directory for PNGs.")
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_targets", nargs="+", default=list(DEFAULT_HF_TARGETS),
                   help="LoRA target module names (must match training).")
    p.add_argument("--shared_pca", action="store_true",
                   help="Fit PCA only on Zero-Shot features and reuse for LoRA, "
                        "so colours are directly comparable.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image once.
    img_disp, img_tensor = load_image(args.image, size=args.img_size)
    stem = Path(args.image).stem

    # Build Zero-Shot backbone.
    print(f"[zs] Loading Zero-Shot {args.model} from {args.weights_dir}")
    zs = build_backbone(args.weights_dir, None, 0, 0.0, (), device)
    h_p = w_p = args.img_size // zs.patch_size
    feats_zs = extract_patch_features(zs, img_tensor, device)
    pca_zs_img, pca_obj = pca_to_rgb(feats_zs, h_p, w_p)
    plot_single(img_disp, pca_zs_img,
                f"Zero-Shot DINOv3-{args.model.capitalize()}  (rich semantics)",
                out_dir / f"pca_{args.model}_{stem}_zs.png")

    del zs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optionally load LoRA backbone and produce the comparison plot.
    if args.checkpoint:
        print(f"[lora] Loading LoRA {args.model} ckpt = {args.checkpoint}")
        lora = build_backbone(
            args.weights_dir, args.checkpoint,
            args.lora_rank, args.lora_alpha, tuple(args.lora_targets),
            device,
        )
        feats_lora = extract_patch_features(lora, img_tensor, device)

        if args.shared_pca:
            pca_lora_img, _ = pca_to_rgb(feats_lora, h_p, w_p, pca=pca_obj)
        else:
            pca_lora_img, _ = pca_to_rgb(feats_lora, h_p, w_p)

        plot_single(img_disp, pca_lora_img,
                    f"LoRA-finetuned DINOv3-{args.model.capitalize()}",
                    out_dir / f"pca_{args.model}_{stem}_lora.png")

        plot_pair(img_disp, pca_zs_img, pca_lora_img,
                  f"Zero-Shot ({args.model})",
                  f"LoRA-finetuned ({args.model})",
                  out_dir / f"pca_{args.model}_{stem}_compare.png")

        del lora
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[done] Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
