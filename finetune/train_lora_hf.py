"""
LoRA fine-tuning of DINOv3 (HuggingFace safetensors) for local feature
matching.

This script is the HF / ViT-B counterpart of `train_lora.py`. Differences:
  1. Backbone is `transformers.Dinov3ViTModel` loaded from a local HF
     directory (e.g. `dinov3_weights/`). No re-implemented ViT, no .pth.
  2. LoRA is injected into Q/V projections (HF stores them as separate
     `q_proj` / `v_proj` Linear modules, unlike the fused `qkv` in the
     .pth code path).
  3. `diversity_weight` defaults to 0.0 (cleaner InfoNCE baseline; the
     1024x1024 / 768x768 redundancy matrix was unstable on small batches).
  4. `img_size` is no longer hard-coded — it is forwarded into the loss
     so that the safe-radius mask is computed correctly.

Usage:
    python -m finetune.train_lora_hf \
        --weights_dir dinov3_weights \
        --train_pairs finetune/navi_train_pairs.txt \
        --data_root datasets/navi \
        --depth_root "" \
        --output_dir finetune_output_lora_hf \
        --epochs 15 --batch_size 1 --img_size 448
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.stdout.reconfigure(line_buffering=True)

from .config import LOG_EVERY, SAVE_EVERY
from .model_hf import DINOv3HFBackbone
from .lora_hf import inject_lora_hf, get_lora_parameters, DEFAULT_HF_TARGETS
from .dataset import MatchingPairDataset, collate_matching_pairs
from .loss import MatchingLoss


# =====================================================================
#  Model: HF backbone + LoRA, no projection head
# =====================================================================

class LoRADINOv3MatcherHF(nn.Module):
    """
    HF-loaded DINOv3 ViT with LoRA adapters; no projection head.

    Output is the L2-normalised patch tokens (in the native embed_dim
    of the backbone — 768 for ViT-B, 1024 for ViT-L, ...).

    At init, LoRA's B matrix is zero so the model output is **identical**
    to the zero-shot DINOv3.
    """

    def __init__(
        self,
        weights_dir: str,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_targets: tuple[str, ...] = DEFAULT_HF_TARGETS,
    ):
        super().__init__()
        self.backbone = DINOv3HFBackbone(weights_dir=weights_dir)

        # Freeze everything first.
        for p in self.parameters():
            p.requires_grad = False

        # Inject LoRA — these become the only trainable parameters.
        inject_lora_hf(
            self.backbone.hf_model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_targets,
        )

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[LoRA-HF] Trainable: {n_train/1e6:.3f}M / {n_total/1e6:.1f}M "
            f"params ({100*n_train/n_total:.3f}%)"
        )

        # Convenience accessors for the dataset / loss / matching code.
        self.patch_size: int = self.backbone.patch_size
        self.embed_dim: int = self.backbone.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns `(B, num_patches, embed_dim)` L2-normalised tokens."""
        return self.backbone(x)

    @torch.no_grad()
    def get_patch_coords(self, H: int, W: int) -> torch.Tensor:
        return self.backbone.get_patch_coords(H, W)


# =====================================================================
#  CLI
# =====================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="LoRA fine-tune HF DINOv3 for local feature matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / model
    p.add_argument("--weights_dir", type=str, default="dinov3_weights",
                   help="Local HF directory with config.json + model.safetensors")
    p.add_argument("--train_pairs", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--depth_root", type=str, default="")
    p.add_argument("--output_dir", type=str, default="finetune_output_lora_hf")
    # Training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="LR for LoRA params (small param count → can be high)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=448,
                   help="Must be a multiple of patch_size (16)")
    p.add_argument("--num_workers", type=int, default=2)
    # LoRA
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_targets", type=str, nargs="+",
                   default=list(DEFAULT_HF_TARGETS),
                   help="Attention sub-modules to adapt: q_proj k_proj v_proj o_proj")
    # Loss
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--diversity_weight", type=float, default=0.0,
                   help="0.0 disables the redundancy-reduction term (default).")
    p.add_argument("--no_hard_negatives", action="store_true",
                   help="Use plain InfoNCE instead of HardInfoNCE.")
    # Misc
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =====================================================================
#  Train loop
# =====================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    img_size: int,
):
    model.train()
    total_loss = 0.0
    total_contrastive = 0.0
    total_diversity = 0.0
    num_valid = 0
    step = 0

    for batch in dataloader:
        if not batch:
            continue

        # ── True mini-batch forward (was: serial loop, defeating batch_size) ──
        # Stack all 2*B images into a single (2B, 3, H, W) tensor and run ONE
        # forward pass. Then split into A / B halves and gather the patch
        # descriptors at the correspondence indices. The per-image patch
        # indices are offset by `b * N_patches` so that, when concatenated,
        # they index into a global (sum_M, D) tensor — letting the existing
        # contrastive loss treat negatives from OTHER images in the same
        # batch (Inter-Image InfoNCE), which is exactly what large batch
        # sizes are supposed to enable.
        valid = [s for s in batch if len(s["idx_a"]) > 0]
        if not valid:
            continue
        B = len(valid)

        imgs_a = torch.stack([s["img_a"] for s in valid], dim=0).to(device)   # (B, 3, H, W)
        imgs_b = torch.stack([s["img_b"] for s in valid], dim=0).to(device)
        imgs   = torch.cat([imgs_a, imgs_b], dim=0)                           # (2B, 3, H, W)

        feats     = model(imgs)                                # (2B, N, D)
        N_patches = feats.shape[1]
        feats_a   = feats[:B]                                  # (B, N, D)
        feats_b   = feats[B:]                                  # (B, N, D)

        # Concatenate per-sample correspondence indices, offsetting each
        # image by b*N so that idx[b]=k → row b*N+k of the flattened (B*N, D)
        # descriptor matrix. This makes safe-radius masking automatically
        # confine itself to within-image neighbours (different images live
        # in disjoint index ranges → distance >> safe_radius), so the same
        # HardInfoNCE code path stays correct.
        idx_a_list, idx_b_list = [], []
        desc_a_list, desc_b_list = [], []
        for b, s in enumerate(valid):
            ia = s["idx_a"].to(device)
            ib = s["idx_b"].to(device)
            idx_a_list.append(ia + b * N_patches)
            idx_b_list.append(ib + b * N_patches)
            desc_a_list.append(feats_a[b, ia])                 # (M_b, D)
            desc_b_list.append(feats_b[b, ib])

        idx_a   = torch.cat(idx_a_list,  dim=0)                # (sum_M,)
        idx_b   = torch.cat(idx_b_list,  dim=0)
        desc_a  = torch.cat(desc_a_list, dim=0)                # (sum_M, D)
        desc_b  = torch.cat(desc_b_list, dim=0)
        if desc_a.shape[0] == 0:
            continue

        losses = criterion(
            desc_a, desc_b, idx_a, idx_b,
            model.patch_size, img_size,
        )
        loss = losses["total"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss        += loss.item()
        total_contrastive += losses["contrastive"].item()
        if "diversity" in losses:
            total_diversity += losses["diversity"].item()
        num_valid += 1
        step += 1

        if step % LOG_EVERY == 0:
            print(
                f"  [epoch {epoch}] step {step} | "
                f"loss={loss.item():.4f} | "
                f"contrastive={losses['contrastive'].item():.4f} | "
                f"B={B} corr={desc_a.shape[0]}",
                flush=True,
            )

    if num_valid == 0:
        return {"loss": 0.0, "contrastive": 0.0, "diversity": 0.0,
                "num_valid": 0}

    return {
        "loss": total_loss / num_valid,
        "contrastive": total_contrastive / num_valid,
        "diversity": total_diversity / num_valid,
        "num_valid": num_valid,
    }


def main():
    args = get_args()

    torch.manual_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # img_size must be a multiple of patch_size; we infer patch_size from
    # the model after it is built, so re-validate then.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )

    # Build model
    print("Building LoRA-HF model...")
    model = LoRADINOv3MatcherHF(
        weights_dir=args.weights_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=tuple(args.lora_targets),
    ).to(device)

    if args.img_size % model.patch_size != 0:
        raise ValueError(
            f"img_size ({args.img_size}) must be divisible by patch_size "
            f"({model.patch_size})"
        )

    # Dataset
    print("Loading dataset...")
    dataset = MatchingPairDataset(
        pairs_path=args.train_pairs,
        data_root=args.data_root,
        depth_root=args.depth_root,
        img_size=args.img_size,
        training=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_matching_pairs,
        drop_last=True,
    )

    # Loss
    criterion = MatchingLoss(
        temperature=args.temperature,
        use_hard_negatives=(not args.no_hard_negatives),
        diversity_weight=args.diversity_weight,   # default 0.0
    )

    # Optimiser — only LoRA parameters
    lora_params = get_lora_parameters(model)
    if not lora_params:
        raise RuntimeError(
            "No LoRA parameters found. Did `inject_lora_hf` succeed?"
        )
    optimizer = optim.AdamW(
        lora_params, lr=args.lr, weight_decay=args.weight_decay,
    )

    # Cosine LR with warm-up
    steps_per_epoch = max(len(dataset), 1)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = min(2 * steps_per_epoch, total_steps // 5)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Banner
    print("\n" + "=" * 60)
    print(f"  LoRA-HF Training: {args.epochs} epochs")
    print(f"  Pairs: {len(dataset)} | Batch size: {args.batch_size}")
    print(f"  LR: {args.lr} | Rank: {args.lora_rank} "
          f"| Alpha: {args.lora_alpha}")
    print(f"  Targets: {args.lora_targets}")
    print(f"  Feature dim: {model.embed_dim} (native HF DINOv3, no proj head)")
    print(f"  Img size: {args.img_size} (patch={model.patch_size})")
    print(f"  Diversity weight: {args.diversity_weight}")
    print(f"  Warmup steps: {warmup_steps} | Total steps: {total_steps}")
    print("=" * 60 + "\n")

    log_path = output_dir / "training_log.json"
    training_log = []

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        metrics = train_one_epoch(
            model, dataloader, criterion, optimizer, scheduler,
            device, epoch, args.img_size,
        )
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={metrics['loss']:.4f} | "
            f"contrastive={metrics['contrastive']:.4f} | "
            f"valid_pairs={metrics['num_valid']} | "
            f"lr={lr_now:.2e} | time={elapsed:.1f}s"
        )
        training_log.append({
            "epoch": epoch, **metrics,
            "lr": lr_now, "elapsed_s": elapsed,
        })
        log_path.write_text(json.dumps(training_log, indent=2))

        if (epoch + 1) % SAVE_EVERY == 0 or epoch == args.epochs - 1:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "args": vars(args),
                "lora_config": {
                    "rank": args.lora_rank,
                    "alpha": args.lora_alpha,
                    "targets": list(args.lora_targets),
                },
            }
            path = output_dir / f"checkpoint_epoch{epoch:03d}.pth"
            torch.save(ckpt, path)
            torch.save(ckpt, output_dir / "checkpoint_latest.pth")
            print(f"  Saved checkpoint: {path}")

    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
