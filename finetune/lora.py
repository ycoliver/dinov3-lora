"""
LoRA (Low-Rank Adaptation) for DINOv3 ViT backbone.

Key design: Instead of adding a random Projection Head that destroys DINO's
pre-trained features, LoRA injects small trainable low-rank matrices into the
existing QKV attention projections. This preserves the original feature space
while allowing geometry-aware fine-tuning.

Why LoRA solves the mode collapse problem:
  1. LoRA's B matrix is initialised to ZERO → at training start, the model
     outputs are IDENTICAL to the pre-trained DINOv3 (no random noise).
  2. The output stays in the original 1024-dim feature space → no information
     loss from a random projection to 256-dim.
  3. Much fewer parameters → less prone to overfitting with small batch size.
  4. Can be applied to ALL 24 blocks → much broader adaptation than only
     unfreezing the last 2 blocks.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA adapter that wraps an existing nn.Linear layer.

    The output becomes: original_output + x @ A^T @ B^T * scaling
    where A is (rank, in_features) and B is (out_features, rank).

    B is initialised to zero so the adapter has NO effect at the start.
    """

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Match the device/dtype of the wrapped Linear, so that injecting
        # LoRA AFTER `model.to(device)` still puts A/B on the right device.
        w = original.weight
        device = w.device
        dtype = w.dtype

        # A: random init (Kaiming); B: zero init → output starts at zero
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A)

        # Freeze the original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward (frozen)
        out = self.original(x)
        # LoRA residual: x @ A^T @ B^T * scaling
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return out + lora_out


def inject_lora(
    backbone: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: tuple[str, ...] = ("qkv", "proj"),
) -> int:
    """
    Inject LoRA adapters into the backbone's attention layers.

    Args:
        backbone: DINOv3Backbone instance.
        rank: LoRA rank (lower = fewer params, higher = more expressive).
        alpha: LoRA scaling factor.
        target_modules: which Linear layers inside each Block to adapt.
            "qkv" = the Q/K/V projection (most impactful for feature adaptation).
            "proj" = the output projection after attention.

    Returns:
        Number of LoRA parameters added.
    """
    lora_params = 0

    for block in backbone.blocks:
        for name in target_modules:
            if name == "qkv":
                original = block.attn.qkv
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                block.attn.qkv = lora_layer
                lora_params += rank * original.in_features + original.out_features * rank
            elif name == "proj":
                original = block.attn.proj
                lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                block.attn.proj = lora_layer
                lora_params += rank * original.in_features + original.out_features * rank

    return lora_params


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect only the LoRA A and B parameters from the model."""
    params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            params.append(param)
    return params
