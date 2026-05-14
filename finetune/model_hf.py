"""
DINOv3 backbone wrapper using HuggingFace transformers.

Loads `Dinov3ViTModel` from a local HF safetensors checkpoint
(e.g. `dinov3_weights/`) and exposes a clean interface that returns
L2-normalised patch tokens as `(B, N_patches, D)`.

Why this file (vs `model.py`):
  * `model.py` re-implements ViT-L/16 from scratch and loads the original
    `.pth` checkpoint. That code path does NOT match the HF safetensors
    format (different state-dict keys, different RoPE parameterisation).
  * This wrapper delegates everything to `transformers`, so it works
    out-of-the-box with HF safetensors weights and any size of DINOv3 ViT
    (S / S+ / B / L / H+ / 7B).

Usage:
    from finetune.model_hf import DINOv3HFBackbone

    backbone = DINOv3HFBackbone(weights_dir="dinov3_weights")
    feats = backbone(images)            # (B, N_patches, D), L2-normalised
    n_layers = backbone.num_hidden_layers
    embed_dim = backbone.embed_dim
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_hf_dinov3(weights_dir: str | Path):
    """
    Load `Dinov3ViTModel` from a local HF directory.

    Requires `transformers >= 4.56` (DINOv3 was added in 4.56).
    """
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "transformers is required for the HF DINOv3 path. "
            "Install with: pip install 'transformers>=4.56'"
        ) from e

    weights_dir = str(Path(weights_dir).resolve())
    model = AutoModel.from_pretrained(weights_dir)
    return model


class DINOv3HFBackbone(nn.Module):
    """
    Thin wrapper around `transformers.Dinov3ViTModel`.

    Forward returns L2-normalised patch tokens with shape
    `(B, num_patches, embed_dim)`. The CLS token and the register
    tokens are stripped automatically.
    """

    def __init__(self, weights_dir: str | Path):
        super().__init__()
        self.hf_model = _load_hf_dinov3(weights_dir)

        cfg = self.hf_model.config
        self.patch_size: int = int(cfg.patch_size)
        self.embed_dim: int = int(cfg.hidden_size)
        self.num_hidden_layers: int = int(cfg.num_hidden_layers)
        self.num_register_tokens: int = int(getattr(cfg, "num_register_tokens", 0))
        self.num_attention_heads: int = int(cfg.num_attention_heads)

        # In HF DINOv3, the prefix tokens are: [CLS, register_1, ..., register_R]
        # so the patch tokens start at index 1 + num_register_tokens.
        self._n_prefix = 1 + self.num_register_tokens

    def forward_features(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, H, W). H and W must be divisible by `patch_size`.

        Returns:
            dict with keys:
              x_norm_patchtokens: (B, num_patches, D)
              x_norm_clstoken:    (B, 1, D)
        """
        out = self.hf_model(pixel_values=x)
        # `last_hidden_state` already passed through the final LayerNorm.
        h = out.last_hidden_state  # (B, 1 + R + N_patches, D)

        cls_token = h[:, 0:1, :]
        patch_tokens = h[:, self._n_prefix:, :]

        return {
            "x_norm_patchtokens": patch_tokens,
            "x_norm_clstoken": cls_token,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalised patch descriptors with shape
        `(B, num_patches, D)`.
        """
        feats = self.forward_features(x)["x_norm_patchtokens"]
        return F.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def get_patch_coords(self, H: int, W: int) -> torch.Tensor:
        """
        Pixel coordinates `(x, y)` of every patch centre for an image of
        shape `(H, W)`. Returns a tensor of shape `(h*w, 2)` in `float32`.
        """
        ps = self.patch_size
        h, w = H // ps, W // ps
        ys = torch.arange(h) * ps + ps // 2
        xs = torch.arange(w) * ps + ps // 2
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).float()
