"""
LoRA injection for HuggingFace DINOv3 ViT models.

The HF `Dinov3ViTModel` uses **separate** Linear modules for Q / K / V
(i.e. `attention.q_proj`, `attention.k_proj`, `attention.v_proj`,
`attention.o_proj`). This is different from the original DINOv3 .pth
implementation which uses a fused `qkv` Linear.

This module mirrors `finetune/lora.py` but is adapted to the HF naming.

Usage:
    from finetune.lora_hf import inject_lora_hf, get_lora_parameters

    n = inject_lora_hf(hf_model, rank=4, alpha=1.0,
                       target_modules=("q_proj", "v_proj"))
    lora_params = get_lora_parameters(hf_model)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .lora import LoRALinear  # reuse the same LoRALinear (zero-init B)


# Default LoRA targets: only Q and V (LoRA paper finding) keeps memory low
# and is empirically as strong as Q/K/V for ViT adaptation.
DEFAULT_HF_TARGETS = ("q_proj", "v_proj")


def _iter_attention_modules(hf_model: nn.Module):
    """
    Yield (name, attention_module) for every transformer block in a HF
    DINOv3 model.

    HF stores blocks as `model.layer[i].attention` where `attention` has
    `q_proj`, `k_proj`, `v_proj`, `o_proj` linears.
    """
    # Common access paths across transformers versions.
    candidates = []
    for attr in ["layer", "encoder", "blocks"]:
        m = getattr(hf_model, attr, None)
        if m is not None and isinstance(m, (nn.ModuleList, nn.Sequential)):
            candidates.append((attr, m))
        # Some HF models nest as model.encoder.layer
        if m is not None and hasattr(m, "layer"):
            inner = getattr(m, "layer")
            if isinstance(inner, (nn.ModuleList, nn.Sequential)):
                candidates.append((f"{attr}.layer", inner))

    if not candidates:
        raise RuntimeError(
            "Could not find a ModuleList of transformer blocks on the "
            "given HF model. Available attrs: "
            f"{[n for n, _ in hf_model.named_children()]}"
        )

    # Prefer the first plausible one (HF Dinov3ViTModel exposes `.layer`).
    prefix, blocks = candidates[0]
    for i, block in enumerate(blocks):
        if hasattr(block, "attention"):
            yield f"{prefix}.{i}.attention", block.attention
        elif hasattr(block, "attn"):
            yield f"{prefix}.{i}.attn", block.attn
        else:
            raise RuntimeError(
                f"Block {i} has no `attention` or `attn` sub-module."
            )


def inject_lora_hf(
    hf_model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    target_modules: tuple[str, ...] = DEFAULT_HF_TARGETS,
) -> int:
    """
    Replace the target Linear sub-modules of every attention block with
    `LoRALinear` adapters. The original Linear weights are frozen
    (handled inside `LoRALinear`).

    Args:
        hf_model: a `transformers.Dinov3ViTModel` (or any model whose
            transformer blocks expose an `attention` sub-module with
            `q_proj`, `k_proj`, `v_proj`, `o_proj`).
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: tuple of attribute names to wrap. Each must be a
            Linear inside the attention module. Common choices:
              ("q_proj", "v_proj")            — LoRA paper recommendation
              ("q_proj", "k_proj", "v_proj")  — covers all keys
              ("q_proj", "k_proj", "v_proj", "o_proj")  — full attention

    Returns:
        Number of LoRA parameters added (rough count).
    """
    n_added = 0
    n_blocks = 0

    for _path, attn in _iter_attention_modules(hf_model):
        n_blocks += 1
        for name in target_modules:
            if not hasattr(attn, name):
                raise AttributeError(
                    f"Attention module has no `{name}`. "
                    f"Available: {[c for c, _ in attn.named_children()]}"
                )
            original = getattr(attn, name)
            if not isinstance(original, nn.Linear):
                raise TypeError(
                    f"Target `{name}` is not nn.Linear "
                    f"(got {type(original).__name__})."
                )
            lora = LoRALinear(original, rank=rank, alpha=alpha)
            setattr(attn, name, lora)
            n_added += rank * (original.in_features + original.out_features)

    print(
        f"[LoRA-HF] Injected adapters into {len(target_modules)} modules x "
        f"{n_blocks} blocks (rank={rank}, alpha={alpha}); "
        f"~{n_added/1e3:.1f}K params"
    )
    return n_added


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect only the LoRA `lora_A` and `lora_B` parameters."""
    return [
        p for n, p in model.named_parameters()
        if ("lora_A" in n) or ("lora_B" in n)
    ]
