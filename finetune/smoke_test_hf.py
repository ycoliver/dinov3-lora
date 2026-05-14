"""
Smoke test for the LoRA-HF DINOv3 matching pipeline.

This test exercises the *minimum* path that touches every new file
written for the HF route:

  1. Load the HF DINOv3 backbone from `dinov3_weights/`.
  2. Run a forward pass on a random image at the configured `img_size`.
  3. Inject LoRA adapters into Q/V projections.
  4. Verify that BEFORE training (B is zero), the LoRA-injected model
     produces IDENTICAL outputs to the bare backbone.
  5. Run MNN matching on a random pair to make sure shapes work out.

If this script runs end-to-end without error, the pipeline is ready
for real training data.

Usage:
    python -m finetune.smoke_test_hf
    python -m finetune.smoke_test_hf --img_size 224     # smaller / faster
"""

from __future__ import annotations

import argparse

import torch

from .model_hf import DINOv3HFBackbone
from .lora_hf import inject_lora_hf, get_lora_parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_dir", type=str, default="dinov3_weights")
    p.add_argument("--img_size", type=int, default=224,
                   help="Use 224 for a fast CPU smoke test.")
    p.add_argument("--lora_rank", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if (hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available())
              else "cpu")
    )
    print(f"Device : {device}")

    # 1) load
    print("\n[1] Loading HF DINOv3 backbone...")
    backbone = DINOv3HFBackbone(weights_dir=args.weights_dir).to(device).eval()
    print(f"    patch_size={backbone.patch_size}  "
          f"embed_dim={backbone.embed_dim}  "
          f"depth={backbone.num_hidden_layers}  "
          f"register_tokens={backbone.num_register_tokens}")

    # 2) forward
    if args.img_size % backbone.patch_size != 0:
        raise ValueError(
            f"img_size {args.img_size} must be divisible by "
            f"patch_size {backbone.patch_size}"
        )
    H = W = args.img_size
    n_patches = (H // backbone.patch_size) * (W // backbone.patch_size)
    print(f"\n[2] Forward pass on a random {H}x{W} image...")
    x = torch.randn(1, 3, H, W, device=device)
    with torch.no_grad():
        feats = backbone(x)
    expected = (1, n_patches, backbone.embed_dim)
    assert tuple(feats.shape) == expected, \
        f"unexpected output shape {tuple(feats.shape)} vs {expected}"
    print(f"    OK  output shape = {tuple(feats.shape)}")
    print(f"    OK  L2 norm sample = "
          f"{feats[0, 0].norm().item():.4f} (should be ~1.0)")

    # snapshot zero-shot output for the equivalence check
    zs_out = feats.clone()

    # 3) inject LoRA
    print("\n[3] Injecting LoRA adapters (rank={})..."
          .format(args.lora_rank))
    inject_lora_hf(
        backbone.hf_model,
        rank=args.lora_rank,
        alpha=1.0,
        target_modules=("q_proj", "v_proj"),
    )
    # safety: make sure newly added LoRA params live on the same device
    backbone.to(device)
    n_lora = sum(p.numel() for p in get_lora_parameters(backbone))
    n_total = sum(p.numel() for p in backbone.parameters())
    print(f"    LoRA params = {n_lora/1e3:.1f}K / "
          f"{n_total/1e6:.1f}M total")

    # 4) zero-init equivalence check
    print("\n[4] Verifying LoRA(B=0) output == zero-shot output...")
    with torch.no_grad():
        lora_out = backbone(x)
    delta = (lora_out - zs_out).abs().max().item()
    print(f"    max |Δ| = {delta:.2e}")
    assert delta < 1e-5, (
        "LoRA-injected model output differs from zero-shot at init; "
        "B matrix is supposed to be zero."
    )
    print("    OK  outputs are identical at init.")

    # 5) MNN matching shape check
    print("\n[5] MNN matching shape check on a random pair...")
    with torch.no_grad():
        a = torch.nn.functional.normalize(
            torch.randn(1, 3, H, W, device=device), dim=1)
        b = torch.nn.functional.normalize(
            torch.randn(1, 3, H, W, device=device), dim=1)
        da = backbone(a)[0]
        db = backbone(b)[0]
    sim = da @ db.t()
    nn_b = sim.argmax(dim=1)
    nn_a = sim.argmax(dim=0)
    mutual = (nn_a[nn_b] == torch.arange(da.shape[0], device=device)).sum()
    print(f"    sim shape   = {tuple(sim.shape)}")
    print(f"    MNN matches = {mutual.item()} / {da.shape[0]}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
