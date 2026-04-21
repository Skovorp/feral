"""Sample script: load V-JEPA 2.1 via torch.hub and run a dummy forward pass.

The facebookresearch/vjepa2 repo exposes torch.hub entrypoints for both
V-JEPA 2 and V-JEPA 2.1. The 2.1 variants all use 384px inputs and come in
four sizes: base, large, giant, gigantic. Each entrypoint returns an
(encoder, predictor) tuple; for downstream classification we only need the
encoder.

Dependencies (per the repo's hubconf.py): torch, timm, einops.
"""

import torch

# All V-JEPA 2.1 entrypoints. Input resolution is 384 for every size.
VJEPA_2_1_VARIANTS = [
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant_384",
    "vjepa2_1_vit_gigantic_384",
]

# V-JEPA 2 variants for reference (256px, except giant_384).
VJEPA_2_VARIANTS = [
    "vjepa2_vit_large",
    "vjepa2_vit_huge",
    "vjepa2_vit_giant",
    "vjepa2_vit_giant_384",
]


def build_vjepa21(variant: str = "vjepa2_1_vit_large_384", *, pretrained: bool = True):
    """Load a V-JEPA 2.1 encoder from torch.hub.

    Returns just the encoder (the predictor head is used only during JEPA
    pretraining and isn't needed for classification).
    """
    encoder, _predictor = torch.hub.load(
        "facebookresearch/vjepa2",
        variant,
        pretrained=pretrained,
        trust_repo=True,
    )
    return encoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant = "vjepa2_1_vit_large_384"

    encoder = build_vjepa21(variant).to(device).eval()

    # V-JEPA expects video tensors shaped (B, C, T, H, W). Defaults from
    # the backbone entrypoints: num_frames=64, patch_size=16, tubelet_size=2.
    # img_size is baked into the variant name (384 for all 2.1 models).
    B, C, T, H, W = 1, 3, 64, 384, 384
    dummy = torch.randn(B, C, T, H, W, device=device)

    with torch.no_grad():
        out = encoder(dummy)

    # Output is a sequence of patch tokens. For 384/16 spatial patches and
    # T/tubelet_size temporal patches: (384/16)**2 * (64/2) = 576 * 32 = 18432 tokens.
    print(f"variant: {variant}")
    print(f"input:   {tuple(dummy.shape)}")
    print(f"output:  {tuple(out.shape)}  # (B, num_tokens, hidden_dim)")
    print(f"hidden_dim (use this for downstream heads): {out.shape[-1]}")


if __name__ == "__main__":
    main()
