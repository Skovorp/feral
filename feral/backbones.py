"""Registry and loader for supported video backbones.

feral supports two backbone families:
- V-JEPA 2   — loaded via HuggingFace AutoModel (fine-tuned or raw pretrain)
- V-JEPA 2.1 — loaded via torch.hub from facebookresearch/vjepa2

Each key in BACKBONES maps to everything needed to build the backbone: how to
load it, its hidden dimension, and the native input resolution its weights
were trained at. FeralModel reads hidden_dim from here to size its head, and
training/inference entrypoints use img_size to warn on resize_to mismatches.
"""
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

BACKBONES = {
    # V-JEPA 2 — HuggingFace slugs.
    "vjepa2_vitl_diving48": {
        "source": "hf",
        "slug": "facebook/vjepa2-vitl-fpc32-256-diving48",
        "hidden_dim": 1024,
        "img_size": 256,
    },
    "vjepa2_vitl_ssv2": {
        "source": "hf",
        "slug": "facebook/vjepa2-vitl-fpc16-256-ssv2",
        "hidden_dim": 1024,
        "img_size": 256,
    },
    "vjepa2_vitl": {
        "source": "hf",
        "slug": "facebook/vjepa2-vitl-fpc64-256",
        "hidden_dim": 1024,
        "img_size": 256,
    },
    # V-JEPA 2.1 — torch.hub (all 384px).
    "vjepa2_1_vitb_384": {
        "source": "hub",
        "hub_name": "vjepa2_1_vit_base_384",
        "hidden_dim": 768,
        "img_size": 384,
    },
    "vjepa2_1_vitl_384": {
        "source": "hub",
        "hub_name": "vjepa2_1_vit_large_384",
        "hidden_dim": 1024,
        "img_size": 384,
    },
    "vjepa2_1_vitg_384": {
        "source": "hub",
        "hub_name": "vjepa2_1_vit_giant_384",
        "hidden_dim": 1408,
        "img_size": 384,
    },
    "vjepa2_1_vitgg_384": {
        "source": "hub",
        "hub_name": "vjepa2_1_vit_gigantic_384",
        "hidden_dim": 1792,
        "img_size": 384,
    },
}


def _get_entry(backbone):
    if backbone not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Supported: {sorted(BACKBONES)}"
        )
    return BACKBONES[backbone]


def get_hidden_dim(backbone):
    return _get_entry(backbone)["hidden_dim"]


def get_img_size(backbone):
    return _get_entry(backbone)["img_size"]


def warn_if_resize_mismatch(cfg):
    """Log a warning if cfg['data']['resize_to'] doesn't match the backbone's native img_size."""
    backbone = cfg["backbone"]
    native = get_img_size(backbone)
    resize_to = cfg["data"]["resize_to"]
    if resize_to != native:
        logger.warning(
            "resize_to=%d does not match backbone %r native img_size=%d — "
            "this is unusual and may hurt accuracy.",
            resize_to, backbone, native,
        )


class BackboneAdapter(nn.Module):
    """Unified interface over HF (V-JEPA 2) and torch.hub (V-JEPA 2.1) encoders.

    Input:  (B, T, C, H, W) — what feral.dataset produces.
    Output: (B, N, D)       — flat sequence of patch tokens.
    """

    def __init__(self, backbone, *, pretrained=True):
        super().__init__()
        entry = _get_entry(backbone)
        self.source = entry["source"]
        self.hidden_dim = entry["hidden_dim"]

        if self.source == "hf":
            from transformers import AutoConfig, AutoModel
            if pretrained:
                self.model = AutoModel.from_pretrained(entry["slug"])
            else:
                # Config-only — no weight download. Architecture is initialized
                # randomly; useful for shape/wiring tests.
                self.model = AutoModel.from_config(AutoConfig.from_pretrained(entry["slug"]))
            self.model.predictor = None
        elif self.source == "hub":
            encoder, _predictor = torch.hub.load(
                "facebookresearch/vjepa2",
                entry["hub_name"],
                pretrained=pretrained,
                trust_repo=True,
            )
            self.model = encoder
        else:
            raise ValueError(f"unknown source {self.source!r}")

    def forward(self, x):
        if self.source == "hf":
            return self.model(x, skip_predictor=True).last_hidden_state
        return self.model(x.permute(0, 2, 1, 3, 4))

    def num_encoder_layers(self):
        if self.source == "hf":
            return len(self.model.encoder.layer)
        return len(self.model.blocks)

    def freeze_encoder(self, n_layers):
        def _freeze(m):
            for p in m.parameters():
                p.requires_grad = False

        total = self.num_encoder_layers()
        if n_layers > total:
            raise ValueError(
                f"freeze_encoder_layers={n_layers} exceeds {total} available layers"
            )
        if self.source == "hf":
            _freeze(self.model.encoder.embeddings)
            for i in range(n_layers):
                _freeze(self.model.encoder.layer[i])
        else:
            _freeze(self.model.patch_embed)
            for i in range(n_layers):
                _freeze(self.model.blocks[i])
