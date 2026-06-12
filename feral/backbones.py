"""Registry and loader for supported video backbones.

feral supports three backbone families:
- V-JEPA 2   — loaded via HuggingFace AutoModel (fine-tuned or raw pretrain)
- V-JEPA 2.1 — loaded via torch.hub from facebookresearch/vjepa2
- VideoPrism — loaded via HuggingFace AutoModel from the sposiboh/videoprism-*-pt
               ports of Google DeepMind's VideoPrism (PyTorch).

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
        "hidden_dim": 1664,
        "img_size": 384,
    },
    # VideoPrism — HF AutoModel with trust_remote_code. Inputs are channels-last
    # in [0, 1]; the adapter denormalizes + permutes before forward. LvT variants
    # return a single per-video embedding, so they're not wired in (the feral
    # head expects a per-token sequence).
    "videoprism_v1_base": {
        "source": "videoprism_hf",
        "slug": "sposiboh/videoprism-base-f16r288-pt",
        "hidden_dim": 768,
        "img_size": 288,
    },
    "videoprism_v1_large": {
        "source": "videoprism_hf",
        "slug": "sposiboh/videoprism-large-f8r288-pt",
        "hidden_dim": 1024,
        "img_size": 288,
    },
}

# ImageNet mean/std the dataset normalizes with (see ClsDataset.norm). Used to
# denormalize before forwarding into VideoPrism, which expects raw [0, 1].
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_entry(backbone):
    """Return the BACKBONES config dict for backbone, or raise ValueError if unknown."""
    if backbone not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Supported: {sorted(BACKBONES)}"
        )
    return BACKBONES[backbone]


def get_hidden_dim(backbone):
    """Return the backbone's encoder hidden dimension (token feature size)."""
    return _get_entry(backbone)["hidden_dim"]


def get_img_size(backbone):
    """Return the native input resolution (px) the backbone's weights were trained at."""
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
    """Unified interface over HF (V-JEPA 2 / VideoPrism) and torch.hub (V-JEPA 2.1) encoders.

    Input:  (B, T, C, H, W) — what feral.dataset produces.
    Output: (B, N, D)       — flat sequence of patch tokens.
    """

    def __init__(self, backbone, *, pretrained=True):
        """Build the underlying encoder for backbone.

        Dispatches on the entry's source: HF AutoModel (drops the predictor),
        torch.hub V-JEPA 2.1, or VideoPrism (requires torch >= 2.5 and registers
        ImageNet mean/std buffers for denormalization). pretrained=False builds a
        randomly-initialized architecture from config with no weight download.
        """
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
        elif self.source == "videoprism_hf":
            # VideoPrism's modeling code uses torch.nn.attention.flex_attention,
            # which only exists in torch >= 2.5 (feral's floor is 2.4, which is
            # fine for the V-JEPA backbones). Fail with a clear message instead
            # of a cryptic ModuleNotFoundError deep inside trust_remote_code.
            try:
                from torch.nn.attention import flex_attention  # noqa: F401
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    f"VideoPrism backbones require torch >= 2.5 (for "
                    f"torch.nn.attention.flex_attention); found torch {torch.__version__}. "
                    f"Upgrade torch, or use a V-JEPA 2 / 2.1 backbone."
                ) from e
            from transformers import AutoConfig, AutoModel
            if pretrained:
                self.model = AutoModel.from_pretrained(entry["slug"], trust_remote_code=True)
            else:
                cfg = AutoConfig.from_pretrained(entry["slug"], trust_remote_code=True)
                self.model = AutoModel.from_config(cfg, trust_remote_code=True)
            mean = torch.tensor(_IMAGENET_MEAN).view(1, 1, 1, 1, 3)
            std = torch.tensor(_IMAGENET_STD).view(1, 1, 1, 1, 3)
            self.register_buffer("_vp_mean", mean)
            self.register_buffer("_vp_std", std)
        else:
            raise ValueError(f"unknown source {self.source!r}")

    def forward(self, x):
        """Encode a video batch into a flat patch-token sequence.

        Takes x of shape (B, T, C, H, W) and returns (B, N, D). Per source: HF runs
        with skip_predictor; hub permutes to (B, C, T, H, W); videoprism_hf permutes
        to (B, T, H, W, C) and denormalizes ImageNet stats back to [0, 1] first.
        """
        if self.source == "hf":
            return self.model(x, skip_predictor=True).last_hidden_state
        if self.source == "hub":
            return self.model(x.permute(0, 2, 1, 3, 4))
        # videoprism_hf: (B, T, C, H, W) ImageNet-normalized → (B, T, H, W, C) in [0, 1].
        # The dataset normalizes with ImageNet stats; undo that here so the
        # backbone sees the [0, 1] range it was trained on.
        x = x.permute(0, 1, 3, 4, 2)
        x = x * self._vp_std + self._vp_mean
        return self.model(pixel_values=x).last_hidden_state

    def num_encoder_layers(self):
        """Return the number of transformer encoder layers in the underlying model."""
        if self.source == "hf":
            return len(self.model.encoder.layer)
        if self.source == "hub":
            return len(self.model.blocks)
        # videoprism_hf: the inner FactorizedEncoder lives at model.videoprism.
        return len(self.model.videoprism.spatial_encoder.x_layers)

    def freeze_encoder(self, n_layers):
        """Freeze the patch embedding and the first n_layers encoder layers.

        Sets requires_grad=False on those params (raising ValueError if n_layers
        exceeds the available count). For videoprism_hf only the spatial encoder is
        frozen; its 4-layer temporal stack stays trainable.
        """
        def _freeze(m):
            """Set requires_grad=False on all parameters of module m."""
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
        elif self.source == "hub":
            _freeze(self.model.patch_embed)
            for i in range(n_layers):
                _freeze(self.model.blocks[i])
        else:
            # videoprism_hf: freeze patch projection + first n spatial blocks.
            # The 4-layer temporal stack stays trainable.
            inner = self.model.videoprism
            _freeze(inner.patch_projection)
            for i in range(n_layers):
                _freeze(inner.spatial_encoder.x_layers[i])
