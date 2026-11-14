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
import contextlib
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

# torch.hub and trust_remote_code both execute code fetched at load time, so
# each source is pinned to an exact commit. Bump these deliberately.
_VJEPA2_HUB_REPO = "facebookresearch/vjepa2:204698b45b3712590f06245fbfba32d3be539812"

# Upstream vjepa2 ships a leftover test override (VJEPA_BASE_URL pointing at
# localhost:8300), so every V-JEPA 2.1 weight download fails out of the box.
_VJEPA2_BROKEN_BASE_URL = "http://localhost:8300"
_VJEPA2_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"


@contextlib.contextmanager
def _vjepa2_weights_from_cdn():
    """Redirect vjepa2's weight downloads from its test URL to the public CDN."""
    original = torch.hub.load_state_dict_from_url

    def load_state_dict_from_url(url, *args, **kwargs):
        url = url.replace(_VJEPA2_BROKEN_BASE_URL, _VJEPA2_BASE_URL)
        return original(url, *args, **kwargs)

    torch.hub.load_state_dict_from_url = load_state_dict_from_url
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = original

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
        "revision": "82d97c3e1c42b1e960801857541892fb4d7e9aff",
        "hidden_dim": 768,
        "img_size": 288,
    },
    "videoprism_v1_large": {
        "source": "videoprism_hf",
        "slug": "sposiboh/videoprism-large-f8r288-pt",
        "revision": "acc4521b968bba2700a192566f3bfdc54285bd28",
        "hidden_dim": 1024,
        "img_size": 288,
    },
}

# ImageNet mean/std the loops normalize with (see dataset.normalize_frames). Used to
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

    def __init__(self, backbone, *, pretrained=True, gradient_checkpointing=False):
        """Build the underlying encoder for backbone.

        Dispatches on the entry's source: HF AutoModel (drops the predictor),
        torch.hub V-JEPA 2.1, or VideoPrism (registers
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
            if gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        elif self.source == "hub":
            # skip_validation: torch.hub only accepts a commit SHA that is a
            # branch or tag head, which stops being true once upstream moves.
            with _vjepa2_weights_from_cdn():
                encoder, _predictor = torch.hub.load(
                    _VJEPA2_HUB_REPO,
                    entry["hub_name"],
                    pretrained=pretrained,
                    trust_repo=True,
                    skip_validation=True,
                )
            self.model = encoder
            if gradient_checkpointing:
                self.model.use_activation_checkpointing = True
        elif self.source == "videoprism_hf":
            if gradient_checkpointing:
                raise ValueError(
                    "gradient_checkpointing is not supported for VideoPrism backbones. "
                    "Use a V-JEPA 2 or V-JEPA 2.1 backbone, or disable gradient_checkpointing."
                )
            from transformers import AutoConfig, AutoModel
            slug, revision = entry["slug"], entry["revision"]
            if pretrained:
                self.model = AutoModel.from_pretrained(slug, trust_remote_code=True, revision=revision)
            else:
                cfg = AutoConfig.from_pretrained(slug, trust_remote_code=True, revision=revision)
                # from_config takes the modeling code's commit as code_revision.
                self.model = AutoModel.from_config(cfg, trust_remote_code=True, code_revision=revision)
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
