"""Named training/inference presets ("modes") overlaid on default_config.yaml.

A preset is a *sparse* overlay — it only names the keys it changes, and is
deep-merged onto the packaged ``default_config.yaml`` so the base recipe stays
the single source of truth. Exposed on the CLI via ``feral train --mode`` and
``feral infer --mode``.

``lite`` and ``rare`` disable EMA (``ema_decay: None``) — the small full
fine-tunes they target don't benefit from a weight average. ``max`` turns EMA
back on (``ema_decay: 0.999``), matching the SOTA CalMS21 recipe.

Modes
-----
lite : smallest V-JEPA 2.1 (ViT-B/384), full fine-tune with 50% chunk overlap.
       Cheapest to train and run.
max  : same backbone as ``default`` (no override), but bumps the temporal
       overlap to 66% (``chunk_shift`` 21) and keeps EMA on. This is the recipe
       that, in our comparison experiments, achieves SOTA on CalMS21.
rare : tuned for rare-class / rare-positive datasets — turns mixup and label
       smoothing OFF (both hurt when positives are scarce) and turns ON the
       stabilization knobs (grad-norm clip + class-weight cap). Built on the
       lite backbone.
"""

# Sparse overlays, deep-merged onto default_config.yaml. Each preset names ONLY
# the keys that differ from default_config.yaml; everything else is inherited.
PRESETS = {
    # ── lite ── full fine-tune, smallest backbone ─────────────────────────────
    "lite": {
        "backbone": "vjepa2_1_vitb_384",   # smallest V-JEPA 2.1 (384-native; fed at default 256)
        "model": {
            "freeze_encoder_layers": 0,    # full fine-tune
        },
        "ema_decay": None,                 # EMA OFF
        # resize_to inherits default_config (256) — 384-native backbone runs at
        # 256 via interpolated position embeddings; ~2.25x fewer tokens.
    },

    # ── max ── default backbone + 66% overlap + EMA ─────────────────────────
    "max": {
        "data": {
            "chunk_shift": 21,             # 66% overlap = chunk_length / 3 (default is 50%); SOTA on CalMS21
        },
        "ema_decay": 0.999,                # EMA ON (default disables it; max re-enables)
        # backbone, freeze_encoder_layers, resize_to all inherit default_config.
    },

    # ── rare ── rare-class robustness ─────────────────────────────────────────
    "rare": {
        "backbone": "vjepa2_1_vitb_384",
        "model": {
            "freeze_encoder_layers": 0,
            "max_class_weight": 20,        # cap extreme inverse-freq weights
        },
        "training": {
            "label_smoothing": 0.0,        # label smoothing OFF — hurts with rare positives
            "grad_clip_norm": 1.0,         # stabilize rare-positive grad spikes
        },
        "mixup_alpha": None,               # mixup OFF — hurts with rare positives
        "ema_decay": None,                 # EMA OFF
    },
}

# One-line descriptions for CLI --help / logging.
MODE_HELP = {
    "lite": "smallest V-JEPA 2.1 (ViT-B/384), full fine-tune (cheapest)",
    "max":  "default backbone + 66% chunk overlap + EMA on (SOTA on CalMS21)",
    "rare": "rare-class robustness: EMA, mixup & label smoothing off + grad-clip + class-weight cap",
}


def _deep_merge(base, overlay):
    """Recursively merge ``overlay`` onto ``base``, returning a new dict.

    Nested dicts are merged key-by-key; any non-dict value (including ``None``)
    in ``overlay`` replaces the value in ``base``.
    """
    out = dict(base)
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def apply_mode(cfg, mode):
    """Return ``cfg`` deep-merged with preset ``mode``. Unknown mode -> ValueError."""
    if mode is None:
        return cfg
    if mode not in PRESETS:
        raise ValueError(f"Unknown mode {mode!r}. Choices: {sorted(PRESETS)}")
    return _deep_merge(cfg, PRESETS[mode])


def infer_chunk_shift(mode, chunk_length):
    """Chunk shift (stride) for inference-time overlap under a given mode.

    lite -> 50% overlap (chunk_length / 2); max -> 66% overlap (chunk_length / 3).
    Returns None for modes that don't change inference chunking (e.g. ``rare``).
    """
    if mode == "lite":
        return max(1, chunk_length // 2)
    if mode == "max":
        return max(1, chunk_length // 3)
    return None
