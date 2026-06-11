"""Named training/inference presets ("modes") overlaid on default_config.yaml.

A preset is a *sparse* overlay — it only names the keys it changes, and is
deep-merged onto the packaged ``default_config.yaml`` so the base recipe stays
the single source of truth. Exposed on the CLI via ``feral train --mode`` and
``feral infer --mode``.

Modes
-----
fast : smallest V-JEPA 2.1 (ViT-B/384) with the Parkinson fine-tuning recipe
       (full fine-tune, 50% chunk overlap). Cheapest to train and run.
max  : largest *runnable* V-JEPA 2.1 (ViT-g/384) with 75% chunk overlap. Best
       accuracy; needs a large-VRAM GPU. (The gigantic ViT-G is available via
       ``train-config`` with ``backbone: vjepa2_1_vitgg_384`` for users who can
       afford it.)
rare : tuned for rare-class / rare-positive datasets — turns EMA and mixup OFF
       (both hurt when positives are scarce) and turns ON the stabilization
       knobs (grad-norm clip + class-weight cap). Built on the fast backbone.
"""

# Sparse overlays, deep-merged onto default_config.yaml.
PRESETS = {
    # ── fast ── faithful to the Parkinson experiment recipe ───────────────────
    "fast": {
        "backbone": "vjepa2_1_vitb_384",   # smallest V-JEPA 2.1, native 384px
        "predict_per_item": 64,
        "model": {
            "fc_drop_rate": 0.5,
            "class_weights": "inv_freq_sqrt",
            "freeze_encoder_layers": 0,     # full fine-tune (Parkinson recipe)
        },
        "data": {
            "chunk_length": 64,
            "chunk_shift": 32,              # 50% overlap
            "chunk_step": 1,
            "resize_to": 384,              # match ViT-B/384 native size
            "resize_style": "square",
            "do_aa": True,
        },
        "training": {
            "epochs": 10,
            "lr": 4.0e-5,
            "weight_decay": 0.1,
            "label_smoothing": 0.1,
            "part_warmup": 0.2,
            "compile": True,
        },
        "mixup_alpha": 0.8,
        "ema_decay": 0.999,
        "multilabel_threshold": 0.85,
    },

    # ── max ── biggest runnable backbone + 75% overlap ────────────────────────
    "max": {
        "backbone": "vjepa2_1_vitg_384",   # giant (~1.4B); gigantic via train-config
        "predict_per_item": 64,            # keep head output == chunk_length (one pred/frame)
        "data": {
            "chunk_length": 64,
            "chunk_shift": 16,             # 75% overlap = chunk_length / 4
            "chunk_step": 1,
            "resize_to": 384,             # match ViT-g/384 native size
            "resize_style": "square",
        },
        # training recipe inherits default_config (freeze 12 keeps the giant
        # trainable on a single large GPU; full fine-tune would OOM for most).
    },

    # ── rare ── rare-class robustness ─────────────────────────────────────────
    "rare": {
        "backbone": "vjepa2_1_vitb_384",
        "predict_per_item": 64,
        "model": {
            "fc_drop_rate": 0.5,
            "class_weights": "inv_freq_sqrt",
            "freeze_encoder_layers": 0,
            "max_class_weight": 20,        # cap extreme inverse-freq weights
        },
        "data": {
            "chunk_length": 64,
            "chunk_shift": 32,
            "chunk_step": 1,
            "resize_to": 384,
            "resize_style": "square",
            "do_aa": True,
        },
        "training": {
            "epochs": 10,
            "lr": 4.0e-5,
            "weight_decay": 0.1,
            "label_smoothing": 0.1,
            "part_warmup": 0.2,
            "compile": True,
            "grad_clip_norm": 1.0,         # stabilize rare-positive grad spikes
        },
        "mixup_alpha": None,               # mixup OFF — hurts with rare positives
        "ema_decay": None,                 # EMA OFF — hurts with rare positives
        "multilabel_threshold": 0.85,
    },
}

# One-line descriptions for CLI --help / logging.
MODE_HELP = {
    "fast": "smallest V-JEPA 2.1 + Parkinson recipe (cheapest)",
    "max":  "largest runnable V-JEPA 2.1 + 75% chunk overlap (best accuracy)",
    "rare": "rare-class robustness: EMA & mixup off + grad-clip + class-weight cap",
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

    fast -> 50% overlap (chunk_length / 2); max -> 75% overlap (chunk_length / 4).
    Returns None for modes that don't change inference chunking (e.g. ``rare``).
    """
    if mode == "fast":
        return max(1, chunk_length // 2)
    if mode == "max":
        return max(1, chunk_length // 4)
    return None
