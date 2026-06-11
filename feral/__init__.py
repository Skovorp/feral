"""FERAL — Feature Extraction for Recognition of Animal Locomotion.

Public Python API (lazy-loaded so ``import feral`` stays light — heavy deps
like torch are only imported when you touch a symbol that needs them)::

    import feral
    feral.run_training(cfg)                         # train from a config dict
    feral.run_inference_folder(ckpt, video_folder)  # inference on a folder
    cfg = feral.apply_mode(cfg, "fast")             # apply a preset overlay
    from feral import BACKBONES, FeralModel, ClsDataset, validate_labels_json
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("feral")
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "run_training",
    "run_inference_folder",
    "apply_mode",
    "PRESETS",
    "BACKBONES",
    "FeralModel",
    "ClsDataset",
    "validate_labels_json",
]

# name -> (submodule, attribute). Public names deliberately avoid colliding with
# submodule names (e.g. run_training, not "train") so attribute access reaches
# __getattr__ instead of returning the submodule.
_LAZY = {
    "run_training": ("feral.train", "main"),
    "run_inference_folder": ("feral.inference_folder", "run_inference_folder"),
    "apply_mode": ("feral.presets", "apply_mode"),
    "PRESETS": ("feral.presets", "PRESETS"),
    "BACKBONES": ("feral.backbones", "BACKBONES"),
    "FeralModel": ("feral.model", "FeralModel"),
    "ClsDataset": ("feral.dataset", "ClsDataset"),
    "validate_labels_json": ("feral.utils", "validate_labels_json"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module 'feral' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
