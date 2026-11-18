"""Compatibility for Feral's published checkpoint formats (0.1 onward).

Migrations only rename configuration/weight keys. Learned tensors still load
strictly; missing weights and incompatible architectures must remain errors.
"""
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import PurePosixPath, PureWindowsPath

import torch

from feral.backbones import BACKBONES
from feral.run_status import normalize_config_paths


def read_checkpoint(path):
    """Load weights safely, including pathlib values saved in older training cfgs.

    Map concrete paths to inert pure paths, so Windows checkpoints also open on
    POSIX and vice versa. Never fall back to unrestricted pickle loading.
    """
    with torch.serialization.safe_globals([
        PurePosixPath, PureWindowsPath,
        (PurePosixPath, 'pathlib.PosixPath'),
        (PureWindowsPath, 'pathlib.WindowsPath'),
    ]):
        raw = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(raw, Mapping):
        raise ValueError('Expected a Feral checkpoint dictionary or a bare state_dict.')
    return raw


def normalize_checkpoint_config(cfg):
    """Copy a training cfg and translate the pre-backbone-registry model name."""
    cfg = normalize_config_paths(deepcopy(cfg))
    model_name = cfg.pop('model_name', None)
    if model_name is not None:
        aliases = {entry['slug']: key for key, entry in BACKBONES.items()
                   if entry['source'] == 'hf'}
        if model_name not in aliases:
            raise ValueError(f'Unsupported legacy model_name {model_name!r}.')
        backbone = aliases[model_name]
        if cfg.get('backbone', backbone) != backbone:
            raise ValueError('Checkpoint config has conflicting model_name and backbone.')
        cfg['backbone'] = backbone
    cfg.setdefault('training', {}).setdefault('compile', False)
    cfg.setdefault('data', {}).setdefault('resize_style', 'square')
    return cfg


def legacy_inference_config():
    """The inference defaults used before checkpoints embedded cfg (0.1/0.2).

    Keep these independent of today's training defaults. Custom training
    settings cannot be recovered from weights; callers can supply training_cfg.
    """
    return {
        'backbone': 'vjepa2_vitl_diving48', 'predict_per_item': 64,
        'model': {'fc_drop_rate': 0.5, 'freeze_encoder_layers': 0},
        'data': {'resize_to': 256, 'resize_style': 'square',
                 'chunk_length': 64, 'chunk_shift': 32, 'chunk_step': 1},
        'training': {'compile': False},
    }


def checkpoint_state_dict(raw):
    """Unwrap compile/DDP prefixes and migrate HFModel's old encoder namespace."""
    state = raw.get('state_dict', raw)
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Checkpoint state_dict must be a non-empty mapping of tensors.')

    def rename(key):
        while key.startswith(('_orig_mod.', 'module.')):
            key = key.split('.', 1)[1]
        if key.startswith('model.'):
            key = 'backbone.' + key
        return key

    migrated = OrderedDict()
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError('Checkpoint state_dict must contain only named tensors.')
        target = rename(key)
        if target in migrated:
            raise ValueError(f'Checkpoint contains conflicting weights for {target!r}.')
        migrated[target] = value
    # Preserve PyTorch module versions (notably BatchNorm) when renaming keys.
    if hasattr(state, '_metadata'):
        migrated._metadata = OrderedDict(
            (rename(key + '.').rstrip('.') if key else '', value)
            for key, value in state._metadata.items()
        )
    return migrated


def checkpoint_metadata(raw, labels_json=None):
    """Read class metadata, optionally filling old checkpoints from labels.json."""
    if 'state_dict' not in raw:
        stored = {}
    else:
        stored = raw
    metadata = {key: stored.get(key) for key in ('class_names', 'is_multilabel')}
    if labels_json is not None:
        if not isinstance(labels_json, Mapping):
            raise ValueError('labels.json must contain an object with class_names and is_multilabel.')
        for key in metadata:
            fallback = labels_json.get(key)
            existing = metadata[key]
            # JSON uses string class IDs; checkpoints historically used integers.
            if key == 'class_names' and existing is not None and fallback is not None:
                if not isinstance(existing, dict) or not isinstance(fallback, dict):
                    raise ValueError('class_names must be a dictionary keyed by class ID.')
                matches = {str(k): v for k, v in existing.items()} == {str(k): v for k, v in fallback.items()}
            else:
                matches = existing == fallback
            if existing is not None and fallback is not None and not matches:
                raise ValueError(f'labels.json {key} disagrees with the checkpoint.')
            if existing is None:
                metadata[key] = fallback
    if all(value is None for value in metadata.values()):
        return None
    names = metadata['class_names']
    if (not isinstance(names, dict) or not names
            or {str(k) for k in names} != {str(i) for i in range(len(names))}
            or not all(isinstance(v, str) for v in names.values())):
        raise ValueError('Checkpoint needs class_names with consecutive class IDs; provide --labels-json labels.json.')
    if not isinstance(metadata['is_multilabel'], bool):
        raise ValueError('Checkpoint needs is_multilabel; provide --labels-json labels.json.')
    metadata['cfg'] = stored.get('cfg')
    return metadata
