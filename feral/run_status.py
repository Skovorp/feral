"""Atomic run status for explicit output directories; no external logging."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import warnings
from pathlib import Path
import platform
import sys


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def label_fingerprints(raw: bytes, labels: dict) -> dict:
    """Hash labels bytes and canonical split membership/order, not video contents."""
    splits = json.dumps(labels.get('splits', {}), sort_keys=True, separators=(',', ':')).encode()
    return {'labels_sha256': hashlib.sha256(raw).hexdigest(),
            'splits_sha256': hashlib.sha256(splits).hexdigest()}


def normalize_config_paths(value):
    """Copy config containers, converting paths to checkpoint-safe strings."""
    if isinstance(value, os.PathLike):
        return os.fsdecode(value)
    if isinstance(value, dict):
        return {key: normalize_config_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_config_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_config_paths(item) for item in value)
    return value


def _json_default(value):
    if isinstance(value, os.PathLike):
        return os.fsdecode(value)
    raise TypeError(f'Unsupported run configuration value: {type(value).__name__}')


@contextmanager
def training_status(cfg: dict):
    """Record lifecycle when output_dir is set, refusing to overwrite another run."""
    if not cfg.get('output_dir'):
        yield
        return
    from feral import __version__
    # Validate/snapshot configuration before reserving an output file.
    recipe = json.loads(json.dumps({k: v for k, v in cfg.items() if k != 'wandb'}, default=_json_default))
    root = Path(cfg['output_dir']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {root}')
    path = root / 'run.json'
    # Reserve exclusively: concurrent runs must never share status/checkpoints.
    with path.open('x') as handle:
        handle.write('{}\n')
    record = {'schema_version': 1, 'status': 'running', 'started_at': _now(),
              'feral_version': __version__, 'python': platform.python_version(),
              'run_name': cfg['run_name'], 'output_dir': str(root),
              'recipe': recipe,
              'checkpoints': [], 'outputs': []}
    torch = sys.modules.get('torch')
    if torch is not None:
        record['torch_version'] = torch.__version__
        record['cuda_version'] = torch.version.cuda

    def write():
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        temporary.replace(path)

    try:
        write()
        raw = Path(cfg['data']['label_json']).read_bytes()
        record.update(label_fingerprints(raw, json.loads(raw)))
        write()
        yield
    except BaseException as exc:
        record.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    else:
        record['status'] = 'completed'
    finally:
        record['finished_at'] = _now()
        record['checkpoints'] = sorted(str(p.resolve()) for p in (root / 'checkpoints').glob('*.pt'))
        record['outputs'] = sorted(str(p.resolve()) for p in (root / 'answers').glob('*') if p.is_file())
        try:
            write()
        except Exception as exc:
            if record['status'] != 'failed':
                raise
            warnings.warn(f'Could not write failed run status: {exc}', RuntimeWarning)
