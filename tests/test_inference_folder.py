import json
import os
import tempfile

import pytest
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from feral.train import main as train_main
from feral.inference_folder import run_inference_folder

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
VIDEOS_DIR = os.path.join(FIXTURES_DIR, 'videos')


def _build_smoke_cfg():
    with open(os.path.join(REPO_ROOT, 'feral', 'default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = 'test_inference'
    cfg['max_batches'] = 1
    cfg.pop('wandb', None)
    cfg['mixup_alpha'] = None
    cfg['ema_decay'] = None
    cfg['data']['prefix'] = VIDEOS_DIR
    cfg['data']['label_json'] = os.path.join(FIXTURES_DIR, 'labels_singlelabel.json')
    cfg['training']['epochs'] = 1
    cfg['training']['train_bs'] = 1
    cfg['training']['val_bs'] = 1
    cfg['training']['num_workers'] = 0
    cfg['training']['compile'] = False
    cfg['training']['part_warmup'] = 0.0
    return cfg


_skip_no_fixtures = pytest.mark.skipif(
    not os.path.isdir(VIDEOS_DIR) or not os.listdir(VIDEOS_DIR),
    reason=f"Synthetic fixture videos not found at {VIDEOS_DIR}. Run: python tests/generate_synthetic_dataset.py",
)

_checkpoint_path = os.path.join(REPO_ROOT, 'checkpoints', 'test_inference_best_checkpoint.pt')


@pytest.fixture(scope="module")
def trained_checkpoint():
    train_main(_build_smoke_cfg())
    return _checkpoint_path


@_skip_no_fixtures
def test_inference_folder(trained_checkpoint):
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    try:
        run_inference_folder(
            checkpoint_path=trained_checkpoint,
            video_folder=VIDEOS_DIR,
            output=output_path,
            batch_size=1,
            num_workers=0,
        )

        assert os.path.isfile(output_path)

        with open(output_path) as f:
            results = json.load(f)

        assert 'preds' in results
        assert len(results['preds']) > 0
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


@_skip_no_fixtures
def test_checkpoint_embeds_training_cfg(trained_checkpoint):
    """The saved checkpoint must carry the full training cfg so downstream
    inference can read data/model params without default_config.yaml."""
    raw = torch.load(trained_checkpoint, map_location="cpu")
    assert 'cfg' in raw, "checkpoint missing 'cfg' key"
    stored = raw['cfg']
    # Sanity: the cfg we trained with should be echoed back.
    assert stored['data']['resize_to'] == 256
    assert stored['data']['resize_style'] == 'square'
    assert stored['data']['chunk_length'] == 64
    # run_name we set in _build_smoke_cfg
    assert stored['run_name'] == 'test_inference'


@_skip_no_fixtures
def test_inference_uses_stored_cfg_not_default(trained_checkpoint, monkeypatch):
    """When the checkpoint has an embedded cfg, inference must use it and must
    NOT fall back to default_config.yaml. We prove that by patching the stored
    cfg with a distinctive resize_style and asserting the dataset reflects it —
    while stubbing out the actual model forward pass (size-incompatible with a
    real run)."""
    import feral.inference_folder as inf_mod

    raw = torch.load(trained_checkpoint, map_location="cpu")
    raw['cfg']['data']['resize_style'] = 'rectangle'
    patched_path = trained_checkpoint + '.patched.pt'
    torch.save(raw, patched_path)

    captured = {}
    real_ctor = inf_mod.ClsDataset

    def spy_ctor(*args, **kwargs):
        captured['resize_to'] = kwargs.get('resize_to')
        captured['resize_style'] = kwargs.get('resize_style')
        return real_ctor(*args, **kwargs)

    monkeypatch.setattr(inf_mod, 'ClsDataset', spy_ctor)

    def boom(*a, **k):
        raise AssertionError("inference should not read default_config when checkpoint has a cfg")
    monkeypatch.setattr(inf_mod, '_load_default_cfg', boom)

    # Stub the actual forward pass — we only care about cfg plumbing here.
    monkeypatch.setattr(inf_mod, 'run_inference', lambda *a, **k: {})
    monkeypatch.setattr(inf_mod, 'save_inference_results', lambda *a, **k: None)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    try:
        run_inference_folder(
            checkpoint_path=patched_path,
            video_folder=VIDEOS_DIR,
            output=output_path,
            batch_size=1,
            num_workers=0,
        )
        assert captured['resize_to'] == 256  # unchanged from training
        assert captured['resize_style'] == 'rectangle'  # picked up from patched cfg
    finally:
        for p in (output_path, patched_path):
            if os.path.exists(p):
                os.unlink(p)
