import json
import os
import tempfile

import pytest
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
