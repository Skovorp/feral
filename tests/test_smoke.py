import os
import unittest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from feral.train import main as train_main

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def _build_smoke_cfg(label_json_name):
    with open(os.path.join(REPO_ROOT, 'feral', 'default_config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['run_name'] = 'debug'
    cfg['max_batches'] = 1
    cfg.pop('wandb', None)  # disable wandb
    cfg['mixup_alpha'] = None
    cfg['ema_decay'] = None
    cfg['data']['prefix'] = os.path.join(FIXTURES_DIR, 'videos')
    cfg['data']['label_json'] = os.path.join(FIXTURES_DIR, label_json_name)
    cfg['training']['epochs'] = 1
    cfg['training']['train_bs'] = 1
    cfg['training']['val_bs'] = 1
    cfg['training']['num_workers'] = 0
    cfg['training']['compile'] = False
    cfg['training']['part_warmup'] = 0.0
    return cfg


class TestEndToEndSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        videos_dir = os.path.join(FIXTURES_DIR, 'videos')
        if not os.path.isdir(videos_dir) or not os.listdir(videos_dir):
            raise unittest.SkipTest(
                f"Synthetic fixture videos not found at {videos_dir}. "
                f"Run: python tests/generate_synthetic_dataset.py"
            )

    def test_singlelabel_smoke(self):
        train_main(_build_smoke_cfg('labels_singlelabel.json'))

    def test_multilabel_smoke(self):
        train_main(_build_smoke_cfg('labels_multilabel.json'))


if __name__ == '__main__':
    unittest.main()
