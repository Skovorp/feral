"""Training smoke limits must not truncate held-out evaluation."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pytest


@pytest.mark.parametrize('train_limit,legacy_limit', [(1, None), (None, 3)])
def test_train_limit_and_provenance(tmp_path, monkeypatch, train_limit, legacy_limit):
    import feral.train as train
    from feral import load_default_config
    cfg = load_default_config()
    labels = {'class_names': {'0': 'rest'}, 'is_multilabel': False, 'labels': {},
              'splits': {'train': ['train.mp4'], 'val': ['val.mp4']}}
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps(labels))
    cfg.update(run_name='test', device='cpu', max_batches=legacy_limit)
    if train_limit is not None:
        cfg['max_train_batches'] = train_limit
    cfg['data']['label_json'] = path
    cfg['training']['epochs'] = 1
    cfg.pop('wandb', None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, 'check_environment', lambda **kw: None)
    monkeypatch.setattr(train, 'validate_labels_json', lambda *a: None)
    monkeypatch.setattr(train, 'build_datasets_and_loaders', lambda *a: ({'train': object()}, {'train': object(), 'val': object()}))
    monkeypatch.setattr(train, 'build_model', lambda *a: (object(), None))
    monkeypatch.setattr(train, 'build_training_objects', lambda *a: (None, None, None, None))
    init = Mock()
    monkeypatch.setattr(train.wandb, 'init', init)
    monkeypatch.setattr(train.wandb, 'run', None)
    monkeypatch.setattr(train.wandb, 'log', lambda *a: None)
    training = Mock(return_value=([], 0))
    monkeypatch.setattr(train, 'train_one_epoch', training)
    evaluation = Mock(side_effect=RuntimeError('stop after checking evaluation arguments'))
    monkeypatch.setattr(train, 'evaluate', evaluation)
    monkeypatch.setattr(train, 'calculate_multiclass_metrics', lambda *a: {})
    with pytest.raises(RuntimeError, match='stop after'):
        train._run(cfg)
    assert training.call_args.kwargs['max_batches'] == (train_limit or legacy_limit)
    assert evaluation.call_args.kwargs['max_batches'] == legacy_limit
    logged_cfg = init.call_args.kwargs['config']
    assert len(logged_cfg['labels_sha256']) == len(logged_cfg['splits_sha256']) == 64


def test_path_config_checkpoint_is_weights_only_loadable(tmp_path, monkeypatch):
    import pickle
    import torch
    import feral.train as train
    from feral.utils import save_model
    labels_path = tmp_path / 'labels.json'
    original = {'data': {'label_json': labels_path}, 'nested': [tmp_path], 'pair': (1, 2)}
    unsafe = tmp_path / 'path-config.pt'
    torch.save({'cfg': original}, unsafe)
    with pytest.raises(pickle.UnpicklingError):
        torch.load(unsafe, weights_only=True)

    checkpoint = tmp_path / 'normalized.pt'
    def pipeline(cfg):
        save_model(torch.nn.Linear(1, 1), checkpoint, {'cfg': cfg})
        cfg['data']['label_json'] = 'changed inside training'
    monkeypatch.setattr(train, '_run', pipeline)
    train.main(original)
    loaded = torch.load(checkpoint, weights_only=True)
    assert loaded['cfg']['data']['label_json'] == str(labels_path)
    assert loaded['cfg']['nested'] == [str(tmp_path)]
    assert loaded['cfg']['pair'] == (1, 2)
    assert original['data']['label_json'] is labels_path
    assert original['nested'] == [tmp_path]
