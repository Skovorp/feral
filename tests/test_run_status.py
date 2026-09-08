"""A failed run must retain inspectable status without overwriting prior work."""
import json
import pytest
from feral.run_status import training_status


def config(tmp_path):
    labels = tmp_path / 'labels.json'
    labels.write_text('{}')
    return {'run_name': 'example', 'output_dir': str(tmp_path / 'out'),
            'data': {'label_json': str(labels)}, 'wandb': {'key': 'secret'}}


def test_status_lifecycle_and_outputs(tmp_path):
    cfg = config(tmp_path)
    root = tmp_path / 'out'
    with training_status(cfg):
        active = json.loads((root / 'run.json').read_text())
        assert active['status'] == 'running'
        assert 'wandb' not in active['recipe']
        (root / 'checkpoints').mkdir()
        (root / 'checkpoints' / 'best.pt').write_text('checkpoint')
    done = json.loads((root / 'run.json').read_text())
    assert done['status'] == 'completed'
    assert done['checkpoints'] == [str(root / 'checkpoints' / 'best.pt')]
    assert len(done['labels_sha256']) == 64
    with pytest.raises(FileExistsError):
        with training_status(cfg):
            pytest.fail('must not overwrite a prior run')
    assert json.loads((root / 'run.json').read_text()) == done


def test_failed_run_keeps_error(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(RuntimeError):
        with training_status(cfg):
            raise RuntimeError('out of memory')
    report = json.loads((tmp_path / 'out' / 'run.json').read_text())
    assert report['status'] == 'failed'
    assert report['error'] == {'type': 'RuntimeError', 'message': 'out of memory'}


def test_default_has_no_status_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with training_status({}):
        pass
    assert list(tmp_path.iterdir()) == []


def test_missing_labels_still_writes_failed_status(tmp_path):
    cfg = config(tmp_path)
    cfg['data']['label_json'] = str(tmp_path / 'missing.json')
    with pytest.raises(FileNotFoundError):
        with training_status(cfg):
            pytest.fail('missing labels must fail before training')
    report = json.loads((tmp_path / 'out' / 'run.json').read_text())
    assert report['status'] == 'failed'


def test_existing_artifacts_not_overwritten(tmp_path):
    cfg = config(tmp_path)
    root = tmp_path / 'out'
    root.mkdir()
    artifact = root / 'old-results.json'
    artifact.write_text('keep')
    with pytest.raises(FileExistsError):
        with training_status(cfg):
            pytest.fail('nonempty directory must be rejected')
    assert artifact.read_text() == 'keep'
    assert not (root / 'run.json').exists()


def test_path_config_and_unserializable_config(tmp_path):
    from pathlib import Path
    cfg = config(tmp_path)
    cfg['output_dir'] = Path(cfg['output_dir'])
    cfg['data']['label_json'] = Path(cfg['data']['label_json'])
    with training_status(cfg):
        pass
    report = json.loads((cfg['output_dir'] / 'run.json').read_text())
    assert report['recipe']['data']['label_json'] == str(cfg['data']['label_json'])
    cfg['output_dir'] = tmp_path / 'invalid'
    cfg['unsupported'] = object()
    with pytest.raises(TypeError):
        with training_status(cfg):
            pass
    assert not (cfg['output_dir'] / 'run.json').exists()


def test_fingerprints_track_labels_and_split_order():
    from feral.run_status import label_fingerprints
    first = {'splits': {'train': ['a', 'b'], 'test': ['c']}}
    reformatted = {'splits': {'test': ['c'], 'train': ['a', 'b']}}
    a = label_fingerprints(json.dumps(first).encode(), first)
    b = label_fingerprints(json.dumps(reformatted).encode(), reformatted)
    assert a['labels_sha256'] != b['labels_sha256']
    assert a['splits_sha256'] == b['splits_sha256']
    first['splits']['train'].reverse()
    assert label_fingerprints(b'changed', first)['splits_sha256'] != a['splits_sha256']
