"""Host memory a finished phase no longer needs must be given back before the next phase starts.

Background (issue #24): loaders used persistent workers and lived for the whole run, so by the
post-training inference pass the train, val and test workers were all still alive, each holding its
own memory plus a copy-on-write snapshot of whatever the parent held when it was forked (the
previous pass's answers). PyTorch also keeps every pinned batch buffer it ever allocated.
"""
import gc
import json
import multiprocessing
import weakref

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

MiB = 2 ** 20


def _persistent_loader():
    return DataLoader(TensorDataset(torch.zeros(4, 1)), batch_size=2, num_workers=1, persistent_workers=True)


def _stop(loaders):
    """Test cleanup: never leave worker processes behind for the rest of the suite."""
    for loader in loaders:
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()


def _new_live_workers(baseline):
    return [p for p in multiprocessing.active_children() if p not in baseline and p.is_alive()]


def test_release_loaders_stops_idle_workers_and_forgets_the_partition():
    from feral.data import release_loaders
    baseline = set(multiprocessing.active_children())
    loaders = {name: _persistent_loader() for name in ('train', 'test')}
    datasets = {name: loader.dataset for name, loader in loaders.items()}
    created = list(loaders.values())
    try:
        for loader in created:
            for _ in loader:
                pass
        assert len(_new_live_workers(baseline)) == 2  # persistent workers idle after their pass

        release_loaders(loaders, datasets, 'train')

        assert len(_new_live_workers(baseline)) == 1  # only the test loader's worker is left
        assert sorted(loaders) == ['test']
        assert sorted(datasets) == ['test']
    finally:
        _stop(created)


def test_release_loaders_ignores_partitions_that_were_never_built():
    from feral.data import release_loaders
    loaders, datasets = {}, {}
    release_loaders(loaders, datasets, 'train', 'val')  # e.g. a run with no val split
    assert loaders == {} and datasets == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")
def test_release_loaders_returns_cached_pinned_memory():
    from feral.data import release_loaders

    def held():
        return torch.cuda.host_memory_stats()["allocated_bytes.current"]

    torch.zeros(1, device='cuda')  # as in a real run, where the model is already on the GPU
    batch = torch.zeros(48 * MiB, dtype=torch.uint8).pin_memory()
    del batch  # PyTorch keeps the block cached; torch.cuda.empty_cache() would not return it
    before = held()

    release_loaders({}, {})

    assert before - held() >= 48 * MiB


class _Answers(list):
    """A list that can be weakly referenced, standing in for one pass's predictions."""


def test_each_phase_starts_without_the_previous_phases_workers_and_answers(tmp_path, monkeypatch):
    import feral.train as train
    from feral import load_default_config

    cfg = load_default_config()
    labels = {'class_names': {'0': 'rest'}, 'is_multilabel': False, 'labels': {},
              'splits': {'train': ['a.mp4'], 'val': ['b.mp4'], 'test': ['c.mp4'], 'inference': ['d.mp4']}}
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps(labels))
    cfg.update(run_name='test', device='cpu')
    cfg['data']['label_json'] = path
    cfg['data']['prefix'] = str(tmp_path)
    cfg['training']['epochs'] = 1
    cfg.pop('wandb', None)
    monkeypatch.chdir(tmp_path)

    baseline = set(multiprocessing.active_children())
    loaders = {name: _persistent_loader() for name in ('train', 'val', 'test', 'inference')}
    datasets = {name: loader.dataset for name, loader in loaders.items()}
    created = list(loaders.values())

    produced = {}  # phase -> weak reference to the answers that phase returned
    seen = {}      # phase -> what was still being held when that phase started

    def snapshot(phase):
        gc.collect()
        seen[phase] = {'workers': len(_new_live_workers(baseline)),
                       'answers': sorted(name for name, ref in produced.items() if ref() is not None)}

    def run_pass(phase, loader):
        snapshot(phase)
        for _ in loader:  # starts this loader's worker, as a real pass does
            pass
        answers = _Answers()
        produced[phase] = weakref.ref(answers)
        return answers

    eval_phases = iter(['val', 'test'])
    monkeypatch.setattr(train, 'check_environment', lambda **kw: None)
    monkeypatch.setattr(train, 'validate_labels_json', lambda *a: None)
    monkeypatch.setattr(train, 'build_datasets_and_loaders', lambda *a: (datasets, loaders))
    monkeypatch.setattr(train, 'build_model', lambda *a: (object(), None))
    monkeypatch.setattr(train, 'build_training_objects', lambda *a: (None, None, None, None))
    monkeypatch.setattr(train, 'load_model_from_checkpoint', lambda *a, **kw: (object(), None))
    monkeypatch.setattr(train, 'pick_and_save_best', lambda *a: (0.5, 'base'))
    monkeypatch.setattr(train.wandb, 'init', lambda **kw: None)
    monkeypatch.setattr(train.wandb, 'run', None)
    monkeypatch.setattr(train.wandb, 'log', lambda *a: None)
    monkeypatch.setattr(train, 'train_one_epoch', lambda model, loader, *a, **kw: (run_pass('train', loader), 0.0))
    monkeypatch.setattr(train, 'evaluate', lambda model, loader, *a, **kw: (run_pass(next(eval_phases), loader), 0.0))
    monkeypatch.setattr(train, 'run_inference', lambda model, loader, **kw: run_pass('inference', loader) and None)

    class Writer:  # stands in for InferenceWriter; close() is where the result file is finished
        def __init__(self, *a, **kw):
            pass

        def close(self):
            snapshot('save')
    monkeypatch.setattr(train, 'InferenceWriter', Writer)
    for name in ('calculate_multiclass_metrics', 'calculate_f1_metrics', 'calculate_optimal_f1_metrics',
                 'ensemble_predictions', 'postprocess_predictions', 'generate_empty_logits'):
        monkeypatch.setattr(train, name, lambda *a, **kw: {})
    monkeypatch.setattr(train, 'calc_frame_level_map', lambda *a: 0.5)
    monkeypatch.setattr(train, '_add_raster_logs', lambda *a: None)

    try:
        train._run(cfg)
    finally:
        _stop(created)

    assert seen == {
        'train': {'workers': 0, 'answers': []},
        # the train loader is reused next epoch, so its worker stays; its answers are already scored
        'val': {'workers': 1, 'answers': []},
        'test': {'workers': 0, 'answers': []},
        'inference': {'workers': 0, 'answers': []},
        # the result file is finished after the inference workers are gone
        'save': {'workers': 0, 'answers': []},
    }
