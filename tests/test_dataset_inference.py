import numpy as np
import pytest
import torch

from feral import dataset
from feral.metrics import ensemble_predictions


def make_dataset(monkeypatch, partition='inference', n_frames=3, do_aa=True, chunk_step=1, **kwargs):
    monkeypatch.setattr(dataset, 'get_frame_count', lambda path: n_frames)
    monkeypatch.setattr(dataset, 'get_video_dims', lambda path: (8, 8))
    return dataset.ClsDataset(
        partition=partition,
        label_json_dict={'splits': {partition: ['short.mp4']},
                         'labels': {'short.mp4': [0] * (n_frames or 0)}},
        do_aa=do_aa, predict_per_item=4, num_classes=2, prefix='',
        resize_to=8, chunk_shift=4, chunk_length=4, chunk_step=chunk_step, **kwargs)


@pytest.mark.parametrize('n_frames', [1, 3])
def test_short_inference_chunk_produces_real_frame_predictions(monkeypatch, n_frames):
    ds = make_dataset(monkeypatch, n_frames=n_frames)
    monkeypatch.setattr(dataset, 'read_range_video_decord',
                        lambda path, frames, **kwargs: torch.zeros(len(frames), 3, 8, 8, dtype=torch.uint8))
    video, names = ds[0]
    assert video.shape == (4, 3, 8, 8)
    assert [name[1] for name in names] == list(range(n_frames)) + [n_frames - 1] * (4 - n_frames)
    result = ensemble_predictions([(name, [0.2, 0.8]) for name in names],
                                  {'short.mp4': np.zeros((n_frames, 2))})
    np.testing.assert_allclose(result['short.mp4'], [[0.2, 0.8]] * n_frames)


@pytest.mark.parametrize('partition', ['train', 'val', 'test'])
def test_short_labeled_videos_do_not_change_sampling_protocol(monkeypatch, partition):
    with pytest.raises(ValueError, match='No full video chunks'):
        make_dataset(monkeypatch, partition=partition)


@pytest.mark.parametrize('n_frames', [None, 0])
def test_unreadable_video_is_explicit_error(monkeypatch, n_frames):
    with pytest.raises(ValueError, match='Cannot read frames'):
        make_dataset(monkeypatch, n_frames=n_frames)


def test_inference_decode_failure_does_not_substitute_another_chunk(monkeypatch):
    ds = make_dataset(monkeypatch)
    def fail(index):
        raise ValueError('decode failed')
    monkeypatch.setattr(ds, 'get_item_simple', fail)
    monkeypatch.setattr(np.random, 'randint', lambda *args: pytest.fail('must not substitute a chunk'))
    with pytest.raises(RuntimeError, match='Failed to decode inference chunk'):
        ds[0]


@pytest.mark.parametrize('partition', ['train', 'val', 'test', 'inference'])
@pytest.mark.parametrize('do_aa', [False, True])
def test_augmentation_is_training_only(monkeypatch, partition, do_aa):
    ds = make_dataset(monkeypatch, partition=partition, n_frames=4, do_aa=do_aa)
    assert (ds.aug is not None) == (partition == 'train' and do_aa)


def test_inference_scales_uint8_then_normalizes(monkeypatch):
    ds = make_dataset(monkeypatch)
    pixels = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 3, 1, 1).expand(4, 3, 8, 8)
    monkeypatch.setattr(ds, 'get_video', lambda index: (pixels, []))
    actual, _ = ds[0]
    expected = (torch.tensor([0., 128. / 255., 1.]) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    torch.testing.assert_close(actual, expected.reshape(1, 3, 1, 1).expand_as(actual))


def test_short_inference_padding_with_temporal_stride(monkeypatch):
    ds = make_dataset(monkeypatch, n_frames=6, chunk_step=2)
    assert ds.samples == [('short.mp4', [0, 2, 4, 5])]
    ans = [(('short.mp4', frame, i), [frame / 5., 1. - frame / 5.])
           for i, frame in enumerate(ds.samples[0][1])]
    result = ensemble_predictions(ans, {'short.mp4': np.zeros((6, 2))})
    np.testing.assert_allclose(result['short.mp4'][:, 0], np.arange(6) / 5.)
