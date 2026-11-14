import torch
import torch.nn.functional as F

from feral.loops import _global_grad_norm, _per_element_loss, evaluate, run_inference, train_one_epoch


def _param(grad_vals):
    p = torch.nn.Parameter(torch.zeros(len(grad_vals)))
    p.grad = torch.tensor(grad_vals)
    return p


class TestGlobalGradNorm:
    def test_matches_clip_grad_norm_value(self):
        # Read-only norm must equal clip_grad_norm_'s returned pre-clip norm.
        ours = _global_grad_norm([_param([3.0, 4.0])])
        ref = torch.nn.utils.clip_grad_norm_([_param([3.0, 4.0])], max_norm=float('inf'))
        assert ours.item() == ref.item() == 5.0

    def test_does_not_mutate_finite_grads(self):
        p = _param([3.0, 4.0])
        _global_grad_norm([p])
        assert p.grad.tolist() == [3.0, 4.0]

    def test_does_not_corrupt_on_overflow(self):
        # The whole point: an inf gradient must be left untouched (NOT turned to
        # NaN the way clip_grad_norm_(max_norm=inf) would).
        p = _param([3.0, float('inf')])
        norm = _global_grad_norm([p])
        assert p.grad[0].item() == 3.0
        assert torch.isinf(p.grad[1])
        assert torch.isinf(norm)

    def test_none_when_no_grads(self):
        p = torch.nn.Parameter(torch.zeros(2))  # grad is None
        assert _global_grad_norm([p]) is None


class TestPerElementLoss:
    def test_multilabel_shape_and_values(self):
        # (N, C) logits/targets -> one loss per sample (mean over classes).
        output = torch.randn(5, 3)
        target = (torch.rand(5, 3) > 0.5).float()
        out = _per_element_loss(output, target, is_multilabel=True)
        assert out.shape == (5,)
        expected = F.binary_cross_entropy_with_logits(output, target, reduction='none').mean(dim=-1)
        torch.testing.assert_close(out, expected)

    def test_singlelabel_matches_cross_entropy(self):
        # Hard one-hot targets -> should equal F.cross_entropy per-sample.
        output = torch.randn(4, 6)
        idx = torch.tensor([0, 2, 5, 1])
        target = F.one_hot(idx, num_classes=6).float()
        out = _per_element_loss(output, target, is_multilabel=False)
        expected = F.cross_entropy(output, idx, reduction='none')
        assert out.shape == (4,)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_singlelabel_supports_soft_targets(self):
        # Mixup produces soft targets; loss should still be finite and per-sample.
        output = torch.randn(3, 4)
        target = torch.softmax(torch.randn(3, 4), dim=-1)
        out = _per_element_loss(output, target, is_multilabel=False)
        assert out.shape == (3,)
        assert torch.isfinite(out).all()


class _Probe(torch.nn.Module):
    """Records what the loop feeds the model; returns zero logits of the real output shape."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes
        self.seen = []

    def forward(self, data):
        self.seen.append(data.detach().clone())
        return self.weight.expand(data.shape[0] * data.shape[1], self.num_classes) + 0


class _Recorder:
    """Stands in for an InferenceWriter: keeps what run_inference hands over."""

    def __init__(self):
        self.batches = []

    def add(self, names, probs):
        self.batches.append((names, probs))


def test_run_inference_streams_each_batch_to_the_writer_and_keeps_nothing():
    # Two batches of 2 chunks x 4 frames. Zero logits over 2 classes -> probability 0.5 each.
    def batch(video):
        data = torch.zeros((2, 4, 3, 2, 2), dtype=torch.uint8)
        names = tuple([(video, 10 * chunk + frame, frame) for frame in range(4)] for chunk in range(2))
        return data, names

    writer = _Recorder()
    result = run_inference(_Probe(), [batch('a.mp4'), batch('b.mp4')], is_multilabel=False, device='cpu',
                           prediction_writer=writer)

    assert result is None  # predictions are not collected, so memory does not grow with the pass
    assert [names for names, _ in writer.batches] == [
        [(video, 10 * chunk + frame, frame) for chunk in range(2) for frame in range(4)]
        for video in ('a.mp4', 'b.mp4')]
    for _, probs in writer.batches:
        assert probs.shape == (8, 2) and probs.dtype == 'float32'
        assert (probs == 0.5).all()


class TestLoopsNormalizeFrames:
    """Loaders deliver uint8 frames; every loop must hand the model the normalized float32 batch."""

    # white pixels: (1 - mean) / std per channel
    WHITE = torch.tensor([(1 - .485) / .229, (1 - .456) / .224, (1 - .406) / .225]).reshape(3, 1, 1)

    def _batch(self):
        data = torch.full((2, 4, 3, 2, 2), 255, dtype=torch.uint8)
        target = F.one_hot(torch.zeros(2, 4, dtype=torch.long), num_classes=2).float()
        names = tuple([('video.mp4', frame, frame) for frame in range(4)] for _ in range(2))
        return data, target, names

    def _assert_model_saw_normalized_frames(self, model):
        (seen,) = model.seen
        assert seen.dtype == torch.float32
        torch.testing.assert_close(seen, self.WHITE.expand_as(seen))

    def test_run_inference(self):
        data, _, names = self._batch()
        model = _Probe()
        run_inference(model, [(data, names)], is_multilabel=False, device='cpu', prediction_writer=_Recorder())
        self._assert_model_saw_normalized_frames(model)

    def test_evaluate(self):
        model = _Probe()
        evaluate(model, [self._batch()], num_classes=2, is_multilabel=False, device='cpu')
        self._assert_model_saw_normalized_frames(model)

    def test_train_one_epoch(self):
        data, target, _ = self._batch()
        model = _Probe()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        train_one_epoch(model, [(data, target)], torch.nn.CrossEntropyLoss(), optimizer, scheduler,
                        mixup=None, model_ema=None, num_classes=2, is_multilabel=False,
                        predict_per_item=4, device='cpu')
        self._assert_model_saw_normalized_frames(model)
