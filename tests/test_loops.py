import torch
import torch.nn.functional as F

from feral.loops import _global_grad_norm, _per_element_loss


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
