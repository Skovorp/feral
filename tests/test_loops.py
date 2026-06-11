import torch
import torch.nn.functional as F

from feral.loops import _per_element_loss


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
