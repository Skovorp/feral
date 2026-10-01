"""The ROCm LayerNorm replacement must be numerically identical to F.layer_norm, and inert off ROCm."""
import importlib

import pytest
import torch
import torch.nn.functional as F

from feral import rocm_compat


@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm_matches_reference_fp32(affine):
    torch.manual_seed(0)
    d = 64
    x = torch.randn(3, 17, d, requires_grad=True)
    w = torch.randn(d, requires_grad=True) if affine else None
    b = torch.randn(d, requires_grad=True) if affine else None
    leaves = [t for t in (x, w, b) if t is not None]

    ref = F.layer_norm(x, (d,), w, b, 1e-6)
    ref_grads = torch.autograd.grad(ref.square().sum(), leaves)
    out = rocm_compat._layer_norm(x, (d,), w, b, 1e-6)
    out_grads = torch.autograd.grad(out.square().sum(), leaves)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    for g, r in zip(out_grads, ref_grads):
        torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-4)


def test_layer_norm_bf16_no_less_accurate_than_builtin():
    """In bf16 the replacement computes in fp32, so vs an fp32 ground truth it must be at
    least as accurate as the built-in kernel (it is usually more accurate)."""
    torch.manual_seed(0)
    d = 64
    x32, w32, b32 = torch.randn(3, 17, d), torch.randn(d), torch.randn(d)
    truth = F.layer_norm(x32, (d,), w32, b32, 1e-6)
    x, w, b = (t.bfloat16() for t in (x32, w32, b32))
    err_builtin = (F.layer_norm(x, (d,), w, b, 1e-6).float() - truth).abs().max()
    out = rocm_compat._layer_norm(x, (d,), w, b, 1e-6)
    assert out.dtype == torch.bfloat16
    assert (out.float() - truth).abs().max() <= err_builtin * 1.01 + 1e-6


def test_apply_is_noop_without_hip(monkeypatch):
    mod = importlib.reload(rocm_compat)
    original = F.layer_norm
    monkeypatch.setattr(torch.version, "hip", None)
    mod.apply()
    assert F.layer_norm is original


def test_apply_patches_on_hip_and_respects_opt_out(monkeypatch):
    monkeypatch.setattr(F, "layer_norm", F.layer_norm)  # restored after the test
    monkeypatch.setattr(torch.version, "hip", "7.2.0")

    monkeypatch.setenv("FERAL_ROCM_LN_FIX", "0")
    mod = importlib.reload(rocm_compat)
    original = F.layer_norm
    mod.apply()
    assert F.layer_norm is original

    monkeypatch.delenv("FERAL_ROCM_LN_FIX")
    mod = importlib.reload(rocm_compat)
    mod.apply()
    assert F.layer_norm is mod._layer_norm
    assert torch.nn.LayerNorm(8)(torch.randn(2, 8)).shape == (2, 8)
