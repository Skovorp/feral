"""ROCm workarounds, applied only on HIP builds of torch (no-op on CUDA/CPU/MPS).

LayerNorm backward on RDNA4 (gfx1201, e.g. RX 9070 XT / Radeon AI PRO R9700) with
torch 2.12 + ROCm 7.2 never writes the odd-indexed elements of the weight/bias
gradients once the input has >= 128 rows; they hold whatever was in that memory
(0 in a fresh process, stale values later). Even-indexed elements, the input
gradient and the forward are correct. pytorch/pytorch#199265 has the root cause
(ROCm's tiled gamma/beta kernel assumes 64-wide waves; RDNA runs 32-wide).
Training still "runs", but on V-JEPA 2.1 ViT-L with 12 frozen layers every
LayerNorm in the trainable blocks and the pooler gets half its gradient replaced
by garbage.

We swap F.layer_norm for the same math built from elementwise ops in fp32, so
autograd never calls the fused backward kernel. With it, all 160 tensors match a CPU
reference (exact in fp32) for ~15% step-time cost. Set FERAL_ROCM_LN_FIX=0 to disable.
"""
import os

import torch
import torch.nn.functional as F

_applied = False


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    dims = tuple(range(-len(normalized_shape), 0))
    xf = x.float()
    mean = xf.mean(dims, keepdim=True)
    var = (xf - mean).pow(2).mean(dims, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype)


def apply():
    """Patch F.layer_norm on ROCm builds. Idempotent."""
    global _applied
    if _applied or torch.version.hip is None or os.environ.get("FERAL_ROCM_LN_FIX", "1") == "0":
        return
    F.layer_norm = _layer_norm  # nn.LayerNorm.forward looks this up through F at call time
    _applied = True
