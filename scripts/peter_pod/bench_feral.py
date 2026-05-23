"""Benchmark feral FeralModel: peak memory + fwd/bwd ms across
backbone × chunk_length × batch_size.

Training mode (full fine-tune — worst case for memory).
bf16 autocast, 256x256 input, regression head with 6 targets (tremor).
Skips configs that OOM and continues.
"""
import os, sys, time, gc, traceback
sys.path.insert(0, "/root/feral")
import torch
import numpy as np
from feral.model import FeralModel

torch.backends.cudnn.benchmark = True

BACKBONES = [
    ("vjepa2_1_vitb_384", "ViT-B"),
    ("vjepa2_1_vitl_384", "ViT-L"),
]
TIMES    = [16, 32, 64]
BSIZES   = [1, 2, 4, 8]
H = W   = 256
N_TARGETS = 6
WARMUP    = 2
TIMED     = 4

device = "cuda"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

results = []
for backbone, label in BACKBONES:
    for T in TIMES:
        for B in BSIZES:
            for k in (k for k in list(globals())):  # nothing — keep state clean
                pass
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            tag = f"{label} T={T:>2} B={B}"
            try:
                model = FeralModel(
                    backbone=backbone, num_classes=1, predict_per_item=T,
                    fc_drop_rate=0.5, freeze_encoder_layers=0,
                    pretrained=False,  # skip checkpoint download; weights don't matter for bench
                    task="regression", num_targets=N_TARGETS,
                ).to(device).train()
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                params_m = sum(p.numel() for p in model.parameters()) / 1e6
                target = torch.randn(B, N_TARGETS, device=device, dtype=torch.float32)
                fwd_times, bwd_times = [], []
                for i in range(WARMUP + TIMED):
                    x = torch.randn(B, T, 3, H, W, device=device, dtype=torch.float32)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(x)
                        loss = (out.float() - target).pow(2).mean()
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    opt.step()
                    if i >= WARMUP:
                        fwd_times.append((t1 - t0) * 1000)
                        bwd_times.append((t2 - t1) * 1000)
                peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
                fwd_ms = float(np.median(fwd_times))
                bwd_ms = float(np.median(bwd_times))
                results.append((label, T, B, params_m, peak_gib, fwd_ms, bwd_ms, "ok"))
                print(f"{tag:<22}  params={params_m:>6.1f}M  peak={peak_gib:>5.2f}GiB  "
                      f"fwd={fwd_ms:>6.1f}ms  bwd={bwd_ms:>6.1f}ms")
            except torch.cuda.OutOfMemoryError:
                results.append((label, T, B, 0, 0, 0, 0, "OOM"))
                print(f"{tag:<22}  OOM")
            except Exception as e:
                results.append((label, T, B, 0, 0, 0, 0, f"err:{e.__class__.__name__}"))
                print(f"{tag:<22}  ERR: {e.__class__.__name__}: {e}")
                traceback.print_exc()
            finally:
                try: del model, opt, x, out, loss
                except Exception: pass
                gc.collect(); torch.cuda.empty_cache()

print()
print(f"{'model':<6} {'T':>3} {'B':>3} {'params(M)':>10} {'peak(GiB)':>10} {'fwd(ms)':>9} {'bwd(ms)':>9}  status")
print("-" * 70)
for r in results:
    label, T, B, p, mem, fwd, bwd, st = r
    if st == "ok":
        print(f"{label:<6} {T:>3} {B:>3} {p:>10.1f} {mem:>10.2f} {fwd:>9.1f} {bwd:>9.1f}  {st}")
    else:
        print(f"{label:<6} {T:>3} {B:>3} {'':>10} {'':>10} {'':>9} {'':>9}  {st}")
