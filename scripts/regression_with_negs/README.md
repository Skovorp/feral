# scripts/regression_with_negs

Helpers used by the 2026-05-23 negs-and-randaug experiments. See
`../../configs/regression_with_negs/README.md` for context.

| Script | What it does |
|---|---|
| `build_chair_negs.py` | Symlinks 72 TULIP chair videos + 241 FoG-negs into `/root/data/chair_with_negs/`; writes `chair_labels_with_negs.json` (orig labels + negs at z-scored 0). |
| `build_walking_negs.py` | Same for the unified gait dataset (auto-gait + koa-pd-nm-gait + tulip-gait, precrop variant) + FoG-negs. |
| `build_fingertap_negs.py` | Same for HUBU-FIS + FoG-negs. |
| `run_queue.sh` | Sequential GPU-aware queue runner. Sleeps until `nvidia-smi memory.used < 5 GiB` before launching the next `python -m feral.cli train-config <cfg>`. Skips already-trained runs by checking for `checkpoints/<name>_best_checkpoint.pt`. |
| `scan_deployable.py` | For every `*_best_checkpoint.pt` on disk, find the matching-timestamp answers JSON and compute **per-video corr / r² on the canonical (orig) val** plus the polluted full-val. Use this to pick the right checkpoint to deploy. |
| `recompute_clean.py` | Subset of `scan_deployable.py`: per-experiment per-epoch orig-val corr / r² for hand-picked answer JSONs. Useful when you want to see which epoch of a single run is actually the best on the original task videos. |

## Quick usage

```bash
# 1) Build the augmented labels + symlink farms (one-time)
python scripts/regression_with_negs/build_chair_negs.py
python scripts/regression_with_negs/build_walking_negs.py
python scripts/regression_with_negs/build_fingertap_negs.py

# 2) Launch the queue
bash scripts/regression_with_negs/run_queue.sh

# 3) Pick the best deployable checkpoint per symptom
python scripts/regression_with_negs/scan_deployable.py
```

## Heads-up

- The build scripts add FoG-negs to **both train and val** of each labels
  JSON. **Do not use these labels for apples-to-apples eval** vs prior
  baselines — `scan_deployable.py` re-filters to orig val for honest numbers.
  See `dont-mutate-val-without-permission` skill in the vault.
- Hard-coded R2 paths and `/root/...` paths assume RunPod layout. Adapt for
  other clusters.
