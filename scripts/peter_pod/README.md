# scripts/peter_pod — pod-local helpers from Peter's RunPod 5090 box

Salvaged from `/root/*.py` and `/root/*.sh` on `sound_lime_vole` (RunPod ID `1dbejphavxrq10`) on 2026-05-23 before the pod was killed. These were never on github; preserving them on the `reg` branch for future reference.

These scripts assume specific pod paths (`/root/feral/`, `/root/data/`, `/root/labels/`, `/root/tulip_labels/`, etc.). They are documentation of what was run, not portable tooling — adapt paths for any reuse.

## Categories

**Gait / chair / tremor precrop pipeline** (Sapiens-pose-driven smooth-pan crops):
- `precrop_dims.py` — figure out smallest containing square per video
- `precrop_dryrun.py` — without actually re-encoding
- `precrop_large_bbox.py` — wide-pad variant
- `precrop_smooth.py` — production smooth-pan precrop (sigma=8 frames, 1024² output)
- `gait_aug.py` — per-chunk lerp between small (legs) and large (whole patient) bboxes for training-time aug

**Pose / keypoint analysis**:
- `analyze_kpts.py`, `analyze_kpts2.py` — Sapiens-pose parquet inspectors
- `analyze_gait_patient.py` — per-patient gait QC
- `propagate_patient.py` — propagate patient bbox identity across frames
- `render_middle_frames.py` — middle-frame snapshot helper
- `draw_overlay.py`, `draw_overlay_patient.py` — render prediction overlays

**Tremor (TULIP) sweep tooling**:
- `make_tremor_variants.py` — generate per-camera tremor label JSONs
- `make_variants.py` — broader variant generator
- `val_tremor_sweep.py` — eval sweep across tremor checkpoints

**Labels and conversion**:
- `convert_to_feral_labels.py` — unified_gait_labels.json → feral_gait_labels.json (z-scored)
- `inspect_labels.py` — sanity-check a labels JSON
- `large_size_dist.py` — distribution of bbox sizes
- `selections_labeled.json` would be referenced but is data; not bundled.

**Queue runners** (pre-`scripts/regression_with_negs/run_queue.sh`):
- `queue_runs.sh`, `queue_tremor.sh`, `queue_tremor2-7.sh`, `run_tremor_after_queue.sh` — successive tremor queueing scripts. Superseded by `scripts/regression_with_negs/run_queue.sh`.
- `pull_all.sh` — pulled multiple datasets at once

**Misc smoke / bench**:
- `bench_feral.py`, `bench_then_val.sh` — timing / throughput
- `sanity_chunks.py` — chunk-id sanity checks
- `check_vitb_layers.py` — ViT-B layer freeze inspection
- `smoke_feral.py`, `smoke_unified.py` — quick smoke runs
