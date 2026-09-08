[![Documentation](https://img.shields.io/badge/Documentation-getferal.ai-lightgrey)](https://getferal.ai)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-yellow.svg)](https://colab.research.google.com/drive/1wPe7MX3IiY3zsFkeTzLzHgynrtj-wXFP)
[![Website](https://img.shields.io/badge/Project%20Website-getferal.ai-blue)](https://getferal.ai)

# FERAL

FERAL is an open-source toolkit for supervised animal behavior segmentation that leverages state-of-the-art video-understanding models.

- No pose estimation
- Install and start training in 3 commands
- Scale a small set of manual labels to terabytes of video

## Quickest Start (Google Colab)

The easiest way to run FERAL is directly in your browser through Google Colab, but you'll need a Colab Pro subscription to access good GPUs. Good news: Colab Pro is free for academics with an institutional email.

[Open in Google Colab](https://colab.research.google.com/drive/1wPe7MX3IiY3zsFkeTzLzHgynrtj-wXFP)

> Use with an A100 or L4 GPU.

## Manual Installation

```
pip install feral
```

That's it. Now you should be able to run everything.

### System Requirements

- Preferably Linux. Windows should work too. We haven't tested on Mac.
- Python 3.10+
- PyTorch 2.5+ (with a compatible CUDA version)
- NVIDIA GPU with Ampere architecture or newer (compute capability 8.0+). 8 GB configurations are benchmarked below; 24+ GB gives more room for batch size. FERAL uses bfloat16 and flash attention, which require newer architectures.

> Historical V-JEPA experiments used PyTorch 2.4, but `pip install feral` requires PyTorch 2.5+ and may upgrade a 2.4 environment. VideoPrism needs `torch.nn.attention.flex_attention`, introduced in 2.5.

> Older GPUs like V100 (Volta) and T4 (Turing) — including free Google Colab T4 instances — will **not** work. Supported GPUs include: A100, H100, A10, L4, L40, RTX 3000/4000/5000 series, and newer.

For 8 GB GPUs, start with:

```sh
feral train path/to/videos path/to/labels.json --mode lite --gradient-checkpointing --no-wandb
```

[Paper v2, Table 3](https://www.biorxiv.org/content/10.1101/2025.11.16.688666v2.full.pdf) benchmarks an RTX 3070 (8 GB): lite with gradient checkpointing trains at 1.82 chunks/s and runs inference at 7 chunks/s, with an estimated 24.2 hours for full CalMS21 training. These are measured recipe-specific rates, not guarantees for every dataset; Table 8 compares preset accuracy. Training stays on your own hardware with `--no-wandb` (initial model-weight downloads still require network access).

The Docker image uses PyTorch 2.5.1 / CUDA 12.4. The V-JEPA experiments also used PyTorch 2.4 / CUDA 12.4; install 2.5+ for the full backbone selection. PyTorch is intentionally not pinned so existing GPU environments can keep a compatible build.

### Windows Notes

- `decord` and `triton-windows` are installed automatically. Windows GPU training has not been revalidated for this release.
- Triton must match your PyTorch minor version: 2.5 → 3.1, 2.6 → 3.2, 2.7 → 3.3, 2.8 → 3.4, 2.9 → 3.5, 2.10 → 3.6. For example, with PyTorch 2.10 run `pip install "triton-windows>=3.6,<3.7"`. Unlike the other dependencies, it is not pinned because it follows the user-installed PyTorch build. See the [upstream compatibility table](https://github.com/triton-lang/triton-windows).
- **PyTorch 2.8-2.9 has a known bug on Windows** where `torch.compile` crashes with `OverflowError: Python int too large to convert to C long` ([pytorch#162430](https://github.com/pytorch/pytorch/issues/162430)). Use PyTorch 2.7 or 2.10+ to avoid this. See [issue #11](https://github.com/Skovorp/feral/issues/11) for details.

We've used RunPod extensively to run experiments. It's very easy to set up.

### Troubleshooting

- **V-JEPA 2.1 backbone download fails with `URLError: [Errno 111] Connection refused`.** Upstream's `facebookresearch/vjepa2` `torch.hub` repo currently ships a leftover test URL (`VJEPA_BASE_URL = "http://localhost:8300"`). Point it back at the public CDN in the cached copy:
  ```
  sed -i 's|http://localhost:8300|https://dl.fbaipublicfiles.com/vjepa2|' \
    ~/.cache/torch/hub/facebookresearch_vjepa2_main/src/hub/backbones.py
  ```
  This only affects the `vjepa2_1_*` (V-JEPA 2.1) backbones.

## Running FERAL

### 1. Re-encode Videos

FERAL needs videos in a format that supports random frame access, so we recommend re-encoding your videos with the provided tool. It will install FFmpeg if it doesn't find it on your system.

```
feral reencode path/to/videos path/for/reencoded/videos
```

### 2. Start Training

```
feral train path/to/videos path/to/labels
```

For batch/HPC jobs, add `--no-wandb` to disable external logging and prompts, or `--public-wandb` to explicitly publish logs to the community project. Run `feral train --help` for preset, batch-size, and gradient-checkpointing options.

Without either flag, this will prompt you about logging options — you can select what fits you. You'll be able to see per-epoch metrics either way. If you select W&B, you'll also get automatically generated ethograms, per-step loss, and system utilization info.

Results will be saved to `answers/_inference_{run_name}_{timestamp}.json`.

### 3. Training from a Config

If you don't want to run this in an interactive terminal, you can start training with:

```
feral train-config path/to/config
```

This requires a config in the same format as the packaged `feral/default_config.yaml`.

### 4. Inference

You can run inference through the train command by removing the train partition from your config. But you can also run inference standalone without a config:

```
feral infer path/to/checkpoint path/to/videos
```

## Example Datasets

FERAL has been validated on multiple datasets:

- **CalMS21** - mouse social interactions
- **MaBE** - multi-species benchmark (mice, beetles, ants, flies)
- **C. elegans** - locomotor states (forward/reverse/turn/pause)
- **Ooceraea biroi** - self vs. allogrooming and collective raids

Access details and converters are documented at [getferal.ai](https://getferal.ai).

## Citation

Please cite the [FERAL preprint](https://doi.org/10.1101/2025.11.16.688666); the [revised manuscript (v2)](https://www.biorxiv.org/content/10.1101/2025.11.16.688666v2) includes hardware and preset benchmarks.

## Authors

Peter Skovorodnikov<sup>&dagger;</sup> (Rockefeller University)
Jacopo Razzauti<sup>&dagger;</sup> (Vosshall Lab, Rockefeller University; Price Family Center for the Social Brain)

<sup>&dagger;</sup> Equal contribution
Contact: jacopo.razza@gmail.com | peter.skovorodnikov@gmail.com

---

*FERAL = Feature Extraction for Recognition of Animal Locomotion*
