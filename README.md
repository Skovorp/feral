
[![Documentation](https://img.shields.io/badge/Documentation-getferal.ai-lightgrey)](https://getferal.ai)
[![Colab](https://img.shields.io/badge/Run%20in-Colab-yellow.svg)](https://colab.research.google.com/drive/1bJkHewHpEU7iJT3fIHd0YNrQJ9hkSEdu?usp=sharing)
[![Website](https://img.shields.io/badge/Project%20Website-getferal.ai-blue)](https://getferal.ai)


# FERAL: Feature Extraction for Recognition of Animal Locomotion

Direct video-to-behavior segmentation — no tracking, no pose estimation.


FERAL (Feature Extraction for Recognition of Animal Locomotion) is an open-source video-understanding toolkit that automatically segments animal behavior directly from raw video, without pose tracking or object detection.

FERAL leverages a foundation video model (V-JEPA2) fine-tuned with an attention-based pooling head to produce frame-level behavioral labels and interpretable ethograms across species, experimental setups, and recording modalities.


## Overview

FERAL (Feature Extraction for Recognition of Animal Locomotion) is an open-source video-understanding toolkit that automatically segments animal behavior directly from raw video, without pose tracking or object detection.

FERAL leverages a foundation video model (V-JEPA2) fine-tuned with an attention-based pooling head to produce frame-level behavioral labels and interpretable ethograms across species, experimental setups, and recording modalities.

The pipeline converts:
Raw videos → Spatiotemporal features → Frame-resolved behavioral categories

FERAL is designed for ease of use and reproducibility:
- Fully self-contained Google Colab notebook (no installation)
- Modular command-line and Python API


## Quick Start (Google Colab)

The easiest way to run FERAL is directly in your browser.

Launch FERAL on [Google Colab](https://colab.research.google.com/drive/1bJkHewHpEU7iJT3fIHd0YNrQJ9hkSEdu?usp=sharing)

Recommended: Colab Pro with an A100 or L4 GPU (free for academics with institutional email)

This notebook provides end-to-end execution:
1. Video re-encoding. annotation conversion and dataset validation  
2. Training on labeled videos  
3. Inference and ethogram visualization  
4. Export of predictions as .json  

No installation, driver setup, or local environment configuration required.

## Manual Installation

If you prefer to run locally or on your own cluster:

```
git clone https://github.com/Skovorp/feral.git
cd feral
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.10  
- PyTorch 2.4 + CUDA 12.4  
- NVIDIA GPU (≥ 16 GB VRAM recommended) 

## Dataset Preparation

FERAL expects:
- A folder of re-encoded videos (.mp4, 256 × 256 px)  
- A single annotation JSON mapping each frame to a behavioral category  

Place both in the same directory.

You can validate dataset structure using our built-in Dataset Validator on [getferal.ai](https://getferal.ai).

## Training and Inference

### Run training
```
python run.py path_to_videos path_to_labels.json
```

### Monitor metrics in W&B
FERAL automatically logs:
- Validation raster plots (val_raster_plot, ema_val_raster_plot)
- Mean Average Precision (mAP) per class
- EMA vs. non-EMA metrics
- Frame-level ethograms

## Outputs

After training, predictions are saved in:

```
answers/_inference_{run_name}_{timestamp}.json
```

Each file contains frame-level predicted behaviors suitable for ethogram plotting or downstream analysis.

## Example Datasets

FERAL has been validated on multiple datasets:

- CalMS21 – mouse social interactions  
- MaBE – multi-species benchmark (mice, beetles, ants, flies)  
- C. elegans – locomotor states (forward/reverse/turn/pause)  
- Ooceraea biroi – self vs. allogrooming and collective raids  

Access details and converters are documented at [getferal.ai/datasets](https://getferal.ai).

## Deployment Options

- Google Colab (recommended) — fastest setup for single-GPU runs  
- Local training — custom datasets, full control  
- RunPod / Cluster deploy — scalable multi-GPU fine-tuning (see website guide) 

Step-by-step deployment guides for each environment are available in the documentation.

## Citation

Please contact us at jrazzauti@rockefeller.edu or pskovordnikov@rockefeller.edu for instructions on how to cite our work.

Contributions welcome — see CONTRIBUTING.md.

## Authors

Peter Skovorodnikov† (Rockefeller University)†
Jacopo Razzauti† (Vosshall Lab, Rockefeller University; Price Family Center for the Social Brain)†


† Equal contribution  
Contact: jrazzauti@rockefeller.edu | pskovorodnikov@rockefeller.edu

## License
This project is released under the MIT License.


