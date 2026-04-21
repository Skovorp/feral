"""Run inference on all videos in a folder using a saved checkpoint."""
import importlib.resources
import json
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

_DEFAULT_CONFIG = importlib.resources.files("feral").joinpath("default_config.yaml")

from feral.dataset import (
    ClsDataset,
    collate_fn_inference,
)
from feral.loops import run_inference
from feral.metrics import save_inference_results
from feral.modeling import load_model_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def find_videos(folder):
    all_files = [fn for fn in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, fn))]
    videos = [fn for fn in all_files if os.path.splitext(fn)[1].lower() in VIDEO_EXTENSIONS]
    logger.info("Found %d video files out of %d files in %s", len(videos), len(all_files), folder)
    if not videos:
        raise FileNotFoundError(f"No video files found in {folder}")
    if len(videos) < len(all_files):
        skipped = [fn for fn in all_files if fn not in videos]
        logger.warning("WARNING: %d FILES IN THE FOLDER ARE NOT VIDEOS AND WILL BE SKIPPED: %s",
                        len(skipped), skipped)
    return videos


def build_inference_labels_json(video_filenames, class_names, is_multilabel):
    """Build a minimal labels_json-like dict just for inference (no actual labels needed)."""
    return {
        'class_names': class_names,
        'is_multilabel': is_multilabel,
        'labels': {},
        'splits': {
            'inference': video_filenames,
        },
    }


def _load_default_cfg():
    with importlib.resources.as_file(_DEFAULT_CONFIG) as cfg_path:
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)


def run_inference_folder(checkpoint_path, video_folder, output=None,
                         batch_size=8, num_workers=4, compile=False):
    # Peek at the checkpoint to grab the training cfg (saved since v0.2.1).
    # Falling back to default_config only covers legacy checkpoints where the
    # cfg wasn't persisted — the data/model params in default_config may not
    # match how the model was actually trained.
    raw = torch.load(checkpoint_path, map_location="cpu")
    stored_cfg = raw.get('cfg') if isinstance(raw, dict) else None
    if stored_cfg is not None:
        cfg = stored_cfg
        logger.info("Using training cfg embedded in checkpoint")
    else:
        logger.warning(
            "Checkpoint has no embedded training cfg (legacy format). Falling "
            "back to default_config.yaml — model/data params may not match how "
            "this checkpoint was trained."
        )
        cfg = _load_default_cfg()

    cfg['training']['compile'] = compile
    device = 'cuda'
    video_filenames = find_videos(video_folder)

    # Load model and extract metadata
    model, metadata = load_model_from_checkpoint(cfg, device, checkpoint_path)
    if metadata is None:
        raise SystemExit(
            "This checkpoint is in the legacy format (bare state_dict) and does not "
            "contain class_names or is_multilabel. Please use a checkpoint saved with "
            "the new format, or provide a labels.json and use the full training pipeline."
        )
    class_names = metadata['class_names']
    is_multilabel = metadata['is_multilabel']
    num_classes = len(class_names)

    logger.info("Checkpoint metadata: %d classes, is_multilabel=%s", num_classes, is_multilabel)
    logger.info("Classes: %s", class_names)

    # Build a minimal labels_json for the dataset
    labels_json = build_inference_labels_json(video_filenames, class_names, is_multilabel)

    dataset = ClsDataset(
        partition='inference',
        label_json_dict=labels_json,
        do_aa=False,
        predict_per_item=cfg['predict_per_item'],
        num_classes=num_classes,
        prefix=video_folder,
        resize_to=cfg['data']['resize_to'],
        resize_style=cfg['data'].get('resize_style', 'square'),
        chunk_shift=cfg['data']['chunk_shift'],
        chunk_length=cfg['data']['chunk_length'],
        chunk_step=cfg['data']['chunk_step'],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_inference,
    )

    logger.info("Running inference on %d chunks...", len(dataset))
    answers = run_inference(model, loader, is_multilabel=is_multilabel, device=device)

    if output is None:
        folder_name = os.path.basename(os.path.normpath(video_folder))
        output = f"inference_{folder_name}.json"

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    save_inference_results(answers, [], video_folder, labels_json, output)
    logger.info("Results saved to %s", output)


if __name__ == '__main__':
    from feral.cli import main
    main()
