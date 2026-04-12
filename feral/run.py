from feral.train import main as train_main
import argparse
import importlib.resources
import yaml
import os
import wandb
from urllib.parse import urlparse

from feral.utils import get_random_run_name

_DEFAULT_CONFIG = importlib.resources.files("feral").joinpath("default_config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Run the FERAL training pipeline.")
    parser.add_argument('video_folder', help="Path to the folder containing training videos.")
    parser.add_argument('label_json_path', help="Path to the label JSON file.")
    parser.add_argument(
        '--checkpoint',
        '-c',
        default=None,
        help="Optional path to a checkpoint to load into cfg['starting_checkpoint']."
    )
    parser.add_argument(
        '--part_subsample',
        type=float,
        default=None,
        help="Optional fraction (0-1) that reduces the number of samples in the train dataset. E.g. 0.5 keeps 50% of training samples."
    )
    parser.add_argument(
        '--subsample_keep_rare_threshold',
        type=float,
        default=None,
        help="When subsampling, keep all chunks with rare behaviors (below this frequency threshold) and sample the rest randomly. E.g. 0.01 keeps behaviors appearing in <1%% of frames."
    )
    args = parser.parse_args()

    prefix_path = args.video_folder
    label_json_path = args.label_json_path
    checkpoint_path = args.checkpoint
    part_subsample = args.part_subsample

    assert os.path.isdir(prefix_path), f"Video folder is not a directory: {prefix_path}"
    assert os.path.isfile(label_json_path), f"Label JSON path is not a file: {label_json_path}"
    if checkpoint_path is not None:
        assert os.path.isfile(checkpoint_path), f"Checkpoint path is not a file: {checkpoint_path}"

    with importlib.resources.as_file(_DEFAULT_CONFIG) as cfg_path:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

    cfg['data']['prefix'] = prefix_path
    cfg['data']['label_json'] = label_json_path
    cfg['run_name'] = get_random_run_name()
    if checkpoint_path is not None:
        cfg['starting_checkpoint'] = checkpoint_path
    if part_subsample is not None:
        if not (0.0 <= part_subsample <= 1.0):
            raise ValueError(f"--part_subsample must be between 0 and 1, got {part_subsample}")
        cfg['data']['part_sample'] = part_subsample
    subsample_keep_rare_threshold = args.subsample_keep_rare_threshold
    if subsample_keep_rare_threshold is not None:
        if part_subsample is None:
            raise ValueError("--subsample_keep_rare_threshold requires --part_subsample to be set")
        if not (0.0 <= subsample_keep_rare_threshold <= 1.0):
            raise ValueError(f"--subsample_keep_rare_threshold must be between 0 and 1, got {subsample_keep_rare_threshold}")
    cfg['data']['subsample_keep_rare_threshold'] = subsample_keep_rare_threshold

    SHARED_WANDB_KEY = "dde17687b4b84ba8171dfede64d865243be41a0e"
    SHARED_WANDB_ENTITY = "sposiboh"
    SHARED_WANDB_PROJECT = "feral_public"

    res = input(
        '\nWeights & Biases logging options:\n'
        '  open     - log to a shared community W&B account (no setup, public)\n'
        '  personal - log to your own W&B project\n'
        '  skip     - no W&B, metrics printed to the command line only\n'
        'Type "open", "personal", or "skip": '
    ).strip().lower()

    if res == "open":
        print("Using shared account")
        wandb.login(key=SHARED_WANDB_KEY)
        cfg['wandb'] = {'entity': SHARED_WANDB_ENTITY, 'project': SHARED_WANDB_PROJECT}
    elif res == "personal":
        key = input('Paste your wandb api_key: ').strip()
        wandb.login(key=key)
        link = input("paste link to the project where you want to log your runs: ")
        link = urlparse(link)
        assert link.netloc == 'wandb.ai', f"should be link to wandb.ai, got {link.netloc}"
        parts = [p for p in link.path.split('/') if p]
        assert len(parts) >= 2, f"Expected wandb.ai/<entity>/<project> URL, got: {link.path}"
        entity = parts[0]
        project = parts[1]
        cfg['wandb'] = {'entity': entity, 'project': project}
        print(f"Entity: {entity} project: {project}")
    elif res == "skip":
        print("Skipping W&B; metrics will be printed to stdout only.")
        cfg.pop('wandb', None)
    else:
        raise SystemExit(f'Should be "open", "personal", or "skip". Got {res!r}')

    train_main(cfg)


if __name__ == '__main__':
    main()
