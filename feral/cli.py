import argparse
import importlib.resources
import os
import sys

import yaml

_DEFAULT_CONFIG = importlib.resources.files("feral").joinpath("default_config.yaml")


def _load_default_config():
    """Load and return the packaged default_config.yaml as a dict."""
    with importlib.resources.as_file(_DEFAULT_CONFIG) as cfg_path:
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)


# ── train ────────────────────────────────────────────────────────────────────

def _cmd_train(args):
    """Run the interactive `feral train` command: build cfg from the default config plus CLI args, optionally apply a --mode preset, interactively configure W&B logging (open/personal/skip), then call train.main(cfg)."""
    from urllib.parse import urlparse
    import wandb
    from feral.train import main as train_main
    from feral.utils import get_random_run_name
    from feral.presets import apply_mode, MODE_HELP

    cfg = _load_default_config()
    if args.mode is not None:
        cfg = apply_mode(cfg, args.mode)
        print(f"Using --mode {args.mode}: {MODE_HELP[args.mode]}")
    cfg['data']['prefix'] = args.video_folder
    cfg['data']['label_json'] = args.label_json_path
    cfg['run_name'] = get_random_run_name()

    if args.resolution is not None:
        cfg['data']['resize_to'] = args.resolution
        print(f"Using --resolution {args.resolution} (square input)")
    if args.checkpoint is not None:
        cfg['starting_checkpoint'] = args.checkpoint
    if args.part_subsample is not None:
        cfg['data']['part_sample'] = args.part_subsample
    if args.subsample_keep_rare_threshold is not None:
        cfg['data']['subsample_keep_rare_threshold'] = args.subsample_keep_rare_threshold

    if args.no_wandb:
        print("Skipping W&B (--no-wandb); metrics will be printed to stdout only.")
        cfg.pop('wandb', None)
        train_main(cfg)
        return

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


# ── train-config ─────────────────────────────────────────────────────────────

def _cmd_train_config(args):
    """Run `feral train-config`: load cfg from the given YAML file, log in to W&B if a key is present, then call train.main(cfg)."""
    import wandb
    from feral.train import main as train_main

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    key = cfg.get('wandb', {}).get('key') if cfg.get('wandb') else None
    if key:
        wandb.login(key=key)
    train_main(cfg)


# ── infer ────────────────────────────────────────────────────────────────────

def _cmd_infer(args):
    """Run `feral infer`: dispatch CLI args to run_inference_folder to label every video in a folder from a checkpoint."""
    from feral.inference_folder import run_inference_folder

    run_inference_folder(
        checkpoint_path=args.checkpoint,
        video_folder=args.video_folder,
        output=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compile=getattr(args, 'compile', False),
        mode=args.mode,
        resolution=args.resolution,
    )


# ── reencode ─────────────────────────────────────────────────────────────────

def _cmd_reencode(args):
    """Run `feral reencode`: validate the input dir holds only videos, set up ffmpeg, then re-encode each video into an empty output dir in parallel. Exits non-zero on validation errors or any failed file."""
    from pathlib import Path
    from multiprocessing import Pool
    from feral.reencode_videos import is_video_file, setup_ffmpeg, process_file

    print("FERAL Video Re-encoding Script")
    print("=" * 50)

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    video_paths = []
    for filename in os.listdir(args.input_dir):
        filepath = os.path.join(args.input_dir, filename)
        if os.path.isfile(filepath) and is_video_file(filepath):
            video_paths.append(filepath)
        else:
            print(f"Input directory must only have videos. Found not video: {filepath}")
            sys.exit(1)
    if not video_paths:
        print("No video files found in input directory.")
        sys.exit(1)
    print(f"Found {len(video_paths)} video files to process")

    # Create output directory
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if any(out_dir.iterdir()):
            print(f"Directory '{out_dir}' should be empty")
            sys.exit(1)
    else:
        out_dir.mkdir(parents=True)

    # Setup FFmpeg (download if needed)
    ffmpeg_binary = setup_ffmpeg()
    input_files = [(x, args.output_dir, ffmpeg_binary, args.smallest_side) for x in video_paths]

    print(f"Using this ffmpeg path: {ffmpeg_binary}")
    print(f"Using {args.processes} parallel processes")
    print(f"Output directory: {args.output_dir}")
    print(f"Downsizing videos so smallest side <= {args.smallest_side} px (aspect ratio preserved)")
    print("-" * 50)

    # Process files in parallel
    with Pool(processes=args.processes) as pool:
        results = pool.map(process_file, input_files)

    successful = sum(results)
    total = len(input_files)
    print("-" * 50)
    print(f"Processing complete: {successful}/{total} files successful")

    if successful < total:
        print(f"{total - successful} files failed to process")
        sys.exit(1)
    else:
        print("All videos successfully re-encoded for FERAL!")
        print(f"Converted videos are in: {args.output_dir}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    """CLI entry point: build the argparse parser with the train/train-config/infer/reencode subcommands, validate cross-argument constraints, then dispatch to the selected subcommand's handler."""
    parser = argparse.ArgumentParser(prog='feral', description='FERAL: Feature Extraction for Recognition of Animal Locomotion')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # feral train
    p_train = subparsers.add_parser('train', help='Run interactive training pipeline')
    p_train.add_argument('video_folder', help='Path to the folder containing training videos')
    p_train.add_argument('label_json_path', help='Path to the label JSON file')
    p_train.add_argument('--mode', choices=['fast', 'max', 'rare'], default=None,
                         help='Preset recipe overlay: '
                              'fast (smallest V-JEPA 2.1 + Parkinson recipe, cheapest); '
                              'max (largest runnable V-JEPA 2.1 + 75%% chunk overlap, best accuracy); '
                              'rare (rare-class robustness: EMA & mixup off + grad-clip + class-weight cap)')
    p_train.add_argument('--resolution', type=int, default=None,
                         help='Square input resolution for training (e.g. 512). Overrides the '
                              'backbone-native default; V-JEPA interpolates positional embeddings.')
    p_train.add_argument('--no-wandb', action='store_true',
                         help='Disable Weights & Biases logging non-interactively; metrics print to stdout only.')
    p_train.add_argument('--checkpoint', '-c', default=None,
                         help='Path to a checkpoint to resume from')
    p_train.add_argument('--part_subsample', type=float, default=None,
                         help='Fraction (0-1) of training samples to keep')
    p_train.add_argument('--subsample_keep_rare_threshold', type=float, default=None,
                         help='Keep all chunks with rare behaviors below this frequency threshold')
    p_train.set_defaults(func=_cmd_train)

    # feral train-config
    p_train_cfg = subparsers.add_parser('train-config', help='Run training from a YAML config file')
    p_train_cfg.add_argument('config', help='Path to a YAML config file')
    p_train_cfg.set_defaults(func=_cmd_train_config)

    # feral infer
    p_infer = subparsers.add_parser('infer', help='Run inference on a folder of videos')
    p_infer.add_argument('checkpoint', help='Path to a model checkpoint')
    p_infer.add_argument('video_folder', help='Path to folder containing videos')
    p_infer.add_argument('--output', '-o', default=None,
                         help='Output JSON path (default: inference_<folder_name>.json)')
    p_infer.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size (default: 8)')
    p_infer.add_argument('--num_workers', '-w', type=int, default=4, help='DataLoader workers (default: 4)')
    p_infer.add_argument('--compile', action='store_true', help='Compile model with torch.compile')
    p_infer.add_argument('--mode', choices=['fast', 'max'], default=None,
                         help='Inference overlap preset (model size is fixed by the checkpoint): '
                              'fast (50%% chunk overlap, faster); max (75%% overlap, smoother labels, slower).')
    p_infer.add_argument('--resolution', type=int, default=None,
                         help='Override the square input resolution at inference (default: as trained, '
                              'read from the checkpoint).')
    p_infer.set_defaults(func=_cmd_infer)

    # feral reencode
    p_reencode = subparsers.add_parser('reencode', help='Re-encode videos for FERAL processing')
    p_reencode.add_argument('input_dir', help='Directory containing input videos')
    p_reencode.add_argument('output_dir', help='Directory for re-encoded videos')
    p_reencode.add_argument('--processes', '-p', type=int, default=4,
                            help='Number of parallel processes (default: 4)')
    p_reencode.add_argument('--smallest-side', '-s', type=int, default=512,
                            help='Downsize videos so their smallest side is at most this value, '
                                 'preserving aspect ratio. Videos already smaller are left as-is. '
                                 '(default: 512)')
    p_reencode.set_defaults(func=_cmd_reencode)

    # Validate cross-arg constraints for train
    args = parser.parse_args()

    if args.command == 'train':
        if not os.path.isdir(args.video_folder):
            parser.error(f"Video folder is not a directory: {args.video_folder}")
        if not os.path.isfile(args.label_json_path):
            parser.error(f"Label JSON path is not a file: {args.label_json_path}")
        if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
            parser.error(f"Checkpoint path is not a file: {args.checkpoint}")
        if args.part_subsample is not None and not (0.0 <= args.part_subsample <= 1.0):
            parser.error(f"--part_subsample must be between 0 and 1, got {args.part_subsample}")
        if args.subsample_keep_rare_threshold is not None:
            if args.part_subsample is None:
                parser.error("--subsample_keep_rare_threshold requires --part_subsample to be set")
            if not (0.0 <= args.subsample_keep_rare_threshold <= 1.0):
                parser.error(f"--subsample_keep_rare_threshold must be between 0 and 1, got {args.subsample_keep_rare_threshold}")

    if args.command == 'infer':
        if not os.path.isfile(args.checkpoint):
            parser.error(f"Checkpoint not found: {args.checkpoint}")
        if not os.path.isdir(args.video_folder):
            parser.error(f"Video folder not found: {args.video_folder}")

    if args.command == 'train-config':
        if not os.path.isfile(args.config):
            parser.error(f"Config file not found: {args.config}")

    args.func(args)


if __name__ == '__main__':
    main()
