from train import main
import argparse
import yaml
import os
import wandb
from urllib.parse import urlparse

from utils import get_random_run_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the FERAL training pipeline.")
    parser.add_argument('video_folder', help="Path to the folder containing training videos.")
    parser.add_argument('label_json_path', help="Path to the label JSON file.")
    parser.add_argument(
        '--checkpoint',
        '-c',
        default=None,
        help="Optional path to a checkpoint to load into cfg['starting_checkpoint']."
    )
    args = parser.parse_args()

    prefix_path = args.video_folder
    label_json_path = args.label_json_path
    checkpoint_path = args.checkpoint

    assert os.path.exists(prefix_path), f"Prefix path does not exist: {prefix_path}"
    assert os.path.exists(label_json_path), f"Label JSON file does not exist: {label_json_path}"
    if checkpoint_path is not None:
        assert os.path.exists(checkpoint_path), f"Checkpoint path does not exist: {checkpoint_path}"

    with open('configs/default_vjepa.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['data']['prefix'] = prefix_path
    cfg['data']['label_json'] = label_json_path
    cfg['run_name'] = get_random_run_name()
    if checkpoint_path is not None:
        cfg['starting_checkpoint'] = checkpoint_path

    res = input('\nDo you want logs for your run to be on a community Weights & Biases account? No setup required, but everyone will be able to see logs for your run. You can also create your personal project on WandB and log there. Type "open" or "personal": ').strip().lower()
    if res == "open":
        print("Using shared account")
        wandb.login(key=cfg['wandb']['key'])
    elif res == "personal":
        key = input('Paste your wandb api_key: ').strip()
        wandb.login(key=key)
        link = input("paste link to the project where you want to log your runs: ")
        link = urlparse(link)
        assert link.netloc == f'wandb.ai', "should be link to wandb.ai, got {link.netloc}"
        cfg['wandb']['entity'] = link.path.split('/')[1]
        cfg['wandb']['project'] = link.path.split('/')[2]
        print(f"Entity: {cfg['wandb']['entity']} project: {cfg['wandb']['project']}")
    else:
        print(f'Should be "open" or "personal". Got {res}')

    main(cfg)
