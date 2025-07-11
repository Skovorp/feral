from train import main
import yaml
import sys
import os

if __name__ == '__main__':
    assert len(sys.argv) == 3, "Usage: python train.py <prefix_path> <label_json_path>"

    prefix_path = sys.argv[1]
    label_json_path = sys.argv[2]

    assert os.path.exists(prefix_path), f"Prefix path does not exist: {prefix_path}"
    assert os.path.exists(label_json_path), f"Label JSON file does not exist: {label_json_path}"

    with open('configs/default_vjepa.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['data']['prefix'] = prefix_path
    cfg['data']['label_json'] = label_json_path

    main(cfg)
