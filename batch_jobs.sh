#!/bin/bash

CONFIG_DIR="/home/petr/video_understanding/configs/base_runs"
CODE_DIR="/home/petr/video_understanding"
PYTHON_ENV="/home/petr/miniconda3/envs/feral/bin/python"

for config in "$CONFIG_DIR"/*.yaml; do
    srun -p l40s --gres=gpu:1 -w gpu101 --cpus-per-task=12 \
        /bin/bash -c "cd $CODE_DIR && $PYTHON_ENV train.py $config" &
done

wait
