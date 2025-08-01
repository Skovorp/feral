#!/bin/bash

CONFIG_DIR="/home/petr/video_understanding/configs/freezing_ablations"
CODE_DIR="/home/petr/video_understanding"
PYTHON_ENV="/home/petr/miniconda3/envs/feral/bin/python"

for config in "$CONFIG_DIR"/*.yaml; do
    srun -p h100 --gres=gpu:1 --cpus-per-task=16 \
        /bin/bash -c "cd $CODE_DIR && $PYTHON_ENV train.py $config" &
done

wait
