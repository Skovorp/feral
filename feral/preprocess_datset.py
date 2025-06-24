import os
import json
import cv2
from multiprocessing import Pool, cpu_count, set_start_method
import multiprocessing
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu
import numpy as np
import glob
from functools import lru_cache
import yaml
from safetensors.torch import save_file
from torchvision.transforms.v2.functional import resize
import torch
from multiprocessing import Manager, Value, Lock
import threading
import time
import re
import sys

class SequentialReader():
    def __init__(self, path, start_ind):
        self.vid = cv2.VideoCapture(path)
        self.seek_to(start_ind)  # Ensure the first call returns the correct image

    def seek_to(self, frame_idx):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.cur_frame = frame_idx - 1
    
    @lru_cache(128)
    def get_frame(self, i):
        if self.cur_frame != i - 1:
            self.seek_to(i)

        self.cur_frame += 1
        ret, frame = self.vid.read()
        if not ret:
            raise Exception("Out of bounds")
        return frame
   
def process_row(args):
    ind, df = args
    unique_paths = df['path'].unique()
    assert len(unique_paths) == 1
    this_path = unique_paths[0]

    if len(df) == 0:
        return

    my_reader = SequentialReader(this_path, df['first_frame'].min())

    for i, row in df.iterrows():
        start_frame, end_frame = row['first_frame'], row['last_frame']

        frames = [
            cv2.resize(my_reader.get_frame(i), (row['resize_to'], row['resize_to']), interpolation=cv2.INTER_LANCZOS4)
            for i in range(start_frame, end_frame + 1)
        ]
        frames = torch.from_numpy(np.stack(frames))  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2).contiguous()

        save_file({'data': frames}, f"{row['outp_path']}.safetensor")

def progress_monitor(cache_dir, total_rows):
    pbar = tqdm(total=total_rows)
    while True:
        existing = len(glob.glob(os.path.join(cache_dir, "*.safetensor")))
        pbar.n = existing
        pbar.refresh()
        if existing >= total_rows:
            break
        time.sleep(1)
    pbar.close()

def main(video_dfs, cache_dir, num_processes):
    new_video_dfs = []
    for single_df in video_dfs:
        for i in range(0, len(single_df) // 1000 + 1):
            chunk = single_df[i * 1000: (i + 1) * 1000]
            if len(chunk) > 0:
                new_video_dfs.append(chunk)
    assert sum([len(x) for x in video_dfs]) == sum([len(x) for x in new_video_dfs])
    total_rows = sum([len(x) for x in video_dfs])
    args = enumerate(new_video_dfs)

    thread = threading.Thread(target=progress_monitor, args=(cache_dir, total_rows))
    thread.start()

    with Pool(processes=num_processes) as pool:
        list(pool.imap(process_row, args))
    thread.join()


def get_frame_ids(total_frames, chunk_shift, chunk_length):
        vid_frames = []
        start_ind = 0

        while True:
            inds = list(range(start_ind, min(start_ind + chunk_length, total_frames)))
            if len(inds) != chunk_length:
                break
            vid_frames.append(inds)
            start_ind = inds[0] + chunk_shift
        return vid_frames


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def make_label_csv(data, cache_dir, file_ext):
    cache = os.listdir(os.path.join(cache_dir, 'data'))
    labels = []
    partitions = []

    for vid in cache:
        match = re.search(r"([^/]+)_from_(\d+)_to_(\d+)\.safetensor$", vid)
        name = match.group(1)
        from_num = int(match.group(2))
        to_num = int(match.group(3))
        lbls = data['labels'][name + '.mp4']['frame_labels'][from_num : to_num + 1]
        partitions.append(data['labels'][name + file_ext]['partition'])
        labels.append(lbls)
    
    df = pd.DataFrame({
        'name': cache,
        'lbl': labels,
        'partition': partitions
    })
    df['name'] = df['name'].apply(lambda x: os.path.basename(x))
    df[df['partition'] == 'train'][['name', 'lbl']].to_csv(os.path.join(cache_dir, 'train.csv'), index=False, header=False)
    df[df['partition'] == 'val'][['name', 'lbl']].to_csv(os.path.join(cache_dir, 'val.csv'), index=False, header=False)

if __name__ == "__main__":
    assert len(sys.argv) > 1 and len(sys.argv[1]) > 0, "Usage: python preprocess_datset.py <path_to_config.yaml>"
    config_path = sys.argv[1]

    multiprocessing.set_start_method("spawn", force=True)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg['data']['cache_dir'], exist_ok=True)
    if os.listdir(cfg['data']['cache_dir']):
        raise AssertionError(f"Cache directory {cfg['data']['cache_dir']} is not empty.")
    file_outp_dir = os.path.join(cfg['data']['cache_dir'], 'data')
    os.makedirs(file_outp_dir)

    with open(cfg['data']['label_json'], 'r') as f:
        label_json = json.load(f)

    vids = [os.path.join(cfg['data']['prefix'], x) for x in label_json['labels'].keys()]
    print(vids)

    assert len(set([x.split('.')[1] for x in vids])) == 1, f"found different extentions in video files: {set([x.split('.')[1] for x in vids])}"
    file_ext = '.' + list(set([x.split('.')[1] for x in vids]))[0]

    dfs = []
    for vid in vids:
        frames = get_frame_ids(get_frame_count(vid), chunk_shift=cfg['data']['chunk_shift'], chunk_length=cfg['data']['chunk_length'])
        this_df = pd.DataFrame({
            'first_frame': [min(x) for x in frames],
            'last_frame':  [max(x) for x in frames],
            'path': vid,
            'resize_to': cfg['data']['resize_to']
        })
        assert len(this_df) != 0, f"df for {vid} is empty. this video has {get_frame_count(vid)} frames. Trying to get these frames: {frames[:3]}..."

        pth = os.path.join(file_outp_dir, os.path.basename(vid).split('.')[0])
        this_df['outp_path'] = this_df.apply(lambda row: f"{pth}_from_{row['first_frame']}_to_{row['last_frame']}", axis=1)
        dfs.append(this_df)

    main(dfs, file_outp_dir, cfg['data']['preproc_processes'])
    make_label_csv(label_json, cfg['data']['cache_dir'], file_ext)