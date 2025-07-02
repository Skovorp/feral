import pandas as pd
import torchvision
from transformers import AutoProcessor
import os
import decord
from decord import VideoReader, cpu
import torch
from torchvision.transforms.v2 import AutoAugment
from torch.nn.functional import one_hot
import numpy as np
from safetensors import safe_open
import re
import cv2
import json

def read_range_video_decord(path, start_frame, end_frame):
    vr = VideoReader(path)
    frames = range(start_frame, end_frame + 1)
    video = vr.get_batch(frames).asnumpy()  # (T, H, W, C)
    return torch.from_numpy(video).permute(0, 3, 1, 2)

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
    # cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # cap.release()
    # return frame_count
    vr = VideoReader(video_path)
    return len(vr)

class ClsDataset():
    def __init__(self, partition, label_json, do_aa, predict_per_item, num_classes, prefix, resize_to, chunk_shift, chunk_length, **kwargs):
        self.prefix = prefix
        self.partition = partition
        self.predict_per_item = predict_per_item
        self.num_classes = num_classes
        self.parse_json(label_json, chunk_shift, chunk_length)
        if do_aa:
            self.aug = AutoAugment()
        else:
             self.aug = None
        
        self.resize = torchvision.transforms.v2.Resize((resize_to, resize_to), antialias=True)
        self.norm = torchvision.transforms.v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.scale = 0.00392156862745098

    def parse_json(self, label_json, chunk_shift, chunk_length):
        with open(label_json, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        self.labels = []

        for fn in data['splits'][self.partition]:
            frame_ids = get_frame_ids(
                get_frame_count(os.path.join(self.prefix, fn)), chunk_shift, chunk_length
            )
            for frames in frame_ids:
                start_frame = min(frames)
                end_frame = max(frames)
                self.samples.append((fn, start_frame, end_frame))
                self.labels.append(
                    data['labels'][fn][start_frame: end_frame + 1]
                )

    def proc_target(self, target):
        target = torch.tensor(target).long()
        if len(target.shape) == 1:
            target = one_hot(target, self.num_classes)
        target = target.float() 
        return target
    
    def proc_names(self, sample):
        if self.predict_per_item > 1:
            sample = [f"{sample}_{i}" for i in range(self.predict_per_item)]
        return sample

    def get_video(self, i):
        fn, start, end = self.samples[i]
        pth = os.path.join(self.prefix, fn)
        return read_range_video_decord(pth, start, end), f"{fn}_from_{start}_to_{end}"

    def __getitem__(self, index):
        video, name = self.get_video(index)
        video = video if self.aug is None else self.aug(video)
        video = self.resize(video)
        outputs = self.norm(video * self.scale)
        label = self.labels[index]
        label = self.proc_target(label)
        if self.partition == 'train':
            return outputs, label
        else:
            return outputs, label, self.proc_names(name)
    
    
    def __len__(self):
        return len(self.samples)


def collate_fn_val(batch):
    tensors, targets, names = zip(*batch)
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)
    return tensors, targets, names


if __name__ == "__main__":
    import yaml
    with open('configs/ahen/mabe_beetle.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    ds = ClsDataset(partition='train', predict_per_item=16, num_classes=3, **cfg['data'])
    print(ds[0])