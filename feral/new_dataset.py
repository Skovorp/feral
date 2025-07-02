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

def load_tsr(pth):
    with safe_open(pth, framework="pt", device='cpu') as f:
        return f.get_tensor('data')

def read_range_video_decord(path, start_frame, end_frame):
    vr = VideoReader(path)
    frames = range(start_frame, end_frame + 1)
    video = vr.get_batch(frames).asnumpy()  # (T, H, W, C)
    return torch.from_numpy(video).permute(0, 3, 1, 2)

class ClsDataset():
    def __init__(self, partition, cache_dir, do_aa, predict_per_item, num_classes, prefix, resize_to, **kwargs):
        self.prefix = prefix
        self.cache_dir = cache_dir
        self.partition = partition
        self.predict_per_item = predict_per_item
        self.num_classes = num_classes
        if do_aa:
            self.aug = AutoAugment()
        else:
             self.aug = None
        cleaned = pd.read_csv(os.path.join(cache_dir, partition + '.csv'), header=None, delimiter=',')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        self.resize = torchvision.transforms.v2.Resize((resize_to, resize_to), antialias=True)
        self.norm = torchvision.transforms.v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.scale = 0.00392156862745098

    def proc_target(self, target):
        target = eval(target)
        target = torch.tensor(target).long()
        target = one_hot(target, self.num_classes).float() 
        return target
    
    def proc_names(self, sample):
        sample = sample.split('.')[0]
        if self.predict_per_item > 1:
            sample = [f"{sample}_{i}" for i in range(self.predict_per_item)]
        return sample

    def get_video(self, i):
        sample = self.dataset_samples[i]
        # get file name and start + end frames 
        # read from the middle with decord?
        match = re.search(r"([^/]+)_from_(\d+)_to_(\d+)", sample)
        fn = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        pth = os.path.join(self.prefix, fn + '.mp4')
        # print(fn, start, end)
        return read_range_video_decord(pth, start, end), sample



        # pth = os.path.join(self.cache_dir, 'data', sample)
        # return load_tsr(pth), sample


    def __getitem__(self, index):
        video, name = self.get_video(index)
        video = video if self.aug is None else self.aug(video)
        video = self.resize(video)
        outputs = self.norm(video * self.scale)
        label = self.label_array[index]
        label = self.proc_target(label)
        if self.partition == 'train':
            return outputs, label
        else:
            return outputs, label, self.proc_names(name)
    
    
    def __len__(self):
        return len(self.dataset_samples)


def collate_fn_val(batch):
    tensors, targets, names = zip(*batch)
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)
    return tensors, targets, names


if __name__ == "__main__":
    import yaml
    with open('configs/ants.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    ds = ClsDataset(partition='val', predict_per_item=16, num_classes=3, **cfg['data'])
    print(ds[0])