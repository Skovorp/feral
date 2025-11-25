import pandas as pd
import torchvision
from transformers import AutoProcessor
import os
import decord
from decord import VideoReader, cpu
import torch
from torchvision.transforms.v2 import AutoAugment, TrivialAugmentWide
from torch.nn.functional import one_hot
import numpy as np
from safetensors import safe_open
import re
import cv2
import json
import traceback
import random

def read_range_video_decord(path, frames):
    vr = VideoReader(path)
    video = vr.get_batch(frames).asnumpy()  # (T, H, W, C)
    return torch.from_numpy(video).permute(0, 3, 1, 2)

def get_frame_ids(total_frames, chunk_shift, chunk_length, chunk_step):
        # chunk_step = 1 -- pick every frame        XXXX
        # chunk_step = 2 -- pick every other frame  X_X_X_X
        # chunk_step = 3 -- pick every third        X__X__X__X
        vid_frames = []
        start_ind = 0

        while True:
            last_ind = start_ind + (chunk_length - 1) * chunk_step  + 1
            inds = list(range(start_ind, min(last_ind, total_frames), chunk_step))
            if len(inds) != chunk_length:
                break
            vid_frames.append(inds)
            start_ind = inds[0] + chunk_shift
        return vid_frames

# def get_frame_count(video_path):
#     vr = VideoReader(video_path)
#     return len(vr)

def get_frame_count(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None
    finally:
        cap.release()

class ClsDataset():
    def __init__(self, partition, label_json, do_aa, predict_per_item, num_classes, prefix, resize_to, chunk_shift, chunk_length, chunk_step, part_sample=1.0, **kwargs):
        self.prefix = prefix
        self.partition = partition
        self.predict_per_item = predict_per_item
        self.num_classes = num_classes
        self.is_multilabel = None
        
        with open(label_json, 'r') as f:
            self.json_data = json.load(f)
        
        self.parse_json(chunk_shift, chunk_length, chunk_step)
        if do_aa and self.partition == "train":
            self.aug = TrivialAugmentWide() #AutoAugment()
        else:
             self.aug = None
        
        self.resize = torchvision.transforms.v2.Resize((resize_to, resize_to), antialias=True)
        # self.norm = torchvision.transforms.v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # smolvm
        self.norm = torchvision.transforms.v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # vjepa 
        self.scale = 0.00392156862745098

        if part_sample < 1.0 and (partition == 'train'): # or partition == 'val'):
            print(f"{partition} using {100 * part_sample:.2f}% of chunks")
            new_len = round(part_sample * len(self.samples))
            new_indexes = random.sample(list(range(len(self.samples))), new_len)
            self.samples = [self.samples[i] for i in new_indexes]
            self.labels = [self.labels[i] for i in new_indexes]

        if partition != 'inference':
            concat_labels = np.array(self.labels)
            if self.is_multilabel:
                cls_cnts = concat_labels.sum(axis=(0, 1))
            else:
                cls_cnts = np.bincount(concat_labels.flatten())
            print(f"{partition} class counts: {cls_cnts}")


    def parse_json(self, chunk_shift, chunk_length, chunk_step):
        self.samples = []
        self.labels = []

        for fn in self.json_data['splits'][self.partition]:
            frame_ids = get_frame_ids(
                get_frame_count(os.path.join(self.prefix, fn)), chunk_shift, chunk_length, chunk_step
            )
            for frames in frame_ids:
                self.samples.append((fn, frames))
                if self.partition != 'inference':
                    json_total_frames = len(self.json_data['labels'][fn])
                    video_total_frames = get_frame_count(os.path.join(self.prefix, fn))
                    assert json_total_frames == video_total_frames, f"Bad json for video {fn}. Video has {video_total_frames} frames, labels have {json_total_frames} frames"
                    self.labels.append(
                        [self.json_data['labels'][fn][i] for i in frames]
                    )
        if self.partition != 'inference':
            self.is_multilabel = False if len(torch.tensor(self.labels[0]).shape) == 1 else True

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
        fn, frames = self.samples[i]
        pth = os.path.join(self.prefix, fn)
        return read_range_video_decord(pth, frames), [f"{fn}_globalind_{frames[i]}_chunkind_{i}" for i in range(len(frames))]
    
    def get_item_simple(self, index):
        video, names = self.get_video(index)
        video = video if self.aug is None else self.aug(video)
        video = self.resize(video)
        outputs = self.norm(video * self.scale)
        if self.partition != "inference":
            label = self.labels[index]
            label = self.proc_target(label)
        if self.partition == 'train':
            return outputs, label
        elif self.partition == "val" or self.partition == 'test':
            return outputs, label, names
        else:
            return outputs, names

    def __getitem__(self, index):
        try:
            return self.get_item_simple(index)
        except Exception:
            print(f"Error loading index {index}:\n{traceback.format_exc()}")
            for _ in range(3):
                alt_index = np.random.randint(0, len(self))
                try:
                    return self.get_item_simple(alt_index)
                except Exception:
                    print(f"Error loading index {alt_index}:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to load sample after multiple retries.\nLast error:\n{traceback.format_exc()}")

    
    def __len__(self):
        return len(self.samples)


def collate_fn_val(batch):
    tensors, targets, names = zip(*batch)
    tensors = torch.stack(tensors)
    targets = torch.stack(targets)
    return tensors, targets, names

def collate_fn_inference(batch):
    tensors, names = zip(*batch)
    tensors = torch.stack(tensors)
    return tensors, names


if __name__ == "__main__":
    import yaml
    # with open('/home/petr/video_understanding/configs/base_runs/worms_vjepa.yaml', 'r') as f:
    with open('/home/petr/video_understanding/configs/base_runs/monkeys.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    ds = ClsDataset(partition='val', predict_per_item=64, num_classes=5, **cfg['data'])
    outputs, label, names = ds[0]
    print(outputs.shape)
    # print(set([len(x) for x in ds.labels]))
    # for i in range(len(ds)):
    #     if len(ds.labels[i]) == 15:
    #         print(i, ds.samples[i], get_frame_count(os.path.join(ds.prefix, ds.samples[i][0])), len(ds.h['labels'][ds.samples[i][0]]))

    # for el in ds.h['labels'].keys():
    #     a = get_frame_count(os.path.join(ds.prefix, el))
    #     b = len(ds.h['labels'][el])
    #     if a != b:
    #         print(f"{el} true: {a}, json: {b}")
