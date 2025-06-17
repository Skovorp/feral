import pandas as pd
import torchvision
from transformers import AutoProcessor
import os
import decord
from decord import VideoReader, cpu
import torch
from torchvision.transforms.v2 import AutoAugment
from torch.nn.functional import one_hot
import time
# import av
import numpy as np

def read_single_video_decord(path):
    vr = VideoReader(path)
    video = vr.get_batch(range(len(vr))).asnumpy()  # (T, H, W, C)
    return torch.from_numpy(video).permute(0, 3, 1, 2)

def read_range_video_decord(path, start_frame, end_frame):
    vr = VideoReader(path)
    frames = range(start_frame, end_frame + 1)
    video = vr.get_batch(frames).asnumpy()  # (T, H, W, C)
    return torch.from_numpy(video).permute(0, 3, 1, 2)

def read_range_video_pyav(path, start_frame, end_frame):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'

    frames = []
    for frame_index, frame in enumerate(container.decode(stream)):
        if frame_index > end_frame:
            break
        if frame_index >= start_frame:
            img = frame.to_rgb().to_ndarray()  # (H, W, 3)
            frames.append(img)

    container.close()
    
    video = np.stack(frames)  # (T, H, W, 3)
    return torch.from_numpy(video).permute(0, 3, 1, 2)

class ClsDataset():
    def __init__(self, partition, dataset_type, data_path, prefix, rescale_to, do_aa, predict_per_item, num_classes, **kwargs):
        self.prefix = prefix
        self.partition = partition
        self.predict_per_item = predict_per_item
        self.num_classes = num_classes
        self.dataset_type = dataset_type
        assert self.dataset_type in ('raw', 'chunk'), "dataset_type should be raw or chunk"
        if do_aa:
            self.aug = AutoAugment()
        else:
             self.aug = None
        cleaned = pd.read_csv(os.path.join(data_path, partition + '.csv'), header=None, delimiter=',')
        self.dataset_samples = list(cleaned.values[:, :-1])
        self.label_array = list(cleaned.values[:, -1])

        self.resize = torchvision.transforms.v2.Resize((rescale_to, rescale_to), antialias=True)
        self.norm = torchvision.transforms.v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.scale = 0.00392156862745098

    def proc_target(self, target):
        target = torch.tensor(target).long()
        if self.predict_per_item > 1:
            target = target.unsqueeze(0) // 10 ** torch.arange(self.predict_per_item - 1, -1, -1, device=target.device)
            target = target % 10
        target = one_hot(target, self.num_classes).float() 
        return target
    
    def proc_names(self, sample):
        sample = sample.split('.')[0]
        if self.predict_per_item > 1:
            sample = [f"{sample}_{i}" for i in range(self.predict_per_item)]
        return sample

    def get_video(self, i):
        if self.dataset_type == "chunk":
            sample = self.dataset_samples[i][0]
            pth = os.path.join(self.prefix, sample)
            return read_single_video_decord(pth), sample
        elif self.dataset_type == 'raw':
            sample, start_frame, end_frame = self.dataset_samples[i]
            pth = os.path.join(self.prefix, sample)
            return read_range_video_decord(pth, start_frame, end_frame), f"{sample}_from_{start_frame}_to_{end_frame}"
        else:
            assert False


    def __getitem__(self, index):
        video, name = self.get_video(index)
        video = self.resize(video)
        video = video if self.aug is None else self.aug(video)
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
    # ds = ClsDataset('train', 'chunk', '/data/petr/caltech_mice/16frames_multiple', '/home/petr/home_datasets/videos16', 512, do_aa=True, predict_per_item=16, num_classes=4)
    # print(ds[0])
    ds = ClsDataset('val', 'raw', '/data/petr/ant_data/16_multiple_raw', '/data/petr/ant_data/raw/raw_videos', 512, do_aa=True, predict_per_item=16, num_classes=4)
    print(ds[0])