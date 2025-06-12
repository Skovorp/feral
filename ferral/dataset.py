import pandas as pd
import torchvision
from transformers import AutoProcessor
import os
import decord
from decord import VideoReader, cpu
import torch
from torchvision.transforms.v2 import AutoAugment
from torch.nn.functional import one_hot

def read_video_decord(path):
    vr = VideoReader(path)
    video = vr.get_batch(range(len(vr))).asnumpy()  # (T, H, W, C)
    return torch.from_numpy(video).permute(0, 3, 1, 2)

class ClsDataset():
    def __init__(self, partition, model_name, data_path, prefix, rescale_to, do_aa, predict_per_item, num_classes, **kwargs):
        self.prefix = prefix
        self.partition = partition
        self.predict_per_item = predict_per_item
        self.num_classes = num_classes
        if do_aa:
            self.aug = AutoAugment()
        else:
             self.aug = None
        cleaned = pd.read_csv(os.path.join(data_path, partition + '.csv'), header=None, delimiter=',')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.video_processor.video_sampling['video_size']['longest_edge'] = rescale_to
        self.processor.video_processor.max_image_size['longest_edge'] = rescale_to

    def proc_target(self, target):
        target = torch.tensor(target).long()
        if self.predict_per_item != 1:
            target = target.unsqueeze(1) // 10 ** torch.arange(self.predict_per_item - 1, -1, -1, device=target.device)
            target = target % 10
        target = one_hot(target, self.num_classes).float() 
        return target

    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        pth = os.path.join(self.prefix, sample)
        # video, _, _ = torchvision.io.read_video(pth, pts_unit='sec')
        # video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
        video = read_video_decord(pth)
        video = video if self.aug is None else self.aug(video)
        outputs = self.processor(videos=video, return_tensors="pt")['pixel_values'][0]
        label = self.label_array[index]
        label = self.proc_target(label)
        if self.partition == 'train':
            return outputs, label
        else:
            return outputs, label, sample # TODO: fix sample for packed, make sure that logs work
    
    
    def __len__(self):
        return len(self.dataset_samples)

    
if __name__ == "__main__":
    ds = ClsDataset('train', 'HuggingFaceTB/SmolVLM2-500M-Instruct', '/data/petr/caltech_mice/16frame_single', '/home/petr/home_datasets/videos16', 768)
    vid = ds[0][0]
    print(vid.shape)