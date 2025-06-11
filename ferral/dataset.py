import pandas as pd
import torchvision
from transformers import AutoProcessor
import os

class ClsDataset():
    def __init__(self, partition, model_name, data_path, prefix, **kwargs):
        self.prefix = prefix
        self.partition = partition
        cleaned = pd.read_csv(os.path.join(data_path, partition + '.csv'), header=None, delimiter=',')
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        pth = os.path.join(self.prefix, sample)
        video, _, _ = torchvision.io.read_video(pth, pts_unit='sec')
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
        outputs = self.processor(videos=video, return_tensors="pt")['pixel_values'][0]
        if self.partition == 'train':
            return outputs, self.label_array[index]
        else:
            return outputs, self.label_array[index], sample
    
    
    def __len__(self):
        return len(self.dataset_samples)