from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os
from PIL import Image

def decode_image(img_path):
    with Image.open(img_path) as img:
        return img.convert("RGB")
    
class ImageDataset(Dataset):
    def __init__(self, images_name, labels, class_num, n_channels, dataset_path, dataset_name, img_full_dir, transform=None, target_transform=None):
        self.images_name = images_name
        self.labels = labels
        self.class_num = class_num
        self.n_channels = n_channels
        self.img_dir = img_full_dir
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        image_name = self.images_name[idx]

        img_path = os.path.join(self.img_dir, image_name)

        x = decode_image(img_path)
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return {
            "inputs": {
                "x": x
            },
            "labels": y
        }
    
