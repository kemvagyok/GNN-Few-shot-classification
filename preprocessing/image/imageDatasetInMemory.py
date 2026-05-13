from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os
from PIL import Image

class ImageDatasetInMemory(Dataset):
    def __init__(self, images, labels, class_num, n_channels):
        self.images = images
        self.labels = labels
        self.class_num = class_num
        self.n_channels = n_channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        return {
            "inputs": {
                "x": x
            },
            "labels": y
        }
    
    def get_all(self, indices):
                return {
            "inputs": {
                "x": self.images[indices]
            },
            "labels": self.labels[indices]
        }
