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

    def get_inputs(self, indices):
        return {
            "x": self.images[indices]
        }

    def get_labels(self, indices):
        return self.labels[indices]

    def memory_usage_mb(self):
        img_bytes = self.images.element_size() * self.images.nelement()
        lbl_bytes = self.labels.element_size() * self.labels.nelement()

        total_bytes = img_bytes + lbl_bytes

        return {
            "images_MB": img_bytes / 1e6,
            "labels_MB": lbl_bytes / 1e6,
            "total_MB": total_bytes / 1e6,
            "total_GB": total_bytes / 1e9
        }