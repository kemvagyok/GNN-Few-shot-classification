from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class MNISTDataset(Dataset):
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
