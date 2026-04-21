from torch.utils.data import Dataset
import numpy as np
import torch

class BaseFewShotDataset(Dataset):
    def __init__(self, y, num_class, device=None):
        self.y = y
        self.num_class = num_class
        self.device = device

    @staticmethod
    def filter_by_class(targets, num_class, max_size_as_class):
        idxs = np.arange(len(targets))
        class_indices = [idxs[targets == c] for c in range(num_class)]

        for c in range(num_class):
            np.random.shuffle(class_indices[c])

        selected = np.hstack([
            class_indices[c][:max_size_as_class] for c in range(num_class)
        ])
        return selected