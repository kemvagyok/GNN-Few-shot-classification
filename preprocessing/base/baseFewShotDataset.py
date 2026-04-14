from torch.utils.data import Dataset
import numpy as np
import torch

class BaseFewShotDataset(Dataset):
    def __init__(self, train_y, num_class, device=None):
        self.y = train_y
        self.train_y_full = train_y
        self.num_class = num_class
        self.device = device

        self.train_mask = None

    def create_train_mask(self, train_size, total_size):
        mask = torch.zeros(total_size, dtype=torch.bool)
        mask[:train_size] = True
        return mask
        
    @staticmethod
    def traindatasetFiltering(targets, num_class, max_size_as_class):
        indexs = np.arange(len(targets))
        train_targets_index_bool = [targets == y for y in range(num_class)]
        train_targets_indexs = [np.asarray(indexs[index_bool]) for index_bool in train_targets_index_bool]

        for c in range(num_class):
            np.random.shuffle(train_targets_indexs[c])

        train_mask_index = np.hstack(
            [train_targets_indexs[target][:max_size_as_class] for target in range(num_class)]
        )
        return train_mask_index