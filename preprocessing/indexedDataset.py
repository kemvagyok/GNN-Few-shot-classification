import torch
from torch.utils.data import Dataset
import numpy as np
#-----------
from .image.imageDatasetInMemory import ImageDatasetInMemory
from .tabular.tabularDatasetInMemory import TabularDatasetInMemory
from .text.textDatasetInMemory import TextDatasetInMemory
#-----------

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 base: ImageDatasetInMemory|TabularDatasetInMemory|TextDatasetInMemory, 
                 indices):
        self.base = base
        self.indices = np.array(indices)

        assert self.indices.max() < len(base), "Index out of range!"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]
    
    def get_all(self):
        return self.base.get_all(self.indices)

    def get_batch(self, indices):
        return self.base.get_all(self.indices)
        
    def get_indices(self):
        return self.indices
    
    def fit(self): # FOR UNSW/tabular datasets
        self.base.fit()

    def memory_usage_mb(self):
        return self.base.memory_usage_mb()