from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os

class TabularDatasetInMemory(Dataset):
    def __init__(self, df):
        self.df = df
        self.isTransformed = False

    def add_transformed(self, attributes, labels, class_num):
          self.attributes, self.labels = attributes, labels
          self.class_num = class_num
          self.isTransformed = True
    def __len__(self):
        assert self.isTransformed, "The df is not transformed"
        return len(self.attributes)

    def __getitem__(self, idx):
        assert self.isTransformed, "The df is not transformed"
        x = self.attributes[idx]
        y = self.labels[idx]

        return {
            "inputs": {
                "x": x
            },
            "labels": y
        }
    
    def get_all(self, indices):
        if self.isTransformed:
            return {
                "inputs": {
                    "x": self.attributes[indices]
                },
                "labels": self.labels[indices]
            }

    def get_inputs(self, indices):
        return {
            "x": self.attributes[indices]
        }

    def get_labels(self, indices):
        return self.labels[indices]
        
    def memory_usage_mb(self):
        total = 0

        if self.isTransformed:
            # attributes
            if torch.is_tensor(self.attributes):
                total += self.attributes.element_size() * self.attributes.nelement()
            elif isinstance(self.attributes, np.ndarray):
                total += self.attributes.nbytes

            # labels
            if torch.is_tensor(self.labels):
                total += self.labels.element_size() * self.labels.nelement()
            elif isinstance(self.labels, np.ndarray):
                total += self.labels.nbytes

        # df külön (Pandas overhead, csak becslés)
        if hasattr(self, "df") and self.df is not None:
            total += self.df.memory_usage(deep=True).sum()

        return {
            "total_MB": total / 1e6,
            "total_GB": total / 1e9
        }