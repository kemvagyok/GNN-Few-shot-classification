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