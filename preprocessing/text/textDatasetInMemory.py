from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os

class TextDatasetInMemory(Dataset):
    def __init__(self, texts, attention_masks, labels, class_num):
        self.texts_full = texts
        self.attention_masks_full = attention_masks
        self.labels_full = labels
        self.class_num = class_num

    def __len__(self):
        return len(self.texts)  

    def __getitem__(self, idx):
        return {
            "inputs": {
                "input_ids": self.texts[idx],
                "attention_mask": self.attention_masks[idx]
            },
            "labels": self.labels[idx]
        }
    
    def get_all(self, indices):
        return {
            "inputs": {
                "input_ids": self.texts[indices],
                "attention_mask": self.attention_masks[indices]
            },
            "labels": self.labels[indices]
        }