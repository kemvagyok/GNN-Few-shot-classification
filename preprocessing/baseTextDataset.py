import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class BaseTextDataset:
    def __init__(self, path_raw, tokenizer_name="bert-base-uncased", max_len=128):
        self.path = path_raw
        self.max_len = max_len

        # tokenizer (huggingface)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode_texts(self, texts, labels):
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return encodings, torch.tensor(labels)

    def split_data(self, texts, labels, test_size=0.2, val_size=0.2):
        train_x, test_x, train_y, test_y = train_test_split(
            texts, labels, stratify=labels, test_size=test_size, random_state=42
        )

        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y, test_size=val_size, random_state=42
        )

        return train_x, val_x, test_x, train_y, val_y, test_y