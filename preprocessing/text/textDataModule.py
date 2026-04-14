import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class TextDataModule:
    def __init__(self, path_raw, tokenizer_name="bert-base-uncased", max_len=128, **kwargs):
        self.path = path_raw
        self.max_len = max_len

        # tokenizer (huggingface)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def encode_texts(self, texts):
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return encodings['input_ids'], encodings['attention_mask']

    @staticmethod
    def split_data(texts, masks, labels, test_size=0.2, val_size=0.2):
        train_x, test_x, train_m, test_m, train_y, test_y = train_test_split(
            texts, masks, labels, stratify=labels, test_size=test_size, random_state=42
        )

        train_x, val_x, train_m, val_m, train_y, val_y = train_test_split(
            train_x, train_m, train_y, test_size=val_size, random_state=42
        )

        return (train_x, train_m), (val_x, val_m), (test_x, test_m), train_y, val_y, test_y