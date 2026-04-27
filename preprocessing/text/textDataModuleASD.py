import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def encode_texts(self,  texts, max_len=128, tokenizer_name="bert-base-uncased",):
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = self.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    return encodings['input_ids'], encodings['attention_mask']
