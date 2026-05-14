import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class TextDataModule:
    # 1. Változás: A default tokenizer nevet átírtuk egy Qwen modellre
    def __init__(self, path_raw, tokenizer_name="Qwen/Qwen2.5-7B", max_len=128, **kwargs):
        self.path = path_raw
        self.max_len = max_len

        # 2. Változás: betöltés (a trust_remote_code=True néha kell a Qwen modellekhez)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            trust_remote_code=True
        )

        # 3. Változás: Padding token beállítása!
        # A Qwen (és sok más modern LLM) nem rendelkezik dedikált [PAD] tokennel.
        # Hogy a batch-elt padding működjön, be kell állítanunk a pad_tokent (általában az eos_token-re).
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_texts(self, texts):
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return encodings['input_ids'], encodings['attention_mask']