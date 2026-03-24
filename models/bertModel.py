import torch
import torch.nn as nn
from torchvision import models
from transformers import BertTokenizer, BertModel

class bertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.proj = nn.Linear(768, 64)  # 768 → 64

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS]
        out = self.proj(cls)  # [batch, 64]
        
        return out
