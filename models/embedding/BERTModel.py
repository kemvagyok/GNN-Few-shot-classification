import torch
import torch.nn as nn
from torchvision import models
from transformers import BertTokenizer, BertModel
from ..registry import register_embedding
import torch.nn.functional as F

@register_embedding("bert")
class BERTModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.fc = nn.Linear(self.bert.config.hidden_size, 64)  # 768 → 64
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # A [CLS] token kimenetének kinyerése (pooled_output)
        pooled_output = outputs.pooler_output 
        
        # Vektor levetítése 64 dimenzióra
        latent_vector = self.fc(pooled_output)
        latent_vector = F.normalize(latent_vector, p=2, dim=1)
        
        return latent_vector

