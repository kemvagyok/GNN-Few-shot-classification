import torch
import torch.nn as nn
from transformers import AutoModel
from ..registry import register_embedding
import torch.nn.functional as F

@register_embedding("qwen")
class QwenEmbeddingModel(nn.Module):
    def __init__(self, output_dim, model_name="Qwen/Qwen3-Embedding-0.6B", isFreeze=False, isClassificator = False):
        super().__init__()
        
        self.isClassificator = isClassificator

        # A Qwen modellekhez a generikus AutoModel-t használjuk
        self.model = AutoModel.from_pretrained(model_name)
        
        if isFreeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # A Qwen3-Embedding-0.6B alapértelmezett kimeneti dimenziója (hidden_size) 1024.
        # Ezt vetítjük le adott méretű dimenzióra.
        self.fc = nn.Linear(self.model.config.hidden_size, output_dim) 

    def forward(self, input_ids, attention_mask):
        # Mivel ez az alap AutoModel, nem adunk vissza LM head logitokat, csak a rejtett állapotokat.
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        
        # Az összes token rejtett állapota (batch_size, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state 
        
        # Qwen-specifikus "Last Token Pooling"
        # Megkeressük az utolsó érvényes (nem padding) tokent a maszk alapján minden elemnél a batch-ben.
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        
        # Kinyerjük az utolsó érvényes tokenek vektorait -> (batch_size, hidden_size)
        pooled_output = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), 
            sequence_lengths
        ]
        
        pooled_output = pooled_output.to(self.fc.weight.dtype)
        
        # Vektor levetítése 64 dimenzióra
        latent_vector = self.fc(pooled_output)
        if self.isClassificator:
            latent_vector = F.normalize(latent_vector, p=2, dim=1)
        return latent_vector