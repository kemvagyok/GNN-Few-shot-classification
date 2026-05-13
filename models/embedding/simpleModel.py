import torch
from torch import nn
import torch.nn.functional as F
from ..registry import register_embedding

@register_embedding("simple")
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim, channel_size, isFreeze=False, isClassificator = False):
        super().__init__()

        self.isClassificator = isClassificator

        hidden_dims=[256, 128]

        layers = []
        
        dropout = 0.2

        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.backbone = nn.Sequential(*layers)

        # final embedding layer
        self.embedding = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        h = self.backbone(x)
        output = self.embedding(h)

        if self.isClassificator:
            return F.normalize(output, p=2, dim=1)
        return output