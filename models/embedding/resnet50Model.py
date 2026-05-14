import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from ..registry import register_embedding

@register_embedding("resnet50")
class Resnet50Model(nn.Module):
    def __init__(self, output_dim, channel_size, isFreeze = False, isClassificator = False):
        super().__init__()

        self.isClassificator = isClassificator

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # --- 1) Első Conv módosítása (1 vagy 3 csatorna, tetszőleges inputméret) ---
        """
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            channel_size,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        """
        # Ha 1 csatorna esetén a súlyokat is szeretnéd átlagolni:
        if channel_size == 1:
            resnet.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        # --- 2) Fix avgpool helyett AdaptiveAvgPool: bármekkora inputra jó ---
        resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --- 3) Utolsó FC eltávolítása ---
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if isFreeze:
            for p in self.features.parameters():
                p.requires_grad = False

        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.features(x)          # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if not self.isClassificator: #Ha nem klasszifikátor, hakkor normalizálni kell a GNN-hez.
            x = F.normalize(x, p=2, dim=1)
        return x