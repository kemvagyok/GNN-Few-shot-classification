import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class resnetModel(nn.Module):
    def __init__(self, output_dim=64, in_channels=3, version = 18):
        super().__init__()
        if version == 18:
            resnet = models.resnet18(weights = None)
        elif version == 50:
            resnet = models.resnet50(weights = None)

        # --- 1) Első Conv módosítása (1 vagy 3 csatorna, tetszőleges inputméret) ---
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Ha 1 csatorna esetén a súlyokat is szeretnéd átlagolni:
        if in_channels == 1:
            resnet.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        # --- 2) Fix avgpool helyett AdaptiveAvgPool: bármekkora inputra jó ---
        resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --- 3) Utolsó FC eltávolítása ---
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.features(x)          # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
