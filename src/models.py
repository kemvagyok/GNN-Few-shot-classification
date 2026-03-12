import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data





class CNNModel(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_size, 32,  kernel_size=3, stride=2, padding=1) # 14 x 14
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1) # 7 x 7
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 6 x 6
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1) # 3 x 3
        self.bn4 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256* 3* 3, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = F.normalize(x, p=2, dim=1)  # L2 normalizálás a latens térben

        output = x
        return output
    


class GCNModel(torch.nn.Module):
  def __init__(self,num_features, num_classes):
    super().__init__()
    self.conv1 = GCNConv(
        num_features, 32
    )
    self.conv2 = GCNConv(
        32, num_classes
    )

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index)
    return x


