import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
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