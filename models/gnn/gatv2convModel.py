import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from ..registry import register_gnn

@register_gnn("gatv2conv")
class Gatv2convModel(torch.nn.Module):
  def __init__(self,num_features, num_classes):
    super().__init__()
    self.conv1 = GATv2Conv(
        num_features, 32
    )
    self.conv2 = GATv2Conv(
        32, num_classes
    )

  def forward(self, data):
    x, edge_index = data.x, data.edge_index
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index)
    return x