import torch.nn as nn
from CNNAutocoder import CNNModel
from GCN_model import GCN
from graph_tools import graph_creating


class FewShotModel(nn.Module):
  def __init__(self, input_size, output_size, latens_size, channel_size , device):
    super(FewShotModel, self).__init__()
    self.encoder = CNNModel(channel_size = channel_size).to(device)
    self.classifier = GCN(latens_size, output_size).to(device)
  def forward(self, x, method_similar, p_n):
    x = self.encoder(x)
    x = graph_creating(x, method_similar, p_n)
    #loader = NeighborLoader(data = x,input_nodes=torch.arange(x.num_nodes, device=device),num_neighbors = 20)
    x = self.classifier(x)
    return x
