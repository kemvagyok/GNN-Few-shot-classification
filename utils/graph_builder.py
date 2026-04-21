import torch
import faiss
from .graph_tools import create_edge_index
from torch_geometric.data import Data

def graph_builder(latens, y, K_neigh,device):
    latens_cpu = latens.detach().cpu().numpy()
    index = faiss.IndexFlatL2(latens_cpu.shape[1])
        
    index.add(latens_cpu)

    _, I = index.search(latens_cpu, K_neigh + 1)

    neighbors = torch.tensor(I[:,1:], device=device)
    edge_index = create_edge_index(neighbors).to(device)
    
    data = Data(
        x = latens, 
        edge_index = edge_index).to(device)
    data.y = y.to(device)
    
    return data
