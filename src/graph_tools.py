import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def p_norm_distance_index(latent, p=2, k=3):
    diff = latent.unsqueeze(1) - latent.unsqueeze(0)
    dist = diff.norm(p=p, dim=2)                     
    dist = dist + torch.eye(dist.size(0), device=dist.device) * 1e9
    knn = torch.topk(dist, k, dim=1, largest=False).indices  # [N, k]
    return knn

def cosine_similarity_index(latent, k=3):
    normed = F.normalize(latent, p=2, dim=1)
    sim = torch.matmul(normed, normed.T)        
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 2
    knn = torch.topk(sim, k, dim=1, largest=True).indices
    return knn

def create_edge_index(knn):
    N, k = knn.shape
    row = torch.arange(N, device=knn.device).repeat_interleave(k)
    col = knn.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index.long()
	
def graph_creating(latens_vectors, neighb_method, p=2):
  if neighb_method == 'cosine':
    top_neighbours = cosine_similarity_index(latens_vectors)
  elif neighb_method == 'p_norm':
    top_neighbours = p_norm_distance_index(latens_vectors, p=p)
  edge_index = create_edge_index(top_neighbours)
  data = Data(x= latens_vectors, edge_index=edge_index)
  data.num_classes = 10
  return data