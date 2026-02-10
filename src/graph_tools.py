import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def p_norm_distance_index(latent, p=2, k=3):
    knn_indices = []

    for i in range(latent.shape[0]):
        diff = latent[i].unsqueeze(0) - latent
        dist = diff.norm(p=p, dim=1)
        dist[i] = float("inf")
        knn_indices.append(torch.topk(dist, k, largest=False).indices)
    return torch.stack(knn_indices, dim=0)

def cosine_similarity_index_chunked(latent, k=3, chunk_size=1024):
    latent = F.normalize(latent, p=2, dim=1)
    N = latent.size(0)
    device = latent.device

    knn_indices = []

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)

        # [chunk, D] x [D, N] -> [chunk, N]
        sim = torch.matmul(latent[i:end], latent.T)

        # önmagát kizárjuk
        row_ids = torch.arange(i, end, device=device)
        sim[torch.arange(end - i), row_ids] = float("-inf")

        knn = torch.topk(sim, k, dim=1, largest=True).indices
        knn_indices.append(knn)

        del sim  # segít a memórián

    return torch.cat(knn_indices, dim=0)

def create_edge_index(knn):
    N, k = knn.shape
    row = torch.arange(N, device=knn.device).repeat_interleave(k)
    col = knn.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index.long()


def graph_creating(latens_vectors, neighb_method, p=2):
  if neighb_method == 'cosine':
    top_neighbours = cosine_similarity_index_chunked(latens_vectors)
  elif neighb_method == 'p_norm':
    top_neighbours = p_norm_distance_index(latens_vectors, p=p)
  edge_index = create_edge_index(top_neighbours)
  data = Data(x= latens_vectors, edge_index=edge_index)
  data.num_classes = 10
  return data