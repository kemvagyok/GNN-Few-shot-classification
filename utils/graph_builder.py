import torch
import faiss
from .graph_tools import create_edge_index

def build_graph(cnn, train_test_x, config, device):
    with torch.no_grad():
        latens = cnn(train_test_x.to(device))
        latens_cpu = latens.detach().cpu().numpy()

        index = faiss.IndexFlatL2(latens_cpu.shape[1])
        
        index.add(latens_cpu)

        _, I = index.search(latens_cpu, config.K_neigh + 1)

        neighbors = torch.tensor(I[:,1:], device=device)
        edge_index = create_edge_index(neighbors).to(device)

    return latens, edge_index