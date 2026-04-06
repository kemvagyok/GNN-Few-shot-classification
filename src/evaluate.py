import torch
import wandb
import faiss
from utils import create_edge_index

from torch_geometric.data import Data


def evaluate_model(cnn, gcn, test_x, test_y, config, device):
    cnn.eval()
    gcn.eval()

    with torch.no_grad():
        latens_test = cnn(test_x.to(device))

        # KNN graph a teszt halmazon
        index = faiss.IndexFlatL2(latens_test.shape[1])
        index.add(latens_test.cpu().numpy())

        _, I = index.search(latens_test.cpu().numpy(), config.K_neigh + 1)

        neighbors_test = torch.tensor(I[:,1:], device=device)
        edge_index_test = create_edge_index(neighbors_test)

        data_test = Data(
            x=latens_test,
            edge_index=edge_index_test
        ).to(device)

        out = gcn(data_test)

        pred = out.argmax(dim=1)
        acc = (pred == test_y.to(device)).float().mean().item()

    return acc

