import torch
import faiss
from .graph_tools import create_edge_index
from torch_geometric.data import Data

def graph_builderFAISS(latens, y, K_neigh, device):
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

def graph_builderFAISS_withGPU(latens, y, K_neigh, device):
    assert device.type == 'cuda', "GPU is required for this function."

    x = latens.detach().float().cpu().numpy()
    N, d = x.shape

    res = faiss.StandardGpuResources()

    nlist = 128    # fontos tuning paraméter

    quantizer = faiss.IndexFlatL2(d)
    index_cpu = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index.nprobe = 18
    # training (FONTOS!)
    index.train(x)

    index.add(x)

    D, I = index.search(x, K_neigh + 1)
    neighbors = torch.from_numpy(I[:, 1:]).to(device)

    edge_index = create_edge_index(neighbors).to(device)

    data = Data(
        x=latens,
        edge_index=edge_index,
        y=y.to(device)
    ).to(device)

    return data


def graph_builderFAISS_withGPUProba(
    latens,
    y,
    K_neigh,
    device
):

    x = latens.detach().float()

    x_np = x.cpu().numpy()

    N, d = x_np.shape

    # =====================================================
    # FAISS GPU
    # =====================================================
    res = faiss.StandardGpuResources()

    nlist = min(128, max(1, N // 10))

    quantizer = faiss.IndexFlatL2(d)

    index_cpu = faiss.IndexIVFFlat(
        quantizer,
        d,
        nlist,
        faiss.METRIC_L2
    )

    index_cpu.nprobe = min(18, nlist)

    # =====================================================
    # TRAIN
    # =====================================================
    if not index_cpu.is_trained:
        index_cpu.train(x_np)

    # =====================================================
    # GPU INDEX
    # =====================================================
    index = faiss.index_cpu_to_gpu(
        res,
        0,
        index_cpu
    )

    index.add(x_np)

    # =====================================================
    # SEARCH
    # =====================================================
    D, I = index.search(
        x_np,
        K_neigh + 1
    )

    neighbors = torch.from_numpy(
        I[:, 1:]
    ).long()

    edge_index = create_edge_index(
        neighbors
    ).long()

    # =====================================================
    # BUILD GRAPH
    # =====================================================
    data = Data(
        x=latens.cpu(),
        edge_index=edge_index.cpu(),
        y=y.cpu()
    )

    return data


def graph_builderTorch(latens, y, K_neigh, device):

    dist = torch.cdist(latens, latens)

    knn = dist.topk(k = K_neigh + 1, largest = False).indices[:, 1:]

    edge_index = create_edge_index(knn)

    data = Data(
        x = latens,
        edge_index = edge_index,
        y = y
    )

    return data