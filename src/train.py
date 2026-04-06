from evaluate import evaluate_model
from models import cnnModel, resnetModel, gcnModel, gatv2convModel, BERTModel
from preprocessing.loadingModule import traindatasetFiltering
from utils import build_graph, create_edge_index, is_main_process, reduce_value, dataAboutSpaceGPU

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import faiss
import wandb


# =========================
# Dataset előkészítés
# =========================
def prepare_dataset(train_x, train_y, val_x, val_y, num_class, max_label, device):
    train_mask_index = traindatasetFiltering(train_y, num_class, max_label)
    train_x_filtered = train_x[train_mask_index]
    train_y_filtered = train_y[train_mask_index]
    val_x = val_x
    val_y = val_y
    train_val_x = torch.cat((train_x_filtered, val_x))
    train_val_y = torch.cat((train_y_filtered, val_y))
    train_mask = torch.zeros(len(train_val_y), dtype=torch.bool)
    train_mask[:len(train_x_filtered)] = True
    return train_val_x, train_val_y, train_mask


# =========================
# Modellek inicializálása
# =========================
def initialize_models(config, channel_size, num_class, device, is_ddp):
    if config.embedding == "cnn":
        embedder = cnnModel(output_dim=64, channel_size=channel_size).to(device)
    elif config.embedding == "resnet18":
        embedder = resnetModel(output_dim=64, in_channels=channel_size, version=18).to(device)
    elif config.embedding == "bert":
        embedder = BERTModel().to(device)
    else:
        raise ValueError(f"Unknown embedding: {config.embedding}")

    if config.gcn_model == "GCN":
        gcn = gcnModel(num_features=config.latens_size, num_classes=num_class).to(device)
    elif config.gcn_model == "GAT":
        gcn = gatv2convModel(num_features=config.latens_size, num_classes=num_class).to(device)
    else:
        raise ValueError(f"Unknown gcn_model: {config.gcn_model}")

    gcn = gcnModel(num_features=config.latens_size, num_classes=num_class).to(device)

    if is_ddp:
        if not config.use_minibatch:
            raise ValueError("DDP only supported with minibatch mode.")
        embedder = DDP(embedder, device_ids=[device.index])
        gcn = DDP(gcn, device_ids=[device.index])

    opt_embedder = torch.optim.Adam(embedder.parameters(), lr=config.lr_embedder)
    opt_gcn = torch.optim.Adam(gcn.parameters(), lr=config.lr_gcn)

    return embedder, gcn, opt_embedder, opt_gcn


# =========================
# Gráf létrehozása
# =========================
def build_graph_data(embedder, train_val_x, train_val_y, train_mask, config, device):
    latens, edge_index = build_graph(embedder, train_val_x, config, device)
    data = Data(x=latens, edge_index=edge_index).to(device)
    data.y = train_val_y
    data.train_mask = train_mask
    return data


# =========================
# Loader létrehozása minibatch-hez
# =========================
def create_loader(data, train_mask, config, K_hop, is_ddp):
    if not config.use_minibatch:
        return None

    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    if is_ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        train_idx = train_idx[rank::world_size]

    loader = NeighborLoader(
        data,
        num_neighbors=[config.K_neigh] * K_hop,
        input_nodes=train_idx,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4 if not is_ddp else 0,
        persistent_workers=True if not is_ddp else False
    )
    return loader


# =========================
# Graph újraépítése szükség esetén
# =========================
def rebuild_graph_if_needed(embedder, data, train_val_x, config, device, loader, K_hop, is_ddp):
    if not config.rebuild_graph_each_epoch:
        return data, loader

    embedder.eval()
    latens, edge_index = build_graph(embedder, train_val_x, config, device)
    data.x = latens
    data.edge_index = edge_index

    if config.use_minibatch:
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        if is_ddp:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            train_idx = train_idx[rank::world_size]

        loader = NeighborLoader(
            data,
            num_neighbors=[config.K_neigh] * K_hop,
            input_nodes=train_idx,
            batch_size=config.batch_size,
            shuffle=True
        )
    return data, loader


# =========================
# Validation
# =========================
def validate(embedder, gcn, val_x, val_y, config, device):
    assert val_x.dim() == 4, f"val_x wrong: {val_x.shape}"
    assert val_y.dim() == 1, f"val_y wrong: {val_y.shape}"

    embedder.eval()
    gcn.eval()
    with torch.no_grad():
        latens = embedder(val_x)
        index = faiss.IndexFlatL2(latens.shape[1])
        index.add(latens.cpu().numpy())
        _, I = index.search(latens.cpu().numpy(), config.K_neigh + 1)
        neighbors = torch.tensor(I[:, 1:], device=device)
        edge_index = create_edge_index(neighbors)
        data = Data(x=latens, edge_index=edge_index).to(device)
        out = gcn(data)
        pred = out.argmax(dim=1)
        acc = (pred == val_y).float().mean().item()
    return acc


# =========================
# Full batch training
# =========================
def train_full_batch(embedder, gcn, data, train_val_x, opt_embedder, opt_gcn):
    embedder.train()
    gcn.train()
    opt_embedder.zero_grad()
    opt_gcn.zero_grad()
    latens = embedder(train_val_x)
    data.x = latens
    preds = gcn(data)
    loss = F.cross_entropy(preds[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt_embedder.step()
    opt_gcn.step()
    return loss.item()


# =========================
# Mini batch training
# =========================
def train_mini_batch(embedder, gcn, loader, train_val_x, opt_embedder, opt_gcn):
    embedder.train()
    gcn.train()
    total_loss = 0.0
    count = 0
    for subgraph in loader:
        opt_embedder.zero_grad()
        opt_gcn.zero_grad()
        subimages = train_val_x[subgraph.n_id]
        sublatens = embedder(subimages)
        subgraph.x = sublatens
        preds = gcn(subgraph)
        loss = F.cross_entropy(preds[subgraph.train_mask], subgraph.y[subgraph.train_mask])
        loss.backward()
        opt_embedder.step()
        opt_gcn.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(1, count)


# =========================
# Training loop
# =========================
def train_loop(embedder, gcn, data, loader, train_val_x, opt_embedder, opt_gcn, val_x, val_y, config, device, K_hop, is_ddp):
    best_acc = 0.0
    for epoch in range(config.epochs_max):
        if is_main_process() and epoch % 50 == 0:
            print(f"Epoch {epoch}/{config.epochs_max}")
        data, loader = rebuild_graph_if_needed(embedder, data, train_val_x, config, device, loader, K_hop, is_ddp)
        avg_loss = train_mini_batch(embedder, gcn, loader, train_val_x, opt_embedder, opt_gcn) if config.use_minibatch else train_full_batch(embedder, gcn, data, train_val_x, opt_embedder, opt_gcn)
        avg_loss = reduce_value(avg_loss, device)
        if is_main_process():
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss, "epoch": epoch})
        #Validate
        val_acc = validate(embedder, gcn, val_x, val_y, config, device)
        val_acc = reduce_value(val_acc, device)
        best_acc = max(best_acc, val_acc)
        if is_main_process():
            print(f"Epoch {epoch}: Validation Accuracy = {val_acc:.4f}")
            wandb.log({"val_acc": val_acc})
    return best_acc


# =========================
# Fő run_training
# =========================
def run_training(train_x, train_y, val_x, val_y, test_x, test_y, num_class, channel_size, K_hop, max_label, config, device, is_ddp=False):
    train_val_x, train_val_y, train_mask = prepare_dataset(train_x, train_y, val_x, val_y, num_class, max_label, device)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    train_val_x = train_val_x.to(device)
    train_val_y = train_val_y.to(device)

    embedder, gcn, opt_embedder, opt_gcn = initialize_models(config, channel_size, num_class, device, is_ddp)
    data = build_graph_data(embedder, train_val_x, train_val_y, train_mask, config, device)
    loader = create_loader(data, train_mask, config, K_hop, is_ddp)
    best_acc = train_loop(embedder, gcn, data, loader, train_val_x, opt_embedder, opt_gcn, val_x, val_y, config, device, K_hop, is_ddp)

    # =========================
    # Teszt
    # =========================
    if test_x is not None and test_y is not None:
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        test_acc = evaluate_model(embedder, gcn, test_x, test_y, config, device)
        if is_main_process():
            print(f"Final Test Accuracy = {test_acc:.4f}")
            wandb.log({"test_acc": test_acc})
    return best_acc