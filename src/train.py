from evaluate import evaluate_model

from models import resnetModel, gcnModel, cnnModel

from preproccesing.loadingModule import traindatasetFiltering

from utils import build_graph, create_edge_index, is_main_process, reduce_value

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data

import faiss

import wandb

def run_training(
    train_x, train_y,
    val_x, val_y,
    test_x, test_y,
    num_class, channel_size,
    K_hop, max_label,
    config, device,
    is_ddp=False
) -> float:
    # =========================
    # Dataset előkészítés
    # =========================
    train_mask_index = traindatasetFiltering(train_y, num_class, max_label)

    train_x_filtered = train_x[train_mask_index].to(device)
    train_y_filtered = train_y[train_mask_index].to(device)

    val_x = val_x.to(device)
    val_y = val_y.to(device)
    train_val_x = torch.cat((train_x_filtered, val_x))
    train_val_y = torch.cat((train_y_filtered, val_y))

    train_mask = torch.zeros(len(train_val_y), dtype=torch.bool)
    train_mask[:len(train_x_filtered)] = True

    # =========================
    # Modellek
    # =========================
    #cnn = cnnModel(channel_size=channel_size).to(device)
    cnn = resnetModel(output_dim=64, in_channels=channel_size).to(device)
    gcn = gcnModel(num_features=config.latens_size,
                num_classes=num_class).to(device)

    if is_ddp:
        if not config.use_minibatch:
            raise ValueError("DDP only supported with minibatch mode.")
        cnn = DDP(cnn, device_ids=[device.index])
        gcn = DDP(gcn, device_ids=[device.index])
        
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=config.lr_cnn)
    opt_gcn = torch.optim.Adam(gcn.parameters(), lr=config.lr_gcn)
    # =========================
    # Gráf inicializálása
    # =========================
    latens, edge_index = build_graph(cnn, train_val_x, config, device)

    data = Data(x=latens, edge_index=edge_index).to(device)
    data.y = train_val_y
    data.train_mask = train_mask
    # =========================
    # Loader (csak ha minibatch)
    # =========================
    if config.use_minibatch:
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        if is_ddp:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            #print(f"Process {rank}: World size: {world_size}")
            train_mask_index = train_idx[rank::world_size]
            loader = NeighborLoader(
                data,
                num_neighbors=[config.K_neigh] * K_hop,
                input_nodes=train_mask_index,  # <-- itt a rank-specifikus subset
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True
            )
        else:
            loader = NeighborLoader(
                data,
                num_neighbors=[config.K_neigh] * K_hop,
                input_nodes=data.train_mask,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True
            )
    else:
        loader = None
    # =========================
    # Training loop
    # =========================
    print(f"\nStarting training for K_hop={K_hop}...")
    print(f"  Training with max_label={max_label}...")

    best_loss = -1
    #best_acc = 0.0
    for epoch in range(config.epochs_max):
        if is_main_process() and epoch % 50 == 0:
            print(f"Epoch {epoch}/{config.epochs_max}")
        
        data, loader = rebuild_graph_if_needed(
            cnn = cnn,
            data = data,
            train_val_x = train_val_x,
            config = config,
            device = device,
            loader = loader,
            K_hop = K_hop,
            is_ddp = is_ddp 
        )

        if config.use_minibatch:
            avg_loss = train_mini_batch(
                cnn, gcn, loader, train_val_x,
                opt_cnn, opt_gcn
            )
        else:
            avg_loss = train_full_batch(
                cnn, gcn, data, train_val_x,
                opt_cnn, opt_gcn
            )

        avg_loss = reduce_value(avg_loss, device)

        if is_main_process():
            #print("Epoch {}: Train Loss = {:.4f}".format(epoch, avg_loss))
            wandb.log({"train_loss": avg_loss, "epoch": epoch})

        # =========================
        # VALIDATION
        # =========================
        val_loss = validate(cnn = cnn, gcn = gcn, val_x = val_x, val_y = val_y, config = config, device = device)

        global_val_loss = reduce_value(val_loss, device)
        
        if global_val_loss > best_loss:
            best_loss = global_val_loss

        if is_main_process():
            #print("Epoch {}: Validation Accuracy = {:.4f}".format(epoch, global_acc))
            wandb.log({"val_loss": global_val_loss})
    # =========================
    # TEST
    # =========================
    if test_x is not None and test_y is not None:
        test_acc = evaluate_model(cnn, gcn, test_x.to(device), test_y.to(device), config, device)

    if is_main_process():
        print(f"Final Test Accuracy = {test_acc:.4f}")
        wandb.log({"test_acc": test_acc})
        
    return best_loss

def rebuild_graph_if_needed(cnn, data, train_val_x, config, device, loader, K_hop, is_ddp):
    if not config.rebuild_graph_each_epoch:
        return data, loader

    cnn.eval()
    latens, edge_index = build_graph(cnn, train_val_x, config, device)

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
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )

    return data, loader

def validate(cnn, gcn, val_x, val_y, config, device):
    cnn.eval()
    gcn.eval()

    with torch.no_grad():
        latens = cnn(val_x)

        index = faiss.IndexFlatL2(latens.shape[1])
        index.add(latens.cpu().numpy())

        _, I = index.search(latens.cpu().numpy(), config.K_neigh + 1)

        neighbors = torch.tensor(I[:, 1:], device = device)
        edge_index = create_edge_index(neighbors)

        data = Data(x = latens, edge_index = edge_index).to(device)

        out = gcn(data)

        loss = F.cross_entropy(
            out,
            val_y
        )

        #pred = out.argmax(dim=1)

        #acc = (pred == val_y).float().mean().item()
    return loss.item()
    #return acc


def train_full_batch(cnn, gcn, data, train_val_x, opt_cnn, opt_gcn):
    cnn.train()
    gcn.train()

    opt_cnn.zero_grad()
    opt_gcn.zero_grad()

    latens = cnn(train_val_x)
    data.x = latens

    preds = gcn(data)

    loss = F.cross_entropy(
        preds[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    opt_cnn.step()
    opt_gcn.step()

    return loss.item()


def train_mini_batch(cnn, gcn, loader, train_val_x, opt_cnn, opt_gcn):
    cnn.train()
    gcn.train()

    total_loss = 0.0
    count = 0

    for subgraph in loader:
        opt_cnn.zero_grad()
        opt_gcn.zero_grad()

        subimages = train_val_x[subgraph.n_id]
        sublatens = cnn(subimages)

        subgraph.x = sublatens
        preds = gcn(subgraph)

        loss = F.cross_entropy(
            preds[subgraph.train_mask],
            subgraph.y[subgraph.train_mask]
        )

        loss.backward()
        opt_cnn.step()
        opt_gcn.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(1, count)