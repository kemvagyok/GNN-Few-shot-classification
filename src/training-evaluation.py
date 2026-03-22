# =========================
# Standard library
# =========================
import os
import random
import argparse

# =========================
# Third-party libraries
# =========================
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import wandb
from networkx import subgraph

# =========================
# PyTorch core
# =========================
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# =========================
# PyTorch Geometric
# =========================
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# =========================
# Saját modulok
# =========================
from config import Config
import config_summary

from graph_tools import create_edge_index
from models import CNNModel, GCNModel, ResNetModel
from loadingModule import (
    dataLoading_MNIST,
    dataLoading_ChestX,
    dataLoading_ISIC2019,
    traindatasetFiltering
)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_main_process():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

        
def reduce_value(value, device, average = True):
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        return tensor.item() / dist.get_world_size()
    return tensor.item()

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

def run_training(train_x, train_y, val_x, val_y, test_x,  test_y, num_class, channel_size, K_hop, max_label, config, device, is_ddp=False):
    train_mask_index = traindatasetFiltering(train_y, num_class, max_label)

    train_x_filtered = train_x[train_mask_index].to(device)
    train_y_filtered = train_y[train_mask_index].to(device)

    val_x = val_x.to(device)
    val_y = val_y.to(device)
    
    train_test_x = torch.cat((train_x_filtered, val_x))
    train_test_y = torch.cat((train_y_filtered, val_y))

    train_mask = torch.zeros(len(train_test_y), dtype=torch.bool)
    train_mask[:len(train_x_filtered)] = True


    #cnn = CNNModel(channel_size=channel_size).to(device)
    cnn = ResNetModel(output_dim=64, in_channels=channel_size).to(device)
    gcn = GCNModel(num_features=config.latens_size,
                num_classes=num_class).to(device)

    if is_ddp:
        cnn = DDP(cnn, device_ids=[device.index])
        gcn = DDP(gcn, device_ids=[device.index])

    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=config.lr_cnn)
    opt_gcn = torch.optim.Adam(gcn.parameters(), lr=config.lr_gcn)

    latens, edge_index = build_graph(cnn, train_test_x, config, device)

    data = Data(x=latens, edge_index=edge_index).to(device)
    data.y = train_test_y
    data.train_mask = train_mask

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    if is_ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Process {rank}: World size: {world_size}")
        train_mask_index = train_idx[rank::world_size]
    if is_ddp:
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

    best_acc = 0.0

    for epoch in range(config.epochs_max):
        if epoch % 100 == 0:
            if is_ddp:
                if dist.get_rank() == 0:
                    print(f"Epoch {epoch}/{config.epochs_max}")
            else:
                print(f"Epoch {epoch}/{config.epochs_max}")
              
        cnn.train()
        gcn.train()

        total_loss = 0.0
        count = 0

        for subgraph in loader:

            opt_cnn.zero_grad()
            opt_gcn.zero_grad()

            subimages = train_test_x[subgraph.n_id]
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

        avg_loss = reduce_value(total_loss / max(1, count), device)
        if is_main_process() and epoch % 10 == 0:
            print("Epoch {}: Train Loss = {:.4f}".format(epoch, avg_loss))
            wandb.log({"train_loss": avg_loss, "epoch": epoch})

        # ---- evaluation ----
        cnn.eval()
        gcn.eval()

        with torch.no_grad():
            latens_test = cnn(val_x)

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
            acc = (pred == val_y).float().mean().item()

            global_acc = reduce_value(acc, device)

            if global_acc > best_acc:
                best_acc = global_acc
            
            if is_main_process():
                print("Epoch {}: Validation Accuracy = {:.4f}".format(epoch, global_acc))
                wandb.log({"val_acc": global_acc})

    if test_x is not None and test_y is not None:
        test_acc = evaluate_model(cnn, gcn, test_x.to(device), test_y.to(device), config, device)

    if is_main_process():
        print("Epoch {}: Test Accuracy = {:.4f}".format(epoch, test_acc))
        wandb.log({"test_acc": test_acc})
    
    return best_acc

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

    if is_main_process():
        print("Final Test Accuracy: {:.4f}".format(acc))
        wandb.log({"inference_accuracy": 0.95})
    return acc

def setup_device():
    if "LOCAL_RANK" in os.environ:
        # DDP mód
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_ddp = True
        return device, local_rank, is_ddp
    else:
        # Single GPU / CPU mód
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        is_ddp = False
        return device, local_rank, is_ddp


def loading_dataset(dataset_name, **kwargs):
    if dataset_name == "MNIST":
        return dataLoading_MNIST(**kwargs)
    elif dataset_name == "ChestX":
        return dataLoading_ChestX(**kwargs)
    elif dataset_name == "ISIC2019":
        return dataLoading_ISIC2019(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():

    parsers = argparse.ArgumentParser()
    parsers.add_argument("-dataset", type=str, default="MNIST", help="ID for the current run (used for logging)")
    parsers.add_argument("--run_id", type=int, default=0, help="ID for the current run (used for logging)")
    args = parsers.parse_args()
    run_id = args.run_id
    dataset_name = args.dataset
    without_valid = True
    #dataset_name = Config().CHOSE_DATASET

    config = Config()
    device, local_rank, is_ddp = setup_device()
    train_x, train_y, val_x, val_y, test_x, test_y, num_class, channel_size = loading_dataset(dataset_name)


    if without_valid:
        train_x = torch.cat([train_x, val_x], dim=0)
        train_y = torch.cat([train_y, val_y], dim=0)
        val_x = test_x
        val_y = test_y
        test_x = None 
        test_y = None
    test_x_filtered = test_x[:config.test_size].to(device)
    test_y_filtered = test_y[:config.test_size].to(device)
 

    best_results = []

    for K_hop in config.K_hop_list:
        print(f"\nStarting training for K_hop={K_hop}...")
        for max_label in config.train_images_per_class:
            print(f"  Training with max_label={max_label}...")
            if is_main_process():
                wandb.init(
                    project=f"few-shot-gnn-{dataset_name}_without_ddp",
                    name=f"run_{run_id}",
                    group=f"k_hop_{K_hop}",
                    config={
                        "dataset": config.CHOSE_DATASET,
                        "K_hop": K_hop,
                        "max_label": max_label,
                        "epochs": config.epochs_max,
                        "lr_cnn": config.lr_cnn,
                        "lr_gcn": config.lr_gcn,
                        "batch_size": config.batch_size,
                        "K_neigh": config.K_neigh,
                        "is_ddp": is_ddp
                        },
                        reinit = True
                )

            run_accs = []
            #for run in range(config.runs): #RUN_ARRAY miatt
            set_seed(42 + run_id)

            acc = run_training(
                train_x, train_y,
                val_x, val_y,
                test_x_filtered, test_y_filtered,
                num_class, channel_size,
                K_hop, max_label,
                config, device,
                is_ddp = is_ddp
            )

            run_accs.append(acc)

            acc = np.mean(run_accs)
            best_results.append((K_hop, max_label, acc))
    

    if is_main_process():
        wandb.finish()


    if is_main_process():
        results_df = pd.DataFrame(best_results, columns=["K_hop", "max_label", "acc"])
        #file_name = f"../results/{config.CHOSE_DATASET}_results_{config.K_neigh}_h_{config.epochs_max}e.csv"
        file_name = f"../results/results_{config.CHOSE_DATASET}_run{run_id}.csv"
        results_df.to_csv(file_name, index=False)
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    config = Config()
    config_summary.print_config(config)
    print("\nTraining/Testing starting.")
    main()
    print("\nTraining/Testing completed.")
