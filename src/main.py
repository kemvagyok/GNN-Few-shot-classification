# --- Standard library ---
import argparse

# --- Third-party ---
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

# --- Project modules ---
from configs import Config

from criterions import FocalLoss, compute_class_weights, effective_num_weights

from metrics import macro_f1

from models import initalizeModels

from preprocessing.factory import build_dataset
from preprocessing.loadingModule import load_dataset_cached

from training import TrainerEmbeddingOnly, TrainerEmbeddingOnlyDDP, Trainer, TrainerDDP

from utils import (
    setup_device,
    is_main_process,
    set_seed,
    graph_builder,
    save_results,
    get_class_distribution,
    print_distribution,
    sample_k_per_class,
    split_dataset,
    wandb_run
)

from utils.loss_factory import build_loss
from utils.metric_factory import build_metrics


def main():
    parsers = argparse.ArgumentParser()
    parsers.add_argument("-config_fn", type=str, default="mnist", help="Choosing a config name")
    parsers.add_argument("--run_id", type=int, default=0, help="ID for the current run (used for logging)")
    args = parsers.parse_args()
    config_filename = args.config_fn
    run_id = args.run_id
    
    config = Config(f"./configs/{config_filename}.yaml")
    K_hop_list = config.K_hop_list
    img_size = config.img_size

    device, local_rank, is_ddp = setup_device()
    print(device, local_rank, is_ddp)
    #--------------------------
    x, y, num_class, channel_size = \
        load_dataset_cached(
            dataset_name=config.dataset_name,
            data_pth=config.dataset_path,
            img_size=img_size,
            files_size=config.files_size if hasattr(config, "files_size") else 4000,
            force_reload=False
        )
    print_distribution(get_class_distribution(y, num_class))
    if config.dataset_name == "ISIC2019":
        num_class -=1
    train_x, train_y, test_x, test_y = split_dataset(x,y, test_size=0.2)
    train_x, train_y, val_x, val_y = split_dataset(train_x,train_y, test_size=0.1)

    train_size = len(train_y)
    val_size = len(val_y)
    test_size = len(test_y)
    print(train_size, val_size, test_size)
    all_size = train_size + val_size + test_size
    print(train_size / all_size, val_size / all_size, test_size / all_size)
    #--------------------------
    val_dataset = build_dataset(
            datasetType=config.dataset_type,
            x=val_x,
            y=val_y,
            num_class=num_class,
            device=device
    )
    
    test_dataset = build_dataset(
            datasetType=config.dataset_type,
            x=test_x,
            y=test_y,
            num_class=num_class,
            device=device
    )
    #--------------------------
    if config.use_class_weights:
        class_counts = y.bincount() 
        print(class_counts)
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum()
    else:
        weights = None

    criterion = build_loss(
        config=config,
        targets=train_y,
        num_classes=num_class,
        device=device,
        weights=weights
    )
    #--------------------------
    metrics = build_metrics(config)

    mode = "fullbatch" if not config.use_minibatch else "minibatch"


    best_results = []

    for max_label in config.train_images_per_class:
        K_hops = [None] if not config.use_minibatch else K_hop_list
        K_hops = [None] if config.train_mode == "embedding_only" else K_hop_list

        if max_label == -1:
            max_label = len(train_y)
            filtered_train_x, filtered_train_y = train_x, train_y
        else:
            filtered_train_x, filtered_train_y = sample_k_per_class(train_x, train_y, num_class, max_label)

        train_dataset = build_dataset(
            datasetType=config.dataset_type,
            x=filtered_train_x,
            y=filtered_train_y,
            num_class=num_class,
            device=device)

        for K_hop in K_hops:
            with wandb_run(config, is_ddp, run_id, K_hop, max_label, mode, train_size, val_size, test_size):
                run_accs = []
                set_seed(42 + run_id)
                embedder, gnn = initalizeModels(
                    config = config, 
                    channel_size = channel_size, 
                    num_class = num_class,
                    latens_size= num_class,
                    device = device, 
                    is_ddp = is_ddp)
                
                trainer = get_trainer(is_ddp, embedder, graph_builder, gnn, criterion, metrics, config, device, local_rank)
                    
                acc = trainer.train(
                    train_dataset=train_dataset, 
                    val_dataset=(val_dataset if val_dataset is not None else None), 
                    test_dataset=(test_dataset if test_dataset is not None else None),
                    K_hop = K_hop
                    )
                
                run_accs.append(acc)
                acc = np.mean(run_accs)

            best_results.append((K_hop, max_label, acc))

    if is_main_process():
        save_results(best_results, config, run_id)

    if is_ddp:
        dist.destroy_process_group()

def get_trainer(is_ddp, embedder, graph_builder, gnn, criterion, metrics, config, device, local_rank):
    if is_ddp:
            if config.train_mode == "embedding_only":
                return TrainerEmbeddingOnlyDDP(
                    embedder=embedder, 
                    criterion=criterion, 
                    metric_fn=metrics,
                    config=config, 
                    rank=local_rank,
                    world_size=dist.get_world_size()
                )
            else:
                return TrainerDDP(
                embedder=embedder, 
                graph_builder=graph_builder, 
                gnn=gnn, 
                criterion=criterion, 
                metric_fn=metrics,
                config=config, 
                device=device,
                rank=local_rank,
                world_size=dist.get_world_size()
                )

    if config.train_mode == "embedding_only":
        return TrainerEmbeddingOnly(
            embedder=embedder, 
            criterion=criterion, 
            metric_fn=metrics,
            config=config, 
            device=device
            )
    return Trainer(
        embedder=embedder, 
        graph_builder=graph_builder, 
        gnn=gnn, criterion=criterion, 
        metric_fn=metrics,
        config=config, 
        device=device
        )

if __name__ == "__main__":
    main()

