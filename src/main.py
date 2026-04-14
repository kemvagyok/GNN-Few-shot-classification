# --- Standard library ---
import argparse

# --- Third-party ---
import numpy as np
import torch
import torch.distributed as dist

# --- Project modules ---
from configs import Config

from criterions import FocalLoss, compute_class_weights, effective_num_weights

from metrics import macro_f1

from models import initalizeModels

from preprocessing.factory import build_dataset
from preprocessing.loadingModule import load_dataset_cached

from training.Trainer import Trainer
from training.TrainerDDP import TrainerDDP

from utils import (
    setup_device,
    is_main_process,
    set_seed,
    graph_builder,
    save_results,
    get_class_distribution,
    print_distribution,
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
    
    x, y, num_class, channel_size = \
        load_dataset_cached(
            dataset_name=config.dataset_name,
            data_pth=config.dataset_path,
            img_size=img_size,
            files_size=config.files_size if hasattr(config, "files_size") else 4000,
            force_reload=False
        )
    
    train_x, train_y, test_x, test_y = split_dataset(x,y, test_size=0.2)
    
    train_dataset = build_dataset(
        datasetType=config.dataset_type,
        train_x=train_x,
        train_y=train_y,
        val_x=test_x,
        val_y=test_y,
        num_class=num_class,
        device=device
    )

    test_dataset = build_dataset(
            datasetType=config.dataset_type,
            train_x=test_x,
            train_y=test_y,
            num_class=num_class,
            device=device
        )
    
    print_distribution(get_class_distribution(targets = train_y, num_classes = num_class))

    criterion = build_loss(
        config=config,
        targets=train_y,
        num_classes=num_class,
        device=device
    )
    print(train_dataset.get_train_val_size(), len(test_dataset))
    all_size = sum(train_dataset.get_train_val_size()) + len(test_dataset)
    print(train_dataset.get_train_val_size()[0]/all_size, train_dataset.get_train_val_size()[1]/all_size, len(test_dataset)/all_size)
    metrics = build_metrics(config)
    mode = "fullbatch" if not config.use_minibatch else "minibatch"
    best_results = []
    for max_label in config.train_images_per_class:
        K_hops = [None] if not config.use_minibatch else K_hop_list
        for K_hop in K_hops:
            train_dataset.update_train_mask(max_label)
            with wandb_run(config, is_ddp, run_id, K_hop, max_label, mode):
                run_accs = []
                set_seed(42 + run_id)
                embedder, gnn = initalizeModels(
                    config = config, 
                    channel_size = channel_size, 
                    num_class = num_class, 
                    device = device, 
                    is_ddp = is_ddp)
                """
                trainer = Trainer(
                    embedder=embedder, 
                    graph_builder=graph_builder, 
                    gnn=gnn, criterion=criterion, 
                    metric_fn=metrics,
                    config=config, 
                    device=device
                    )
                """
                trainer = TrainerDDP(
                    embedder=embedder, 
                    graph_builder=graph_builder, 
                    gnn=gnn, criterion=criterion, 
                    metric_fn=metrics,
                    config=config, 
                    device=device,
                    local_rank=local_rank,
                    is_ddp=is_ddp
                    )
                acc = trainer.train(
                    train_dataset=train_dataset, 
                    val_dataset=(test_dataset if test_dataset is not None else None), 
                    K_hop = K_hop
                    )
                run_accs.append(acc)
                acc = np.mean(run_accs)

            best_results.append((K_hop, max_label, acc))

    if is_main_process():
        save_results(best_results, config, run_id)
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()