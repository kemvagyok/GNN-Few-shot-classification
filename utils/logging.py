from contextlib import contextmanager
from utils.ddp_utils import is_main_process
import wandb

@contextmanager
def wandb_run(config, is_ddp, run_id, K_hop, max_label, mode, train_size, val_size, test_size):
    if is_ddp and not is_main_process():
        yield None
        return

    run = wandb.init(
        project=f"few-shot-gnn-{config.dataset_name}_{config.train_mode}_{'ddp' if is_ddp else 'single'}_proba",
        name=f"run_{run_id}",
        group="fullbatch" if K_hop is None else f"k_hop_{K_hop}",
        config={
            "dataset": config.dataset_name,
            "K_hop": K_hop,
            "max_label": max_label,
            "criterion": config.criterion,
            "metrics": config.metrics,
            "embedding": config.embedding,
            "gnn_model": config.gnn_model,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "patience": config.patience,
            "delta": config.delta
        }
    )

    try:
        yield run
    finally:
        wandb.finish()