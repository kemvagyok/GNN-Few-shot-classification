from contextlib import contextmanager
from utils.ddp_utils import is_main_process
import wandb

@contextmanager
def wandb_run(config, is_ddp, run_id, K_hop, max_label):
    if is_ddp and not is_main_process():
        yield None
        return
    
    project_name=f"fewshootgnn_{config.dataset_name}_{'ddp' if is_ddp else 'single'}_{'embedding&GNN' if config.train_mode == 'full' else 'embedding'}_{"freezing" if config.isFreeze else "not_freezing"}"
    
    run = wandb.init(
        settings=wandb.Settings(init_timeout=config.wandb_init_timeout),
        mode=config.wandb_mode,
        project=project_name,
        name=f"run_{run_id}",
        config={
            "dataset": config.dataset_name,
            "K_hop": K_hop,
            "max_label": max_label,
            "criterion": config.criterion,
            "metrics": config.metrics,
            "embedding": config.embedding,
            "gnn_model": config.gnn_model,
            "train_size": config.train_size,
            "val_size": config.val_size,
            "test_size": config.test_size,
            "patience": config.patience,
            "delta": config.delta,
            "isFreeze": config.isFreeze
        }
    )

    try:
        yield run
    finally:
        wandb.finish()

