from contextlib import contextmanager
import wandb

@contextmanager
def wandb_run(config, is_ddp, run_id, K_hop, max_label, mode):
    if is_ddp:
        yield None
        return

    run = wandb.init(
        project=f"few-shot-gnn-{config.dataset_name}_{mode}",
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
        }
    )

    try:
        yield run
    finally:
        wandb.finish()