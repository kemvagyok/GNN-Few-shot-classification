from dataclasses import dataclass
import torch

@dataclass
class RunContext:
    config: any
    device: torch.device
    K_hop: int | None
    max_label: int
    mode: str

    train_dataset: any
    val_dataset: any

    num_class: int
    channel_size: int