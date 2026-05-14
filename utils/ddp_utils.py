# =========================
# Standard library
# =========================
import os

# =========================
# PyTorch core
# =========================
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

def is_main_process():
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def reduce_value(value, device, average = True):
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        return tensor.item() / dist.get_world_size()
    return tensor.item()