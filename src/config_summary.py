# config_summary.py

import torch
import platform
from datetime import datetime


def print_config(config, model_cnn=None, model_gcn=None):
    print("=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)

    # Időbélyeg
    print(f"Start time: {datetime.now()}")
    print("-" * 60)

    # Rendszer info
    print("SYSTEM INFO:")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA available: YES")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA available: NO")

    print("-" * 60)

    # Hyperparaméterek
    print("HYPERPARAMETERS:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")

    print("-" * 60)

    # Model paraméterszám
    if model_cnn is not None:
        cnn_params = sum(p.numel() for p in model_cnn.parameters())
        print(f"CNN parameters: {cnn_params:,}")

    if model_gcn is not None:
        gcn_params = sum(p.numel() for p in model_gcn.parameters())
        print(f"GCN parameters: {gcn_params:,}")

    print("=" * 60)
    print()