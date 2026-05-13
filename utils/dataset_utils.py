import numpy as np
import torch
from collections import Counter
import torch

def get_class_distribution(targets, num_classes=None):
    """
    targets: torch.Tensor vagy numpy array (N,)
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    counter = Counter(targets)

    if num_classes is not None:
        return {i: counter.get(i, 0) for i in range(num_classes)}

    return dict(counter)


def print_distribution(dist, title="Distribution"):
    print(f"\n--- {title} ---")
    total = sum(dist.values()) if isinstance(dist, dict) else dist.sum()

    if isinstance(dist, dict):
        for k, v in dist.items():
            print(f"Class {k}: {v} ({v/total:.2%})")
    else:
        for i, v in enumerate(dist):
            print(f"Class {i}: {v} ({v/total:.2%})")


def build_indices(labels, max_per_class=None, seed=42):
    labels = np.array(labels)
    indices = np.arange(len(labels))

    rng = np.random.default_rng(seed)

    selected = []

    for cls in np.unique(labels):
        cls_idx = indices[labels == cls]
        rng.shuffle(cls_idx)

        if max_per_class is None:
            selected.extend(cls_idx)
        else:
            selected.extend(cls_idx[:max_per_class])

    return np.array(selected)