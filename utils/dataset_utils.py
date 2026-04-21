import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
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


def split_dataset(X, y, test_size=0.2, val_size=0.2, stratify=True):
    indices = torch.arange(len(y))

    strat = y if stratify else None

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=strat
    )
    def select(X, idx):
        if isinstance(X, dict):
            return {k: v[idx] for k, v in X.items()}
        elif isinstance(X, (list, tuple)):
            return [x[idx] for x in X]
        else:
            return X[idx]

    return (
        select(X, train_idx), y[train_idx],
        #select(X, val_idx), y[val_idx],
        select(X, test_idx), y[test_idx],
    )


# K classonként k mintát választunk ki, és visszaadjuk a kiválasztott és a maradék adatokat
def sample_k_per_class_indices(y, num_class, k):
    selected_idx = []

    for c in range(num_class):
        class_idx = np.where(y == c)[0]
        chosen = np.random.choice(class_idx, size=min(k, len(class_idx)), replace=False)
        selected_idx.append(chosen)

    selected_idx = np.concatenate(selected_idx)

    mask = np.zeros(len(y), dtype=bool)
    mask[selected_idx] = True

    return selected_idx, mask

def index_data(x, idx):
    if isinstance(x, dict):
        return {k: v[idx] for k, v in x.items()}
    else:
        return x[idx]
    
def sample_k_per_class(x, y, num_class, k):
    idx, mask = sample_k_per_class_indices(y, num_class, k)

    x_k = index_data(x, idx)
    y_k = y[idx]

    return (x_k, y_k)