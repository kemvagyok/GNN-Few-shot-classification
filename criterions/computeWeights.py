import torch

def compute_class_weights(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return weights

def effective_num_weights(y, num_classes, beta=0.9999):
    counts = torch.bincount(y, minlength=num_classes).float()
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * num_classes
    return weights