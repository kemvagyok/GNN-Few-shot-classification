# utils/loss_factory.py

import torch
from criterions import FocalLoss, effective_num_weights

def build_loss(config, targets, num_classes, device):
    if config.criterion == "focal":
        weights = effective_num_weights(
            targets,
            num_classes=num_classes,
            beta=config.loss_beta
        )
        weights = torch.clamp(weights, max=config.loss_weight_clip).to(device)

        return FocalLoss(
            alpha=weights,
            gamma=config.loss_gamma
        )

    elif config.criterion == "cross_entropy":
        return torch.nn.CrossEntropyLoss()

    elif config.criterion == "weighted_ce":
        weights = effective_num_weights(
            targets,
            num_classes=num_classes,
            beta=config.loss_beta
        )
        weights = weights.to(device)

        return torch.nn.CrossEntropyLoss(weight=weights)

    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")