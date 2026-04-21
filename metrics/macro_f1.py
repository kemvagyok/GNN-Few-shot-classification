import torch

def macro_f1(preds, targets, num_classes=8, eps=1e-8):
    if preds.ndim > 1:
        preds = torch.argmax(preds, dim=1)

    preds = preds.view(-1)
    targets = targets.view(-1)

    if num_classes is None:
        num_classes = max(preds.max(), targets.max()) + 1

    conf_mat = torch.zeros((num_classes, num_classes), device=preds.device)

    conf_mat.index_put_(
        (targets, preds),
        torch.ones_like(targets, dtype=torch.float),
        accumulate=True
    )

    tp = torch.diag(conf_mat)
    fp = conf_mat.sum(dim=0) - tp
    fn = conf_mat.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    # 🔥 csak a valid osztályok
    valid = (tp + fn) > 0
    f1 = f1[valid]

    return f1.mean().item()