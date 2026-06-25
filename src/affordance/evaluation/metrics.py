import torch


def _auc_roc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float("nan"))
    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    tp = torch.cumsum(labels_sorted, dim=0)
    fp = torch.cumsum(1 - labels_sorted, dim=0)
    tpr = torch.cat([torch.zeros(1, device=scores.device), tp / n_pos])
    fpr = torch.cat([torch.zeros(1, device=scores.device), fp / n_neg])
    return torch.trapezoid(tpr, fpr)


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n_pos = labels.sum()
    if n_pos == 0:
        return torch.tensor(float("nan"))
    order = torch.argsort(scores, descending=True)
    labels_sorted = labels[order]
    tp = torch.cumsum(labels_sorted, dim=0)
    n = torch.arange(1, len(labels_sorted) + 1, dtype=scores.dtype, device=scores.device)
    precision = tp / n
    return (precision * labels_sorted).sum() / n_pos


def compute_metrics(probs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """
    probs:   [N] sigmoid predictions for all visible vertices across the split
    targets: [N] soft pseudolabels in [0, 1]
    """
    eps = 1e-7
    t = targets.clamp(eps, 1 - eps)
    p = probs.clamp(eps, 1 - eps)
    kl = (t * (t.log() - p.log()) + (1 - t) * ((1 - t).log() - (1 - p).log())).mean()

    binary = (targets > 0.5).float()
    auc = _auc_roc(probs, binary)
    ap = _average_precision(probs, binary)

    return {"kl": kl.item(), "auc": auc.item(), "ap": ap.item()}
