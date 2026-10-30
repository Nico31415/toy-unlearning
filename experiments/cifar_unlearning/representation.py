"""Representation-level analysis, following "Are We Truly Forgetting?" (Kim, Cha & Kim, 2025).

Output-level metrics can look right while the features are unchanged (only the head
moved). These measurements look at the penultimate features instead:

    CKA      linear centered kernel alignment between this model's features and those
             of Retrain (should be high after real unlearning) and of Pretrained
             (should drop), on test images of the forgotten and the kept classes;
             on the forgotten classes also within-class (see linear_cka)
    k-NN     cosine k-nearest-neighbour accuracy using the features alone: on the
             forgotten classes (support = the probe images P_f, query = T_f), and on
             the kept classes (support = kept training images, query = T_r)

The paper's k-NN transfer to other datasets (Office-Home, CUB) has no natural CIFAR-10
counterpart; k-NN on the forgotten classes tests the same thing, whether the features
still separate them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def penultimate_features(model: nn.Module, x: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    """Inputs to the final linear layer `fc` (512-d for ResNet-18), in eval mode."""
    if len(x) == 0:
        return torch.empty(0, 0, device=x.device)
    feats = []
    handle = model.fc.register_forward_hook(lambda module, inp, out: feats.append(inp[0].detach().flatten(1)))
    try:
        model.eval()
        for i in range(0, len(x), batch_size):
            model(x[i:i + batch_size])
    finally:
        handle.remove()
    return torch.cat(feats)


def _centre(x: torch.Tensor, groups: Optional[torch.Tensor]) -> torch.Tensor:
    x = x.double()
    if groups is None:
        return x - x.mean(dim=0, keepdim=True)
    out = x.clone()
    for g in groups.unique():
        mask = groups == g
        out[mask] -= x[mask].mean(dim=0, keepdim=True)
    return out


def linear_cka(x: torch.Tensor, y: torch.Tensor, groups: Optional[torch.Tensor] = None) -> float:
    """Linear CKA (Kornblith et al., 2019) between feature matrices with matching rows.

    1 means identical up to rotation and isotropic scaling; it ignores feature dimension.
    With `groups` (class labels), features are centred within each class, so the score
    compares within-class structure. On T_f, which holds only two classes, plain CKA is
    dominated by the single direction separating them and is close to 1 for almost any
    network that separates the classes; the within-class version is more informative.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA needs features of the same examples")
    x, y = _centre(x, groups), _centre(y, groups)
    cross = (x.T @ y).pow(2).sum()
    norm = torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)
    if norm == 0:
        return float("nan")
    return (cross / norm).item()


@torch.no_grad()
def knn_accuracy(support: torch.Tensor, support_y: torch.Tensor, query: torch.Tensor, query_y: torch.Tensor,
                 k: int = 20, chunk: int = 1000) -> float:
    """Cosine k-NN accuracy with similarity-weighted votes."""
    if len(query) == 0 or len(support) == 0:
        return float("nan")
    k = min(k, len(support))
    s = F.normalize(support.float(), dim=1)
    q = F.normalize(query.float(), dim=1)
    num_classes = int(max(support_y.max(), query_y.max()).item()) + 1
    correct = 0
    for i in range(0, len(q), chunk):
        sims = q[i:i + chunk] @ s.T
        top, idx = sims.topk(k, dim=1)
        votes = torch.zeros(len(top), num_classes, device=q.device, dtype=top.dtype)
        votes.scatter_add_(1, support_y[idx], top)
        correct += (votes.argmax(dim=1) == query_y[i:i + chunk]).sum().item()
    return correct / len(q)


@dataclass
class RepSets:
    """Normalised images on the device for representation analysis."""
    x_test_forget: torch.Tensor  # T_f (empty in the random split)
    y_test_forget: torch.Tensor
    x_test_retain: torch.Tensor  # T_r (whole test set in the random split)
    y_test_retain: torch.Tensor
    x_support_forget: torch.Tensor  # P_f: removed images, never used by unlearning or relearning
    y_support_forget: torch.Tensor
    x_support_retain: torch.Tensor  # a stratified subset of the kept training images
    y_support_retain: torch.Tensor


def reference_features(model: nn.Module, rs: RepSets) -> Dict[str, torch.Tensor]:
    return {"f": penultimate_features(model, rs.x_test_forget), "r": penultimate_features(model, rs.x_test_retain)}


def representation_metrics(model: nn.Module, rs: RepSets, refs: Dict[str, Dict[str, torch.Tensor]],
                           pair: Optional[Sequence[int]], k: int = 20) -> Dict[str, float]:
    """CKA to each reference model (keys of `refs`, e.g. retrain, pretrained) on T_f and
    T_r, plus k-NN accuracy on the forgotten classes (class split) and the kept classes."""
    feats = reference_features(model, rs)
    out: Dict[str, float] = {}
    for name, ref in refs.items():
        out[f"cka_{name}_r"] = linear_cka(feats["r"], ref["r"])
        if pair is not None:
            out[f"cka_{name}_f"] = linear_cka(feats["f"], ref["f"])
            out[f"cka_{name}_f_within"] = linear_cka(feats["f"], ref["f"], groups=rs.y_test_forget)
    support_r = penultimate_features(model, rs.x_support_retain)
    out["knn_r"] = knn_accuracy(support_r, rs.y_support_retain, feats["r"], rs.y_test_retain, k)
    if pair is not None:
        support_f = penultimate_features(model, rs.x_support_forget)
        out["knn_f"] = knn_accuracy(support_f, rs.y_support_forget, feats["f"], rs.y_test_forget, k)
    return out
