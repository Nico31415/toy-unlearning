"""Evaluation: accuracies, losses and a loss-threshold membership inference attack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.cifar_unlearning.losses import pair_targets, restrict_to_pair


@torch.no_grad()
def predict_logits(model: nn.Module, x: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
    model.eval()
    if len(x) == 0:
        return torch.empty(0, 0, device=x.device)
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def two_way_accuracy(logits: torch.Tensor, y: torch.Tensor, pair: Sequence[int]) -> float:
    """Accuracy when the model only chooses between the two forget classes (chance 50%)."""
    preds = restrict_to_pair(logits, pair).argmax(dim=1)
    return (preds == pair_targets(y, pair)).float().mean().item()


def per_sample_ce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y, reduction="none")


def mia_auc(member_losses: torch.Tensor, nonmember_losses: torch.Tensor, chunk: int = 2048) -> float:
    """AUC of the attack that calls an image a member when its loss is low.

    Equals P(member loss < non-member loss) + 0.5 P(equal). 0.5 means the attack
    cannot tell removed training images from unseen ones.
    """
    m = member_losses.flatten().double()
    n = nonmember_losses.flatten().double()
    if len(m) == 0 or len(n) == 0:
        return float("nan")
    wins = 0.0
    for i in range(0, len(m), chunk):
        block = m[i:i + chunk, None]
        wins += (block < n[None, :]).sum().item() + 0.5 * (block == n[None, :]).sum().item()
    return wins / (len(m) * len(n))


@dataclass
class EvalSets:
    """Normalised evaluation tensors on the device, built once per process."""
    x_test_retain: torch.Tensor
    y_test_retain: torch.Tensor
    x_test_forget: torch.Tensor
    y_test_forget: torch.Tensor
    x_probe: torch.Tensor
    y_probe: torch.Tensor


def evaluate(model: nn.Module, ev: EvalSets, pair: Optional[Sequence[int]]) -> Dict[str, float]:
    """All post-stage measurements. Keys:

    A_r            10-way accuracy on T_r (whole test set in the random split)
    A_f10, A_f2    10-way and 2-way accuracy on T_f (class split only)
    ce_f           10-way CE on T_f (class split only)
    probe_acc      10-way accuracy on P_f, the removed images never used
    probe_acc2     2-way accuracy on P_f (class split only)
    probe_ce       10-way CE on P_f
    mia_auc        loss-threshold attack, P_f against test images of the same classes
    """
    out: Dict[str, float] = {}
    logits_r = predict_logits(model, ev.x_test_retain)
    out["A_r"] = accuracy(logits_r, ev.y_test_retain)

    logits_p = predict_logits(model, ev.x_probe)
    losses_p = per_sample_ce(logits_p, ev.y_probe)
    out["probe_acc"] = accuracy(logits_p, ev.y_probe)
    out["probe_ce"] = losses_p.mean().item()

    if pair is not None:
        logits_f = predict_logits(model, ev.x_test_forget)
        out["A_f10"] = accuracy(logits_f, ev.y_test_forget)
        out["A_f2"] = two_way_accuracy(logits_f, ev.y_test_forget, pair)
        out["ce_f"] = per_sample_ce(logits_f, ev.y_test_forget).mean().item()
        out["probe_acc2"] = two_way_accuracy(logits_p, ev.y_probe, pair)
        nonmember_losses = per_sample_ce(logits_f, ev.y_test_forget)
    else:
        # In the random split the non-members are the whole test set, which is T_r.
        nonmember_losses = per_sample_ce(logits_r, ev.y_test_retain)
    out["mia_auc"] = mia_auc(losses_p, nonmember_losses)
    return out


@torch.no_grad()
def confusion_matrix(model: nn.Module, x: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    preds = predict_logits(model, x).argmax(dim=1)
    idx = y * num_classes + preds
    return torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes).cpu()
