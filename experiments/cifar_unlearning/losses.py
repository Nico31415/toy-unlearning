"""Losses for every stage.

The default is cross-entropy. `kind="mse"` is the fallback noted in the plan:
squared loss on logits against one-hot targets, and for unlearning the effective-target
form of the workshop paper (Lemma 1), which is bounded and therefore converges.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

LOSS_KINDS = ("ce", "mse")


def restrict_to_pair(logits: torch.Tensor, pair: Sequence[int]) -> torch.Tensor:
    """Keep only the two forget-class logits, in the order of `pair`."""
    return logits[:, list(pair)]


def pair_targets(y: torch.Tensor, pair: Sequence[int]) -> torch.Tensor:
    """Map labels pair[0] -> 0 and pair[1] -> 1."""
    return (y == pair[1]).long()


def _squared(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """0.5 * squared error summed over classes, averaged over the batch."""
    return 0.5 * (logits - targets).pow(2).sum(dim=1).mean()


def classification_loss(logits: torch.Tensor, y: torch.Tensor, kind: str = "ce",
                        pair: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Ordinary training loss. With `pair`, a 2-way loss on the two forget logits only."""
    if pair is not None:
        logits, y = restrict_to_pair(logits, pair), pair_targets(y, pair)
    if kind == "ce":
        return F.cross_entropy(logits, y)
    if kind == "mse":
        return _squared(logits, F.one_hot(y, logits.shape[1]).float())
    raise ValueError(f"unknown loss kind {kind!r}")


def signed_unlearning_loss(logits_r: torch.Tensor, y_r: torch.Tensor,
                           logits_f: torch.Tensor, y_f: torch.Tensor,
                           mu_r: float, mu_f: float, kind: str = "ce") -> torch.Tensor:
    """Unlearning loss on a retain batch and a forget batch.

    ce:  mu_r * CE(retain) - mu_f * CE(forget). Unbounded below; the training loop's
         stop rule halts it.
    mse: the workshop paper's effective-target form, needing mu_r > mu_f. With
         rho = mu_f / (mu_r - mu_f), retain images regress to (1 + rho) * onehot and
         forget images to -rho * onehot, weighted by A = mu_r - mu_f.
    """
    if kind == "ce":
        return mu_r * F.cross_entropy(logits_r, y_r) - mu_f * F.cross_entropy(logits_f, y_f)
    if kind == "mse":
        if not mu_r > mu_f:
            raise ValueError("the MSE effective-target loss needs mu_r > mu_f")
        rho = mu_f / (mu_r - mu_f)
        k = logits_r.shape[1]
        target_r = (1.0 + rho) * F.one_hot(y_r, k).float()
        target_f = -rho * F.one_hot(y_f, k).float()
        return (mu_r - mu_f) * (_squared(logits_r, target_r) + _squared(logits_f, target_f))
    raise ValueError(f"unknown loss kind {kind!r}")


def unlearning_retain_loss(logits_r: torch.Tensor, y_r: torch.Tensor, mu_r: float, mu_f: float,
                           kind: str = "ce") -> torch.Tensor:
    """Retain loss used by the unlearning stop rule: CE for "ce"; for "mse", the
    squared error to the effective retain target (1 + rho) * onehot."""
    if kind == "ce":
        return F.cross_entropy(logits_r, y_r)
    if kind == "mse":
        rho = mu_f / (mu_r - mu_f)
        return _squared(logits_r, (1.0 + rho) * F.one_hot(y_r, logits_r.shape[1]).float())
    raise ValueError(f"unknown loss kind {kind!r}")
