"""Layer-wise representation read-outs for the phase experiments (E8 of the CIFAR plan).

For every tap (see `models.feature_taps`), on held-out images:

    fresh probe    logistic regression, forget class(es) vs the rest, on standardised features. Standardising makes
                   it blind to the scale of the features: the CIFAR version of the toy's "has B up to scale".
    frozen probe   the probe fitted on A(D)'s features, applied to this model's features (standardised with A(D)'s
                   statistics) after refitting only a scalar gain and a bias. High accuracy means B is still encoded
                   in the same directions (the toy's "support and signs kept").
    amplitude      |mean projection of the forget images minus that of the retain images| on A(D)'s probe direction,
                   in A(D)'s standardised units: how strongly the forget/retain split is expressed.
    CKA            linear CKA to A(D) and to A(D minus S_full) on the forget test images.
    PR, ENSD       participation ratio of the forget-image representation, and the effective number of dimensions
                   it shares with A(D)'s (Giaffar et al. 2024), as in Anguita et al. (2026).

Residual decodability rho = (acc(U) - acc(A(D minus S_full))) / (acc(A(D)) - acc(A(D minus S_full))) is formed
afterwards (aggregate_phase.py), because it needs the references' probe accuracies.

Every forward goes through a `Forward`: a plain network (hooks), or the linearised fine-tune f(theta0) + J delta, whose
features are linearised the same way (one jvp returns logits and every tap).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.cifar_unlearning.models import feature_taps
from experiments.cifar_unlearning.representation import linear_cka

Features = Dict[str, torch.Tensor]


def _pool(t: torch.Tensor) -> torch.Tensor:
    return t.mean(dim=(2, 3)) if t.dim() == 4 else t.flatten(1)


class Forward:
    """Callable x -> (logits, {tap: pooled features}). Subclasses define `_run` on one batch."""

    def __call__(self, x: torch.Tensor, batch_size: int = 500) -> Tuple[torch.Tensor, Features]:
        logits, feats = [], {}
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                lo, fe = self._run(x[i:i + batch_size])
                logits.append(lo.detach())
                for k, v in fe.items():
                    feats.setdefault(k, []).append(v.detach())
        return torch.cat(logits), {k: torch.cat(v) for k, v in feats.items()}

    def _run(self, x):  # pragma: no cover - abstract
        raise NotImplementedError


class ModelForward(Forward):
    def __init__(self, model: nn.Module):
        self.model = model
        self.taps = feature_taps(model)

    def _run(self, x):
        store: Features = {}
        handles = [m.register_forward_hook(lambda mod, inp, out, n=n: store.__setitem__(n, _pool(out)))
                   for n, m in self.taps.items()]
        try:
            self.model.eval()
            logits = self.model(x)
        finally:
            for h in handles:
                h.remove()
        return logits, store


def linear_function(model: nn.Module, theta0: Dict[str, torch.Tensor]):
    """f(delta, x) -> (logits, features) of the linearisation of `model` (eval mode) around theta0:
    f(theta0) + J delta for the logits and for every tap. Differentiable in delta (reverse over forward AD)."""
    from torch.func import functional_call, jvp

    taps = feature_taps(model)
    names = list(taps)
    buffers = {k: v for k, v in model.named_buffers()}

    def run(delta: Dict[str, torch.Tensor], x: torch.Tensor):
        store: Features = {}

        def f(params):
            handles = [m.register_forward_hook(lambda mod, inp, out, n=n: store.__setitem__(n, _pool(out)))
                       for n, m in taps.items()]
            try:
                logits = functional_call(model, {**params, **buffers}, (x,))
            finally:
                for h in handles:
                    h.remove()
            return (logits,) + tuple(store[n] for n in names)

        model.eval()
        primal, tangent = jvp(f, (theta0,), (delta,))
        outs = [p + t for p, t in zip(primal, tangent)]
        return outs[0], dict(zip(names, outs[1:]))

    return run


class LinearForward(Forward):
    def __init__(self, model: nn.Module, theta0: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]):
        self.fn = linear_function(model, theta0)
        self.delta = delta

    def _run(self, x):
        return self.fn(self.delta, x)


# --------------------------------------------------------------------------- probes

@dataclass
class Probe:
    mean: torch.Tensor
    std: torch.Tensor
    w: torch.Tensor
    b: torch.Tensor

    def score(self, feats: torch.Tensor) -> torch.Tensor:
        return ((feats - self.mean) / self.std) @ self.w + self.b


def _fit_logistic(z: torch.Tensor, y: torch.Tensor, l2: float, iters: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """Binary logistic regression with class-balanced weights and an L2 penalty on w (LBFGS)."""
    z, yf = z.double(), y.double()
    pos = yf.mean().clamp(1e-6, 1 - 1e-6)
    weight = torch.where(yf > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    w = torch.zeros(z.shape[1], dtype=torch.float64, device=z.device, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, device=z.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (weight * F.binary_cross_entropy_with_logits(z @ w + b, yf, reduction="none")).mean() \
            + 0.5 * l2 * w.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return w.detach().float(), b.detach().float()


def balanced_accuracy(scores: torch.Tensor, y: torch.Tensor) -> float:
    pred = scores > 0
    pos, neg = y > 0.5, y < 0.5
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return 0.5 * ((pred[pos]).float().mean() + (~pred[neg]).float().mean()).item()


L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def fit_probe(feats: torch.Tensor, y: torch.Tensor, seed: int = 0) -> Probe:
    """Standardise, choose the L2 strength on a 20% hold-out of the training rows, refit on all of them."""
    mean, std = feats.mean(0), feats.std(0).clamp_min(1e-8)
    z = (feats - mean) / std
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=gen).to(z.device)
    cut = max(1, len(z) // 5)
    hold, fit = perm[:cut], perm[cut:]
    best = max(L2_GRID, key=lambda l2: balanced_accuracy(
        (lambda wb: z[hold] @ wb[0] + wb[1])(_fit_logistic(z[fit], y[fit], l2)), y[hold]))
    w, b = _fit_logistic(z, y, best)
    return Probe(mean, std, w, b)


def refit_gain_bias(raw: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Fit logit = a * raw + c (two scalars) by balanced logistic regression."""
    a, c = _fit_logistic(raw[:, None], y, l2=0.0, iters=100)
    return a.item(), c.item()


# --------------------------------------------------------------------------- geometry

def _gram(x: torch.Tensor) -> torch.Tensor:
    x = x.double() - x.double().mean(0, keepdim=True)
    return x @ x.T


def participation_ratio(x: torch.Tensor) -> float:
    k = _gram(x)
    return (torch.trace(k) ** 2 / (k * k).sum()).item()


def ensd(x: torch.Tensor, y: torch.Tensor) -> float:
    """Effective number of shared dimensions (Giaffar et al. 2024): tr(Kx) tr(Ky) tr(Kx Ky) / (tr(Kx^2) tr(Ky^2)),
    with Gram matrices of the same images; ENSD(x, x) = PR(x)."""
    kx, ky = _gram(x), _gram(y)
    return (torch.trace(kx) * torch.trace(ky) * (kx * ky).sum() / ((kx * kx).sum() * (ky * ky).sum())).item()


# --------------------------------------------------------------------------- read-out of one model

@dataclass
class ProbeData:
    """Held-out images for the read-outs. train: probe fitting (removed images never trained on, plus kept-class
    test images); eval: forget-class test images plus other kept-class test images. y = 1 for the forget class(es).
    x_forget: the forget-class test images used for CKA, PR and ENSD."""
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_eval: torch.Tensor
    y_eval: torch.Tensor
    x_forget: torch.Tensor
    labels_train: Optional[torch.Tensor] = None  # class labels of x_train (for the output bias refit)


@dataclass
class RefFeatures:
    """What a reference contributes: its features on every probe set and, for A(D), its fresh probes."""
    train: Features
    eval: Features
    forget: Features
    probes: Optional[Dict[str, Probe]] = None


def reference_features(fwd: Forward, pd: ProbeData, with_probes: bool) -> RefFeatures:
    _, tr = fwd(pd.x_train)
    _, ev = fwd(pd.x_eval)
    _, fo = fwd(pd.x_forget)
    probes = {k: fit_probe(tr[k], pd.y_train) for k in tr} if with_probes else None
    return RefFeatures(tr, ev, fo, probes)


def readout(fwd: Forward, pd: ProbeData, a_d: RefFeatures, a_0: RefFeatures,
            mine: Optional[RefFeatures] = None) -> Dict[str, float]:
    """Every per-layer measurement of one model. `a_d` must carry its probes (A(D)); `a_0` is A(D minus S_full)."""
    me = mine or reference_features(fwd, pd, with_probes=False)
    out: Dict[str, float] = {}
    for k in me.train:
        fresh = fit_probe(me.train[k], pd.y_train)
        out[f"probe_{k}"] = balanced_accuracy(fresh.score(me.eval[k]), pd.y_eval)
        frozen = a_d.probes[k]
        raw_tr, raw_ev = frozen.score(me.train[k]) - frozen.b, frozen.score(me.eval[k]) - frozen.b
        gain, bias = refit_gain_bias(raw_tr, pd.y_train)
        out[f"frozen_{k}"] = balanced_accuracy(gain * raw_ev + bias, pd.y_eval)
        out[f"frozen_raw_{k}"] = balanced_accuracy(frozen.score(me.eval[k]), pd.y_eval)  # no refit at all
        pos, neg = pd.y_eval > 0.5, pd.y_eval < 0.5
        out[f"amp_{k}"] = (raw_ev[pos].mean() - raw_ev[neg].mean()).item()
        out[f"cka_ad_{k}"] = linear_cka(me.forget[k], a_d.forget[k])
        out[f"cka_a0_{k}"] = linear_cka(me.forget[k], a_0.forget[k])
        out[f"pr_{k}"] = participation_ratio(me.forget[k])
        out[f"ensd_ad_{k}"] = ensd(me.forget[k], a_d.forget[k])
    return out
