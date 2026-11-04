"""Training loops: from scratch, unlearning with its stop rule, relearning to convergence.

Every loop works on normalised tensors that already sit on the device, and every
stopping decision uses training data only.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from experiments.cifar_unlearning.config import RelearnConfig, ScratchConfig, UnlearnConfig
from experiments.cifar_unlearning.losses import (
    _squared, classification_loss, signed_unlearning_loss, unlearning_forget_loss, unlearning_retain_loss)
from experiments.cifar_unlearning.metrics import accuracy, predict_logits
from experiments.cifar_unlearning.models import set_train_mode

# Relearning curves are recorded at these steps (and at the final step).
CURVE_STEPS = (0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)

# Data augmentation of the training minibatches (random crop with 4 px zero padding, horizontal flip).
# Off by default. Switched on for a whole run by `run.py --augment`. Evaluation and the stop-rule
# checks always use the un-augmented images.
AUGMENT = False


def set_augment(on: bool) -> None:
    global AUGMENT
    AUGMENT = bool(on)


def augment_batch(xb: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random crop (zero padding) and horizontal flip, independently for every image of xb (B, C, H, W)."""
    if not AUGMENT:
        return xb
    b, c, h, w = xb.shape
    dev = xb.device
    padded = torch.nn.functional.pad(xb, (pad, pad, pad, pad))
    dy = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    dx = torch.randint(0, 2 * pad + 1, (b,), device=dev)
    flip = torch.rand(b, device=dev) < 0.5
    ar_h, ar_w = torch.arange(h, device=dev), torch.arange(w, device=dev)
    rows = dy[:, None] + ar_h[None, :]
    cols = dx[:, None] + torch.where(flip[:, None], w - 1 - ar_w[None, :], ar_w[None, :])
    return padded[torch.arange(b, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None],
                  rows[:, None, :, None], cols[:, None, None, :]]


def _index_stream(n: int, batch_size: int, generator: torch.Generator,
                  device: torch.device) -> Iterator[torch.Tensor]:
    """Endless minibatch indices: reshuffle every pass, every batch full-size (sampling
    with wrap-around when n is not a multiple of the batch size)."""
    batch_size = min(batch_size, n)
    buffer = torch.empty(0, dtype=torch.int64)
    while True:
        while len(buffer) < batch_size:
            buffer = torch.cat([buffer, torch.randperm(n, generator=generator)])
        batch, buffer = buffer[:batch_size], buffer[batch_size:]
        yield batch.to(device)


def _snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def full_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, kind: str,
              pair: Optional[Sequence[int]] = None, batch_size: int = 1000) -> float:
    """Mean training loss over the whole set, in eval mode."""
    model.eval()
    total = 0.0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i:i + batch_size], y[i:i + batch_size]
        total += classification_loss(model(xb), yb, kind, pair).item() * len(xb)
    return total / len(x)


def full_target_loss(model: nn.Module, x: torch.Tensor, targets: torch.Tensor, batch_size: int = 1000) -> float:
    """Mean squared loss against per-row target vectors over the whole set, in eval mode."""
    model.eval()
    total = 0.0
    for i in range(0, len(x), batch_size):
        total += _squared(model(x[i:i + batch_size]), targets[i:i + batch_size]).item() * len(x[i:i + batch_size])
    return total / len(x)


def train_from_scratch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: ScratchConfig,
                       *, seed: int, loss_kind: str = "ce", log: Callable[[str], None] = print) -> Dict:
    """Pretrain or Retrain: SGD with momentum and weight decay, cosine lr over a fixed
    number of epochs, BatchNorm trained normally."""
    device = x.device
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    n = len(x)
    steps_per_epoch = math.ceil(n / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    history: List[Dict] = []
    step = 0
    start = time.time()
    for epoch in range(cfg.epochs):
        set_train_mode(model, freeze_bn=False)
        perm = torch.randperm(n, generator=gen).to(device)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            for group in opt.param_groups:
                group["lr"] = 0.5 * cfg.lr * (1 + math.cos(math.pi * step / total_steps))
            loss = classification_loss(model(augment_batch(x[idx])), y[idx], loss_kind)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
            step += 1
        history.append({"epoch": epoch + 1, "train_loss_running": running / n,
                        "seconds": time.time() - start})
        log(f"epoch {epoch + 1}/{cfg.epochs} running loss {running / n:.4f} "
            f"({time.time() - start:.0f}s)")
    final = full_loss(model, x, y, loss_kind)
    return {"steps": step, "final_train_loss": final,
            "converged": final <= cfg.max_final_train_loss,
            "history": history, "seconds": time.time() - start}


def unlearn(model: nn.Module, x_r: torch.Tensor, y_r: torch.Tensor,
            x_f: torch.Tensor, y_f: torch.Tensor, cfg: UnlearnConfig, *, seed: int,
            loss_kind: str = "ce", log: Callable[[str], None] = print,
            row_weights: Tuple[float, float] = (1.0, 1.0)) -> Dict:
    """Signed-loss unlearning on n_r retain and n_f forget images.

    Each step uses one retain batch and one forget batch. Every `eval_every` steps
    the stop rule is checked on training data only: 10-way accuracy on all n_f
    forget images at or below threshold and, with stop_rule "both", retain loss below
    threshold (measured on at most `retain_eval_max` of the retain images).

    With stop_rule "none" nothing stops the run early: it takes exactly `max_steps` steps (status "budget"),
    which is how a mu_f = 0 fine-tuning baseline, whose forget accuracy never falls, is run.

    With stop_rule "loss" (MSE only) the run trains to convergence: it stops when the effective retain loss
    plus the effective forget loss is below `retain_loss_threshold`. The MSE objective is bounded with
    optimum 0, so this is the same "train loss below threshold" criterion as pretraining and relearning.
    The collapse abort is a guard against unbounded -CE and is skipped for MSE.

    `row_weights` (w_r, w_f) weight the retain and forget terms (training loss and the "loss" stop rule).
    The default (1, 1) weights the two sets equally; the theory mode passes the row shares, so that every
    row counts once.

    Status: "success"; "step_cap"; "collapsed" (retain loss above
    `retain_loss_abort`); or "diverged" (non-finite loss). Plain -CE is unbounded, so
    on collapse or divergence the model is rolled back to the previous check, which
    is reported as `rollback_step`.
    """
    if cfg.stop_rule not in ("both", "forget", "none", "loss"):
        raise ValueError(f"unknown stop_rule {cfg.stop_rule!r}")
    if cfg.stop_rule == "loss" and loss_kind != "mse":
        raise ValueError("stop_rule 'loss' needs the bounded MSE objective (--loss mse)")
    device = x_r.device
    gen = torch.Generator().manual_seed(seed)
    stream_r = _index_stream(len(x_r), cfg.batch_size, gen, device)
    stream_f = _index_stream(len(x_f), cfg.batch_size, gen, device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=0.0)
    x_r_eval, y_r_eval = x_r[:cfg.retain_eval_max], y_r[:cfg.retain_eval_max]

    def check() -> Dict:
        logits_r = predict_logits(model, x_r_eval)
        retain_loss = unlearning_retain_loss(logits_r, y_r_eval, cfg.mu_r, cfg.mu_f, loss_kind).item()
        logits_f = predict_logits(model, x_f)
        forget_loss = unlearning_forget_loss(logits_f, y_f, cfg.mu_r, cfg.mu_f, loss_kind).item()
        return {"retain_loss": retain_loss, "forget_loss": forget_loss, "forget_acc": accuracy(logits_f, y_f)}

    history = [{"step": 0, **check()}]
    snapshot = _snapshot(model)
    status, step, start = "step_cap", 0, time.time()
    rollback_step, forget_reached_step = None, None
    while step < cfg.max_steps:
        set_train_mode(model, cfg.freeze_bn)
        ir, i_f = next(stream_r), next(stream_f)
        loss = signed_unlearning_loss(model(augment_batch(x_r[ir])), y_r[ir], model(augment_batch(x_f[i_f])), y_f[i_f],
                                      cfg.mu_r, cfg.mu_f, loss_kind, forget_clip=cfg.forget_clip,
                                      w_r=row_weights[0], w_f=row_weights[1])
        if not torch.isfinite(loss):
            status, rollback_step = "diverged", history[-1]["step"]
            model.load_state_dict(snapshot)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            point = {"step": step, "loss": loss.item(), **check()}
            if not (math.isfinite(point["retain_loss"]) and math.isfinite(point["loss"])):
                status, rollback_step = "diverged", history[-1]["step"]
                model.load_state_dict(snapshot)
                break
            if loss_kind != "mse" and point["retain_loss"] > cfg.retain_loss_abort:
                status, rollback_step = "collapsed", history[-1]["step"]
                history.append({**point, "rejected": True})
                model.load_state_dict(snapshot)
                break
            history.append(point)
            snapshot = _snapshot(model)
            forget_ok = point["forget_acc"] <= cfg.forget_acc_threshold
            if forget_ok and forget_reached_step is None:
                forget_reached_step = step
            if cfg.stop_rule == "loss":
                if row_weights[0] * point["retain_loss"] + row_weights[1] * point["forget_loss"] \
                        < cfg.retain_loss_threshold:
                    status = "success"
                    break
            elif cfg.stop_rule != "none" and forget_ok and (
                    cfg.stop_rule == "forget" or point["retain_loss"] < cfg.retain_loss_threshold):
                status = "success"
                break
    if cfg.stop_rule == "none" and status == "step_cap":
        status = "budget"  # ran its fixed number of steps, as intended
    last = [p for p in history if not p.get("rejected")][-1]  # the state the model is in
    log(f"unlearn: {status} after {step} steps, retain loss {last['retain_loss']:.4f}, "
        f"forget acc {last['forget_acc']:.3f}"
        + (f", rolled back to step {rollback_step}" if rollback_step is not None else ""))
    return {"status": status, "steps": step, "rollback_step": rollback_step,
            "forget_reached_step": forget_reached_step, "final_retain_loss": last["retain_loss"],
            "final_forget_acc": last["forget_acc"], "final_forget_loss": last["forget_loss"], "history": history,
            "seconds": time.time() - start}


def relearn(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: RelearnConfig, *,
            seed: int, freeze_bn: bool, pair: Optional[Sequence[int]] = None,
            loss_kind: str = "ce", eval_fn: Optional[Callable[[nn.Module], Dict]] = None,
            log: Callable[[str], None] = print, targets: Optional[torch.Tensor] = None) -> Dict:
    """Train on the relearning set until its loss is below threshold, or the step cap.

    With `pair`, the loss is 2-way on the two forget logits (class split); otherwise
    it is the ordinary 10-way loss (random split). `eval_fn` is called at the steps
    in CURVE_STEPS and at the end to record a relearning curve.

    With `targets` (one target vector per row of x), the loss is the squared loss against those vectors
    over all outputs instead (the theory mode's forget-only attack: onehot for forget rows, zero for retain
    rows); `y` and `pair` are then ignored for the loss.
    """
    device = x.device
    gen = torch.Generator().manual_seed(seed)
    stream = _index_stream(len(x), cfg.batch_size, gen, device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=0.0)
    curve: List[Dict] = []

    def record(step: int) -> None:
        if eval_fn is not None:
            curve.append({"step": step, **eval_fn(model)})

    record(0)
    status, step, start = "step_cap", 0, time.time()
    rollback_step, snapshot, snapshot_step = None, _snapshot(model), 0
    def batch_loss(idx: torch.Tensor) -> torch.Tensor:
        logits = model(augment_batch(x[idx]))
        if targets is not None:
            return _squared(logits, targets[idx])
        return classification_loss(logits, y[idx], loss_kind, pair)

    def set_loss() -> float:
        return full_target_loss(model, x, targets) if targets is not None else full_loss(model, x, y, loss_kind, pair)

    train_loss = set_loss()
    if train_loss < cfg.loss_threshold:  # e.g. Pretrained already fits these images
        status = "converged"
    while status == "step_cap" and step < cfg.max_steps:
        set_train_mode(model, freeze_bn)
        idx = next(stream)
        loss = batch_loss(idx)
        if not torch.isfinite(loss):
            status, rollback_step = "diverged", snapshot_step
            model.load_state_dict(snapshot)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step in CURVE_STEPS:
            record(step)
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            train_loss = set_loss()
            if train_loss < cfg.loss_threshold:
                status = "converged"
                break
            snapshot, snapshot_step = _snapshot(model), step
    if not curve or curve[-1]["step"] != step:
        record(step)
    log(f"relearn: {status} after {step} steps, train loss {train_loss:.4f}")
    return {"status": status, "steps": step, "rollback_step": rollback_step, "final_train_loss": train_loss,
            "curve": curve, "seconds": time.time() - start}
