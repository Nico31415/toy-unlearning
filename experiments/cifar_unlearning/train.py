"""Training loops: from scratch, unlearning with its stop rule, relearning to convergence.

Every loop works on normalised tensors that already sit on the device, and every
stopping decision uses training data only.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn

from experiments.cifar_unlearning.config import RelearnConfig, ScratchConfig, UnlearnConfig
from experiments.cifar_unlearning.losses import (
    classification_loss, signed_unlearning_loss, unlearning_retain_loss)
from experiments.cifar_unlearning.metrics import accuracy, predict_logits
from experiments.cifar_unlearning.models import set_train_mode

# Relearning curves are recorded at these steps (and at the final step).
CURVE_STEPS = (0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)


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
            loss = classification_loss(model(x[idx]), y[idx], loss_kind)
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
            loss_kind: str = "ce", log: Callable[[str], None] = print) -> Dict:
    """Signed-loss unlearning on n_r retain and n_f forget images.

    Each step uses one retain batch and one forget batch. Every `eval_every` steps
    the stop rule is checked on training data only: 10-way accuracy on all n_f
    forget images at or below threshold and, with stop_rule "both", retain loss below
    threshold (measured on at most `retain_eval_max` of the retain images).

    Status: "success"; "step_cap"; "collapsed" (retain loss above
    `retain_loss_abort`); or "diverged" (non-finite loss). Plain -CE is unbounded, so
    on collapse or divergence the model is rolled back to the previous check, which
    is reported as `rollback_step`.
    """
    if cfg.stop_rule not in ("both", "forget"):
        raise ValueError(f"unknown stop_rule {cfg.stop_rule!r}")
    device = x_r.device
    gen = torch.Generator().manual_seed(seed)
    stream_r = _index_stream(len(x_r), cfg.batch_size, gen, device)
    stream_f = _index_stream(len(x_f), cfg.batch_size, gen, device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=0.0)
    x_r_eval, y_r_eval = x_r[:cfg.retain_eval_max], y_r[:cfg.retain_eval_max]

    def check() -> Dict:
        logits_r = predict_logits(model, x_r_eval)
        retain_loss = unlearning_retain_loss(logits_r, y_r_eval, cfg.mu_r, cfg.mu_f, loss_kind).item()
        forget_acc = accuracy(predict_logits(model, x_f), y_f)
        return {"retain_loss": retain_loss, "forget_acc": forget_acc}

    history = [{"step": 0, **check()}]
    snapshot = _snapshot(model)
    status, step, start = "step_cap", 0, time.time()
    rollback_step, forget_reached_step = None, None
    while step < cfg.max_steps:
        set_train_mode(model, cfg.freeze_bn)
        ir, i_f = next(stream_r), next(stream_f)
        loss = signed_unlearning_loss(model(x_r[ir]), y_r[ir], model(x_f[i_f]), y_f[i_f],
                                      cfg.mu_r, cfg.mu_f, loss_kind)
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
            if point["retain_loss"] > cfg.retain_loss_abort:
                status, rollback_step = "collapsed", history[-1]["step"]
                history.append({**point, "rejected": True})
                model.load_state_dict(snapshot)
                break
            history.append(point)
            snapshot = _snapshot(model)
            forget_ok = point["forget_acc"] <= cfg.forget_acc_threshold
            if forget_ok and forget_reached_step is None:
                forget_reached_step = step
            if forget_ok and (cfg.stop_rule == "forget" or point["retain_loss"] < cfg.retain_loss_threshold):
                status = "success"
                break
    last = [p for p in history if not p.get("rejected")][-1]  # the state the model is in
    log(f"unlearn: {status} after {step} steps, retain loss {last['retain_loss']:.4f}, "
        f"forget acc {last['forget_acc']:.3f}"
        + (f", rolled back to step {rollback_step}" if rollback_step is not None else ""))
    return {"status": status, "steps": step, "rollback_step": rollback_step,
            "forget_reached_step": forget_reached_step, "final_retain_loss": last["retain_loss"],
            "final_forget_acc": last["forget_acc"], "history": history,
            "seconds": time.time() - start}


def relearn(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: RelearnConfig, *,
            seed: int, freeze_bn: bool, pair: Optional[Sequence[int]] = None,
            loss_kind: str = "ce", eval_fn: Optional[Callable[[nn.Module], Dict]] = None,
            log: Callable[[str], None] = print) -> Dict:
    """Train on the relearning set until its loss is below threshold, or the step cap.

    With `pair`, the loss is 2-way on the two forget logits (class split); otherwise
    it is the ordinary 10-way loss (random split). `eval_fn` is called at the steps
    in CURVE_STEPS and at the end to record a relearning curve.
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
    train_loss = full_loss(model, x, y, loss_kind, pair)
    if train_loss < cfg.loss_threshold:  # e.g. Pretrained already fits these images
        status = "converged"
    while status == "step_cap" and step < cfg.max_steps:
        set_train_mode(model, freeze_bn)
        idx = next(stream)
        loss = classification_loss(model(x[idx]), y[idx], loss_kind, pair)
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
            train_loss = full_loss(model, x, y, loss_kind, pair)
            if train_loss < cfg.loss_threshold:
                status = "converged"
                break
            snapshot, snapshot_step = _snapshot(model), step
    if not curve or curve[-1]["step"] != step:
        record(step)
    log(f"relearn: {status} after {step} steps, train loss {train_loss:.4f}")
    return {"status": status, "steps": step, "rollback_step": rollback_step, "final_train_loss": train_loss,
            "curve": curve, "seconds": time.time() - start}
