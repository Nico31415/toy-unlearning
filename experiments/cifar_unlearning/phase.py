"""Phase experiments of the CIFAR plan (E0-E8): the toy model's untraining / unlearning predictions on CIFAR-10.

Stages (called from run.py):

    family         --exp E --seed S --fam_L L ...       A(L): every retain image plus the first L forget images of a
                                                         nested order, same init and batches for every L. One sweep
                                                         gives every reference: A(D) = A(L_full), A(D minus S) =
                                                         A(L_full - n_S), A(D minus S_full) = A(0).
    phase          --exp E --seed S --ph_Lfull L ...    stage 2 from A(L_full) for every n_S, method, target, arm and
                                                         rescaling; distances to both references and layer read-outs
    phase_relearn  --exp E --seed S --ph_Lfull L ...    relearning attacks from every saved U and from the references

Index conventions. usable = the forget pool minus its last n_attack images (the attack pool). S_full = usable[:L_full]
and S = its LAST n_S images, so D minus S keeps usable[:L_full - n_S], exactly the family member A(L_full - n_S).

Methods (the paper's Section 5):
    F   forget term on S plus every retain image (a clean retain set)
    FR  forget term on S plus all of D minus S with true labels (retain images and the unflagged forget images)
    R   all of D minus S, no forget term
Forget targets (method F and FR):
    ybar    the class prior: 1/K on every output (mse) or the uniform distribution (ce)
    neg     the workshop paper's signed loss: retain rows (1 + rho) onehot, forget rows -rho onehot (mse only)
    oracle  A(D minus S_full)'s outputs on S (logits for mse, its softmax for ce)
    approx  A(D)'s outputs on S without the forget classes (forget logits 0 for mse; softmax over kept classes for ce)
Arms (removal geometry): full (every weight), readout (only `fc`), linear (the linearised network around the start).
Each source of rows (retain, forget) gives one batch per step and the per-source mean losses are added with equal
weight; under gradient flow to interpolation the end point does not depend on these weights, and equal weights keep a
small S from being drowned by 40,000 retain images. With --ul_stop_rule loss (mse) a run stops when every source's loss
is below --ul_retain_loss_threshold; with "none" it runs --ul_max_steps steps.
"""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.cifar_unlearning.data import Split, normalize, stratified_order
from experiments.cifar_unlearning.losses import classification_loss, target_loss
from experiments.cifar_unlearning.metrics import accuracy, kept_class_accuracy
from experiments.cifar_unlearning.models import (
    load_state, rescale_, set_train_mode, state_dict_fp16)
from experiments.cifar_unlearning.probes import (
    Forward, LinearForward, ModelForward, ProbeData, balanced_accuracy, fit_probe, linear_function,
    participation_ratio, readout, reference_features)
from experiments.cifar_unlearning import train as train_module
from experiments.cifar_unlearning.train import _index_stream, augment_batch

METHODS = ("F", "FR", "R")
TARGETS = ("ybar", "neg", "oracle", "approx")
ARMS = ("full", "readout", "linear")


def fmt(x: float) -> str:
    return f"{x:g}"


# --------------------------------------------------------------------------- indices and data

def usable_pool(split: Split, n_attack: int) -> np.ndarray:
    pool = split.forget_pool_order
    if not 0 < n_attack < len(pool):
        raise SystemExit(f"--ph_n_attack must be in (0, {len(pool)}), got {n_attack}")
    return pool[:-n_attack]


def attack_pool(split: Split, n_attack: int) -> np.ndarray:
    return split.forget_pool_order[-n_attack:]


def family_dir(ctx, exp: str, seed: int, L: int, read: bool = False) -> Path:
    """Where the family member A(L) is written; with read=True, where stage 2 loads it from (--family_root, if given,
    so that a variant of stage 2 in its own output root can share one family)."""
    base = Path(ctx.args.family_root) if (read and getattr(ctx.args, "family_root", None)) else ctx.out
    return base / "family" / f"exp={exp}" / f"seed={seed}" / f"L={L}"


def phase_dir(ctx, exp: str, seed: int, l_full: int, n_s: int, method: str, target: str, arm: str,
              rs_mode: str, S: float) -> Path:
    rs = "none" if S == 1.0 else f"{rs_mode}-S{fmt(S)}"
    return (ctx.out / "phase" / f"exp={exp}" / f"seed={seed}" / f"L={l_full}" / f"nS={n_s}" / f"m={method}"
            / f"t={target}" / f"arm={arm}" / f"rescale={rs}")


def probe_data(ctx, split: Split) -> ProbeData:
    """Probe fitting: removed images no model ever trains on (the start of P_f) and the first kept-class test images;
    evaluation: all forget-class test images and the next kept-class test images."""
    ph = ctx.cfg["ph"]
    ds = ctx.dataset

    def test(idx):
        index = torch.as_tensor(np.asarray(idx), dtype=torch.int64)
        return normalize(ds.x_test[index], ctx.device), ds.y_test[index].to(ctx.device)

    if split.kind not in ("class", "mix"):
        raise SystemExit("phase experiments need a class or mix split (pair-A-B, class-K or mix-C-ALPHA)")
    probe = stratified_order(split.probe, ds.y_train.numpy(), np.random.default_rng(0))
    n_pf = min(ph.probe_train_forget, len(probe))
    x_pf, l_pf = ctx.train_subset(probe[:n_pf])
    tr = split.test_retain
    n_tr = min(ph.probe_train_retain, len(tr) // 2)
    x_tr_a, l_tr_a = test(tr[:n_tr])
    x_tr_b, _ = test(tr[n_tr:n_tr + ph.probe_eval_retain])
    x_tf, _ = test(split.test_forget)
    dev = ctx.device
    x_train = torch.cat([x_pf, x_tr_a])
    y_train = torch.cat([torch.ones(len(x_pf), device=dev), torch.zeros(len(x_tr_a), device=dev)])
    x_eval = torch.cat([x_tf, x_tr_b])
    y_eval = torch.cat([torch.ones(len(x_tf), device=dev), torch.zeros(len(x_tr_b), device=dev)])
    return ProbeData(x_train, y_train, x_eval, y_eval, x_tf, labels_train=torch.cat([l_pf, l_tr_a]))


# --------------------------------------------------------------------------- stage: family

def stage_family(ctx) -> None:
    from experiments.cifar_unlearning import run as R

    exp, seed = ctx.args.exp, ctx.args.seed
    split = ctx.split(exp, seed)
    usable = usable_pool(split, ctx.cfg["ph"].n_attack)
    for L in sorted(ctx.args.fam_L or []):
        if not 0 <= L <= len(usable):
            raise SystemExit(f"--fam_L {L} outside [0, {len(usable)}] (forget pool minus the attack pool)")
        out = family_dir(ctx, exp, seed, L)
        if R._done(out / "result.json", ctx, "family"):
            continue
        R.log(f"=== family {exp} seed {seed} L={L}")
        idx = np.concatenate([split.retain_order, usable[:L]])
        x, y = ctx.train_subset(idx)
        model = ctx.new_model(R.derive_seed("family-init", exp, seed))
        train = R.train_from_scratch(model, x, y, ctx.cfg["pt"], seed=R.derive_seed("family-batches", exp, seed),
                                     loss_kind=ctx.loss, log=R.log)
        ev = ctx.eval_sets(split)
        metrics = R.evaluate(model, ev, split.pair)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        logits = {"f": R.predict_logits(model, ev.x_test_forget).half().cpu(),
                  "r": R.predict_logits(model, ev.x_test_retain).half().cpu()}
        torch.save(logits, out / "logits.pt")
        pd_ = probe_data(ctx, split)
        feats = reference_features(ModelForward(model), pd_, with_probes=False)
        probes = {}
        for k in feats.train:
            probe = fit_probe(feats.train[k], pd_.y_train)
            probes[f"probe_{k}"] = balanced_accuracy(probe.score(feats.eval[k]), pd_.y_eval)
            probes[f"pr_{k}"] = participation_ratio(feats.forget[k])
        record = ctx.base_record(stage="family", exp=exp, seed=seed, L=L, n_retain=len(split.retain_order),
                                 train=train, metrics=metrics, probes=probes)
        if train["converged"]:
            R.write_json(out / "result.json", record)
        else:  # recorded but marked: a member that did not fit its data is not a valid reference
            R.write_json(out / "result.json", {**record, "warning": "final train loss above threshold"})
        R.log(f"wrote {out / 'result.json'}")


# --------------------------------------------------------------------------- references for stage 2

class References:
    """Family members loaded on demand, with their logits and (for A(D) and A(0)) their features."""

    def __init__(self, ctx, exp: str, seed: int, split: Split):
        self.ctx, self.exp, self.seed, self.split = ctx, exp, seed, split
        self._models: Dict[int, nn.Module] = {}
        self._feats: Dict[Tuple[int, bool], object] = {}
        self.pd = probe_data(ctx, split)
        self.ev = ctx.eval_sets(split)

    def path(self, L: int) -> Path:
        return family_dir(self.ctx, self.exp, self.seed, L, read=True) / "model.pt"

    def model(self, L: int) -> nn.Module:
        if L not in self._models:
            p = self.path(L)
            if not p.exists():
                raise SystemExit(f"{p} missing: run the family stage with --fam_L {L} first")
            self._models[L] = self.ctx.load_model(p)
        return self._models[L]

    def logits(self, L: int) -> Dict[str, torch.Tensor]:
        if ("logits", L) not in self._feats:
            fwd = ModelForward(self.model(L))
            self._feats[("logits", L)] = {"f": fwd(self.ev.x_test_forget)[0], "r": fwd(self.ev.x_test_retain)[0]}
        return self._feats[("logits", L)]

    def amps(self, l_full: int) -> Dict[str, Dict[str, float]]:
        """amp_<layer> of A(L_full) and of A(0), both read by A(L_full)'s frozen probes: the two ends of the scale
        that rho_amp is measured on."""
        key = ("amps", l_full)
        if key not in self._feats:
            ad, a0 = self.features(l_full, True), self.features(0, False)
            self._feats[key] = {name: {k: v for k, v in readout(ModelForward(self.model(L)), self.pd, ad, a0).items()
                                       if k.startswith("amp_")} for name, L in (("ad", l_full), ("a0", 0))}
        return self._feats[key]

    def features(self, L: int, with_probes: bool):
        key = (L, with_probes)
        if key not in self._feats:
            self._feats[key] = reference_features(ModelForward(self.model(L)), self.pd, with_probes)
        return self._feats[key]


def _z(logits: torch.Tensor, loss: str) -> torch.Tensor:
    """Outputs compared by the distances: logits for mse, log-probabilities for ce (logits are shift-invariant)."""
    return logits if loss == "mse" else F.log_softmax(logits, dim=1)


def _msd(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).pow(2).sum(1).mean().item()


# --------------------------------------------------------------------------- targets

def onehot_targets(y: torch.Tensor, k: int, scale: float = 1.0) -> torch.Tensor:
    return scale * F.one_hot(y, k).float()


def forget_targets(kind: str, target: str, logits_oracle: Optional[torch.Tensor], logits_ad: Optional[torch.Tensor],
                   y_s: torch.Tensor, k: int, forget_classes: Sequence[int], rho: float) -> torch.Tensor:
    n = len(y_s)
    if target == "ybar":
        return torch.full((n, k), 1.0 / k, device=y_s.device)
    if target == "neg":
        if kind != "mse":
            raise SystemExit("target neg is the MSE effective target; use the old unlearn_relearn stage for -CE")
        return onehot_targets(y_s, k, -rho)
    if target == "oracle":
        return logits_oracle if kind == "mse" else F.softmax(logits_oracle, dim=1)
    if target == "approx":
        fc = list(forget_classes)
        if kind == "mse":
            t = logits_ad.clone()
            t[:, fc] = 0.0
            return t
        masked = logits_ad.clone()
        masked[:, fc] = -float("inf")
        return F.softmax(masked, dim=1)
    raise SystemExit(f"unknown target {target!r}; expected one of {TARGETS}")


# --------------------------------------------------------------------------- stage-2 loop

def fit_sources(logits_of, params: List[torch.Tensor], sources: List[Tuple[torch.Tensor, torch.Tensor]],
                cfg, *, loss_kind: str, seed: int, eval_logits, set_mode, log, loss_eval_max: int,
                track=None, track_every: int = 0) -> Dict:
    """SGD (momentum) on `params`: every step takes one batch from each source (x, target rows) and adds the
    per-source mean `target_loss`. Checks every cfg.eval_every steps: every source's mean loss on at most
    loss_eval_max rows; stop_rule "loss" stops when all are below cfg.retain_loss_threshold."""
    device = sources[0][0].device
    gen = torch.Generator().manual_seed(seed)
    streams = [_index_stream(len(x), cfg.batch_size, gen, device) for x, _ in sources]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=0.0)

    def check() -> List[float]:
        out = []
        for x, t in sources:
            xs, ts = x[:loss_eval_max], t[:loss_eval_max]
            out.append(target_loss(eval_logits(xs), ts, loss_kind).item())
        return out

    history = [{"step": 0, "losses": check()}]
    tracked = [{"step": 0, **track()}] if track is not None and track_every else []
    status, step, start = "budget", 0, time.time()
    while step < cfg.max_steps:
        set_mode()
        idx = [next(s) for s in streams]
        xb = torch.cat([augment_batch(x[i]) for (x, _), i in zip(sources, idx)])
        logits = logits_of(xb)
        loss, at = 0.0, 0
        for (x, t), i in zip(sources, idx):
            loss = loss + target_loss(logits[at:at + len(i)], t[i], loss_kind)
            at += len(i)
        if not torch.isfinite(loss):
            status = "diverged"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if track is not None and track_every and step % track_every == 0:
            tracked.append({"step": step, **track()})
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            losses = check()
            history.append({"step": step, "loss": loss.item(), "losses": losses})
            # a finite but exploding loss (the linearised arm at lr 1e-3 can blow up without overflowing) is also divergence
            cap = 1e3 * max(1.0, max(history[0]["losses"]))
            if not all(math.isfinite(v) and v < cap for v in losses):
                status = "diverged"
                break
            if cfg.stop_rule == "loss" and max(losses) < cfg.retain_loss_threshold:
                status = "success"
                break
    if cfg.stop_rule == "loss" and status == "budget":
        status = "step_cap"
    log(f"stage 2: {status} after {step} steps, source losses {[round(v, 5) for v in history[-1]['losses']]}")
    return {"status": status, "steps": step, "history": history, "tracked": tracked,
            "final_losses": history[-1]["losses"], "seconds": time.time() - start}


# --------------------------------------------------------------------------- measurements of one U

def refit_forget_bias(logits_cal: torch.Tensor, y_cal: torch.Tensor, logits_test: torch.Tensor,
                      forget_classes: Sequence[int]) -> torch.Tensor:
    """'Has B up to scale' at the output: fit a temperature and one bias per forget class (10-way CE on the
    calibration images), return the refitted test logits."""
    fc = list(forget_classes)
    log_t = torch.zeros((), device=logits_cal.device, requires_grad=True)
    bias = torch.zeros(len(fc), device=logits_cal.device, requires_grad=True)

    def apply(lo):
        out = lo * torch.exp(log_t)
        add = torch.zeros_like(out)
        add[:, fc] = bias
        return out + add

    opt = torch.optim.LBFGS([log_t, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(apply(logits_cal), y_cal)
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return apply(logits_test)


def measure(ctx, fwd: Forward, refs: References, l_full: int, n_s: int, pair) -> Dict:
    """Distances to both references (outputs on held-out forget-class test images), accuracies, the bias refit and
    the layer read-outs."""
    loss = ctx.loss
    ev, pd_ = refs.ev, refs.pd
    lo_f, _ = fwd(ev.x_test_forget)
    lo_r, _ = fwd(ev.x_test_retain)
    ref = {name: refs.logits(L) for name, L in (("ad", l_full), ("ut", l_full - n_s), ("ul", 0))}
    zf = {k: _z(v["f"], loss) for k, v in ref.items()}
    zr = {k: _z(v["r"], loss) for k, v in ref.items()}
    uf, ur = _z(lo_f, loss), _z(lo_r, loss)
    norm_f = _msd(zf["ad"], zf["ul"])
    out = {
        "norm_f": norm_f,
        "d_ut2": _msd(uf, zf["ut"]) / norm_f,
        "d_ul2": _msd(uf, zf["ul"]) / norm_f,
        "d_ref2": _msd(zf["ut"], zf["ul"]) / norm_f,
        "d_ad2": _msd(uf, zf["ad"]) / norm_f,
        # the do-nothing baseline: how far A(D) itself already is from A(D minus S) (its distance to A(0) is 1 by the
        # normalisation). U untrains non-trivially only if d_ut2 is well below this.
        "d_noop_ut2": _msd(zf["ad"], zf["ut"]) / norm_f,
        "retain_d_ad2": _msd(ur, zr["ad"]) / max(zr["ad"].float().pow(2).sum(1).mean().item(), 1e-12),
        "A_f10": accuracy(lo_f, ev.y_test_forget),
        "A_r": accuracy(lo_r, ev.y_test_retain),
        "A_r_kept": kept_class_accuracy(lo_r, ev.y_test_retain, pair),
        "ref_A_f10_ad": accuracy(ref["ad"]["f"], ev.y_test_forget),
        "ref_A_f10_ut": accuracy(ref["ut"]["f"], ev.y_test_forget),
        "ref_A_f10_ul": accuracy(ref["ul"]["f"], ev.y_test_forget),
    }
    # bias and temperature refit on the probe-fitting images (removed images and kept-class test images)
    lo_cal, _ = fwd(pd_.x_train)
    refit = refit_forget_bias(lo_cal, pd_.labels_train, lo_f, pair)
    out["A_f10_refit"] = accuracy(refit, ev.y_test_forget)
    out.update(readout(fwd, pd_, refs.features(l_full, True), refs.features(0, False)))
    # Magnitude of the forget representation along A(D)'s probe direction, on the A(0) -> A(D) scale: 1 = as in A(D),
    # 0 = as in A(0) (which never trained on the forget images). In the diagonal network this is the hidden energy
    # ||w_F||^2 that sets relearning speed; a scale-blind probe cannot see it.
    amps = refs.amps(l_full)
    for k, v in amps["ad"].items():
        lo, hi = amps["a0"][k], v
        out[f"ref_{k}_ad"], out[f"ref_{k}_a0"] = hi, lo
        out[f"rho_{k}"] = (out[k] - lo) / (hi - lo) if abs(hi - lo) > 1e-12 else float("nan")
    return out


def per_sample_loss(logits: torch.Tensor, y: torch.Tensor, loss: str) -> torch.Tensor:
    if loss == "mse":
        return 0.5 * (logits.float() - F.one_hot(y, logits.shape[1]).float()).pow(2).sum(1)
    return F.cross_entropy(logits.float(), y, reduction="none")


def auc(members: torch.Tensor, others: torch.Tensor) -> float:
    """P(score of a member > score of a non-member), ties counted half (the Mann-Whitney statistic)."""
    from scipy.stats import rankdata
    m, o = members.double().cpu().numpy(), others.double().cpu().numpy()
    if not len(m) or not len(o):
        return float("nan")
    ranks = rankdata(np.concatenate([m, o]))
    return float((ranks[:len(m)].sum() - len(m) * (len(m) + 1) / 2) / (len(m) * len(o)))


def member_metrics(ctx, fwd: Forward, refs: References, l_full: int, n_s: int, x_s: torch.Tensor,
                   y_s: torch.Tensor, pair) -> Dict:
    """Untraining measured on S's own training images, split into its class part (labels in `pair`) and its iid
    part (the other labels; mix experiments only). For U and for the references A(D) and A(D minus S):
      acc_S_<part>      accuracy on those images (no augmentation)
      gap_S_<part>      mean loss on them minus mean loss on held-out test images of the same classes
      mia_<part>        AUC of minus the loss as a membership score, those images against the held-out ones
                        (A(D minus S) never trained on them: its AUC is the untrained level, about 0.5)
      r_S_ut_<part>     ||U - A(D minus S)||^2 / ||A(D) - A(D minus S)||^2 on their outputs: 0 = untrained,
                        1 = no change from A(D), above 1 = moved past the target."""
    loss, ev = ctx.loss, refs.ev
    in_pair = torch.isin(y_s, torch.as_tensor(list(pair), device=y_s.device))
    held = {"cls": (ev.x_test_forget, ev.y_test_forget), "iid": (ev.x_test_retain, ev.y_test_retain)}
    models = {"U": fwd, "ad": ModelForward(refs.model(l_full)), "ut": ModelForward(refs.model(l_full - n_s))}
    out = {}
    for part, sel in (("cls", in_pair), ("iid", ~in_pair)):
        if not sel.any():
            continue
        xs, ys = x_s[sel], y_s[sel]
        xh, yh = held[part]
        z = {}
        for name, f in models.items():
            lo, _ = f(xs)
            lh, _ = f(xh)
            ls, lhl = per_sample_loss(lo, ys, loss), per_sample_loss(lh, yh, loss)
            pre = "" if name == "U" else f"ref_{name}_"
            out[f"{pre}acc_S_{part}"] = accuracy(lo, ys)
            out[f"{pre}gap_S_{part}"] = ls.mean().item() - lhl.mean().item()
            out[f"{pre}mia_{part}"] = auc(-ls, -lhl)
            z[name] = _z(lo, loss)
        out[f"r_S_ut_{part}"] = _msd(z["U"], z["ut"]) / max(_msd(z["ad"], z["ut"]), 1e-12)
        out[f"n_S_{part}"] = int(sel.sum().item())
    return out


# --------------------------------------------------------------------------- stage: phase

def stage_phase(ctx) -> None:
    from experiments.cifar_unlearning import run as R

    exp, seed, ul, ph = ctx.args.exp, ctx.args.seed, ctx.cfg["ul"], ctx.cfg["ph"]
    split = ctx.split(exp, seed)
    usable = usable_pool(split, ph.n_attack)
    l_full = ctx.args.ph_Lfull
    if l_full is None or not 0 < l_full <= len(usable):
        raise SystemExit(f"--ph_Lfull must be in [1, {len(usable)}]")
    refs = References(ctx, exp, seed, split)
    pair = split.pair
    k = ctx.dataset.num_classes
    rho = ul.mu_f / (ul.mu_r - ul.mu_f) if ul.mu_r > ul.mu_f else 0.0
    x_ret, y_ret = ctx.train_subset(split.retain_order)

    for n_s in sorted(ctx.args.ph_nS):
        if not 0 < n_s <= l_full:
            raise SystemExit(f"--ph_nS {n_s} must be in [1, {l_full}]")
        s_idx, left_idx = usable[l_full - n_s:l_full], usable[:l_full - n_s]
        x_s, y_s = ctx.train_subset(s_idx)
        x_left, y_left = ctx.train_subset(left_idx)
        for method in ctx.args.ph_methods:
            targets = ["-"] if method == "R" else ctx.args.ph_targets
            for target in targets:
                for arm in ctx.args.ph_arms:
                    for S in ctx.args.ph_S:
                        out = phase_dir(ctx, exp, seed, l_full, n_s, method, target, arm, ctx.args.ph_rs_mode, S)
                        if R._done(out / "result.json", ctx, "phase"):
                            continue
                        R.log(f"=== phase {exp} seed {seed} L={l_full} nS={n_s} {method} {target} {arm} S={fmt(S)}")
                        run_one(ctx, refs, out, exp, seed, l_full, n_s, method, target, arm, S, rho, k, pair,
                                x_ret, y_ret, x_s, y_s, x_left, y_left)


def run_one(ctx, refs: References, out: Path, exp, seed, l_full, n_s, method, target, arm, S, rho, k, pair,
            x_ret, y_ret, x_s, y_s, x_left, y_left) -> None:
    from experiments.cifar_unlearning import run as R

    ul, ph = ctx.cfg["ul"], ctx.cfg["ph"]
    model = ctx.load_model(refs.path(l_full))
    if S != 1.0:
        rescale_(model, ctx.args.ph_rs_mode, S=S)
    neg = target == "neg"
    keep_scale = (1.0 + rho) if neg else 1.0
    retain_rows = [(x_ret, onehot_targets(y_ret, k, keep_scale))]
    if method == "F" and len(x_left):
        # F's clean retain set: every image not in the forget pool, plus the pool images left in D that are not of the
        # forget classes (the iid part of a mix experiment: ordinary data, not unflagged forget-class images)
        keep = ~torch.isin(y_left, torch.as_tensor(list(pair), device=y_left.device))
        if keep.any():
            retain_rows = [(torch.cat([x_ret, x_left[keep]]),
                            onehot_targets(torch.cat([y_ret, y_left[keep]]), k, keep_scale))]
    if method in ("FR", "R") and len(x_left):
        retain_rows = [(torch.cat([x_ret, x_left]), onehot_targets(torch.cat([y_ret, y_left]), k, keep_scale))]
    sources = list(retain_rows)
    if method in ("F", "FR"):
        oracle = R.predict_logits(refs.model(0), x_s) if target == "oracle" else None
        ad = R.predict_logits(refs.model(l_full), x_s) if target == "approx" else None
        sources.append((x_s, forget_targets(ctx.loss, target, oracle, ad, y_s, k, pair, rho)))

    seed2 = R.derive_seed("phase", exp, seed, l_full, n_s, method, target, arm, S)
    if arm == "linear":
        theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
        delta = {n: torch.zeros_like(p, requires_grad=True) for n, p in theta0.items()}
        fn = linear_function(model, theta0)
        params = list(delta.values())
        logits_of = lambda xb: fn(delta, xb)[0]  # noqa: E731

        def eval_logits(xs):
            with torch.no_grad():
                return torch.cat([fn(delta, xs[i:i + 500])[0] for i in range(0, len(xs), 500)])

        set_mode = lambda: None  # noqa: E731
        fwd_now = lambda: LinearForward(model, theta0, {n: d.detach() for n, d in delta.items()})  # noqa: E731
    else:
        for p in model.parameters():
            p.requires_grad_(arm == "full")
        if arm == "readout":
            for p in model.fc.parameters():
                p.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        freeze = ul.freeze_bn or arm == "readout"
        logits_of = model
        eval_logits = lambda xs: R.predict_logits(model, xs)  # noqa: E731
        set_mode = lambda: set_train_mode(model, freeze)  # noqa: E731
        fwd_now = lambda: ModelForward(model)  # noqa: E731

    track = (lambda: measure(ctx, fwd_now(), refs, l_full, n_s, pair)) if ph.track_every else None
    if arm == "linear" and ph.linear_lr_scale != 1.0:
        # the linearised network has no ReLU saturation to damp large steps: R at S = 1 blew up at ul_lr 1e-3
        ul = dataclasses.replace(ul, lr=ul.lr * ph.linear_lr_scale)
    fit = fit_sources(logits_of, params, sources, ul, loss_kind=ctx.loss, seed=seed2, eval_logits=eval_logits,
                      set_mode=set_mode, log=R.log, loss_eval_max=ph.loss_eval_max, track=track,
                      track_every=ph.track_every)
    metrics = measure(ctx, fwd_now(), refs, l_full, n_s, pair)
    metrics.update(member_metrics(ctx, fwd_now(), refs, l_full, n_s, x_s, y_s, pair))
    out.mkdir(parents=True, exist_ok=True)
    if arm == "linear":
        torch.save({n: d.detach().half().cpu() for n, d in delta.items()}, out / "delta_fp16.pt")
    else:
        torch.save(state_dict_fp16(model), out / "model_fp16.pt")
    record = ctx.base_record(stage="phase", exp=exp, seed=seed, pair=pair, L_full=l_full, n_S=n_s,
                             L_left=l_full - n_s, method=method, target=target, arm=arm,
                             rescale={"mode": ctx.args.ph_rs_mode, "S": S}, n_retain_rows=len(sources[0][0]),
                             n_forget_rows=len(x_s) if method != "R" else 0, fit=fit, metrics=metrics)
    R.write_json(out / "result.json", record)
    R.log(f"wrote {out / 'result.json'}: d_ut2 {metrics['d_ut2']:.3f} d_ul2 {metrics['d_ul2']:.3f} "
          f"A_f10 {metrics['A_f10']:.3f}")


# --------------------------------------------------------------------------- stage: relearning attacks

def stage_phase_relearn(ctx) -> None:
    """Relearn every saved U of (exp, seed, L_full) (full and readout arms; the linearised fine-tune has no weights
    to attack) and the references A(L_full - n_S), A(0) and a fresh network, on the first n_RL images of the attack
    pool (forget images only, true labels), for every n_RL in --ph_nRL."""
    from experiments.cifar_unlearning import run as R

    exp, seed, ph = ctx.args.exp, ctx.args.seed, ctx.cfg["ph"]
    split = ctx.split(exp, seed)
    attack = attack_pool(split, ph.n_attack)
    ev = ctx.eval_sets(split)
    pair = split.pair if split.pair is not None and len(split.pair) == 2 else None
    l_full = ctx.args.ph_Lfull
    base = ctx.out / "phase" / f"exp={exp}" / f"seed={seed}" / f"L={l_full}"
    sources = sorted(p.parent for p in base.glob("nS=*/m=*/t=*/arm=*/rescale=*/model_fp16.pt"))
    lefts = sorted({R.read_json(p / "result.json")["L_left"] for p in sources} | {0})

    def attack_one(model, out: Path, freeze_bn: bool, n: int, label: str) -> None:
        if R._done(out / "result.json", ctx, "phase_relearn"):
            return
        x, y = ctx.train_subset(attack[:n])
        res = R.relearn(model, x, y, ctx.cfg["rl"], seed=R.derive_seed("phase-relearn", exp, seed, label, n),
                        freeze_bn=freeze_bn, pair=pair, loss_kind=ctx.loss,
                        eval_fn=lambda m: {"A_f10": accuracy(R.predict_logits(m, ev.x_test_forget), ev.y_test_forget),
                                           "A_r": accuracy(R.predict_logits(m, ev.x_test_retain), ev.y_test_retain)},
                        log=R.log)
        R.write_json(out / "result.json", ctx.base_record(stage="phase_relearn", exp=exp, seed=seed, source=label,
                                                          n_RL=n, relearn=res))

    for n in sorted(ctx.args.ph_nRL):
        if not 0 < n <= len(attack):
            raise SystemExit(f"--ph_nRL {n} must be in [1, {len(attack)}]")
        for src in sources:
            model = load_state(ctx.new_model(0), torch.load(src / "model_fp16.pt", map_location=ctx.device))
            attack_one(model, src / "relearn" / f"n={n}", ctx.cfg["rl"].freeze_bn, n, str(src.relative_to(ctx.out)))
        for L in lefts:
            model = ctx.load_model(family_dir(ctx, exp, seed, L, read=True) / "model.pt")
            attack_one(model, family_dir(ctx, exp, seed, L) / "relearn" / f"n={n}", ctx.cfg["rl"].freeze_bn, n,
                       f"family-L={L}")
        scratch = ctx.new_model(R.derive_seed("phase-scratch-init", exp, seed))
        attack_one(scratch, ctx.out / "family" / f"exp={exp}" / f"seed={seed}" / "scratch" / "relearn" / f"n={n}",
                   False, n, "scratch")
