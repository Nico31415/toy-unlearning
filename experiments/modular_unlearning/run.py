"""Unlearning modular addition in a one-layer Transformer (the transformer plan, X0-X6). From the repository root:

    python experiments/modular_unlearning/run.py <stage> --exp p60-q30 --seed S [flags]

Stages
    family         --fam_L L ...     A(L): every retained task's training pairs plus the first L forget-task pairs; same
                                     init and full-batch training for every L. One sweep gives every reference.
    phase          --Lfull L         stage 2 from A(L) for every --nS, --methods (F, FR, R), --targets (ybar, oracle),
                                     --arms (full, readout, linear) and --S; distances, Fourier read-outs and probes
    phase_relearn  --Lfull L         relearning attacks (forget task only, attack pool) from every saved stage-2 model
                                     and from A(L_left), A(0) and a fresh network

Methods and targets as in the CIFAR phase experiments (experiments/cifar_unlearning/phase.py): F = forget term on S plus
every retained task; FR = forget term on S plus all of D minus S (including the unflagged forget pairs); R = D minus S
only. ybar = 1/p on the forget task's p outputs (mse) or the uniform distribution (ce); oracle = A(0)'s outputs on S.
Rescaling --rs_mode head multiplies the unembedding by S (the logits by exactly S); all multiplies every parameter by S
(Lippl and Lindsey's c_FT; not a function scale, since attention and LayerNorm are not homogeneous).

Every run writes result.json and is skipped when it exists with the same settings, so arrays can be resubmitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.modular_unlearning.core import (  # noqa: E402
    TAPS, Rows, Setup, Transformer, amplitude_and_phase, answers, fourier_targets, frequency_rises, key_frequencies,
    labelled_rows, linear_forward, linear_function, make_setup, masked_accuracy, model_forward, r2, ridge, row_loss,
    softmax_probe_accuracy, standardise, tokens, valid_mask)

STAGES = ("family", "phase", "phase_relearn")
METHODS, TARGETS, ARMS = ("F", "FR", "R"), ("ybar", "oracle"), ("full", "readout", "linear")
log = lambda *a: print(*a, flush=True)  # noqa: E731

# settings that make a result; a stored result is reused only when these match
MODEL_KEYS = ("d", "heads", "d_mlp", "layers", "ln", "act", "kappa", "retain_extra", "loss")
PT_KEYS = ("pt_lr", "pt_wd", "pt_beta2", "pt_max_steps", "pt_eval_every", "pt_acc", "pt_loss_thr", "pt_patience")
UL_KEYS = ("ul_opt", "ul_lr", "ul_max_steps", "ul_eval_every", "ul_loss_thr", "ul_stop", "rs_mode", "n_key")
RL_KEYS = ("rl_lr", "rl_max_steps", "rl_eval_every", "rl_acc")


def derive_seed(*parts) -> int:
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    os.replace(tmp, path)


def read_json(path: Path) -> Dict:
    with open(path) as fh:
        return json.load(fh)


class Ctx:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out = Path(args.out_root)
        self.setup: Setup = make_setup(args.exp, args.retain_extra, args.seed)

    def settings(self, keys) -> Dict:
        return {k: getattr(self.args, k) for k in keys}

    def done(self, path: Path, keys) -> bool:
        if not path.exists() or self.args.overwrite:
            return False
        saved = read_json(path).get("settings", {})
        now = json.loads(json.dumps(self.settings(keys)))
        if any(saved.get(k) != v for k, v in now.items()):
            raise SystemExit(f"{path} exists with other settings; use a new --out_root or --overwrite")
        log(f"skip: {path} exists")
        return True

    def new_model(self, seed: int) -> Transformer:
        torch.manual_seed(seed)
        a = self.args
        return Transformer(self.setup.vocab, self.setup.max_p, a.d, a.heads, a.d_mlp, a.layers, a.ln, a.act,
                           a.kappa).to(self.device)

    def load(self, path: Path) -> Transformer:
        m = self.new_model(0)
        state = torch.load(path, map_location=self.device)
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        return m

    def family_dir(self, L: int, read: bool = False) -> Path:
        base = Path(self.args.family_root) if (read and self.args.family_root) else self.out
        return base / "family" / f"exp={self.args.exp}" / f"seed={self.args.seed}" / f"L={L}"


# --------------------------------------------------------------------------- data views

def retain_rows(ctx: Ctx) -> Rows:
    s = ctx.setup
    return Rows.cat([labelled_rows(s, t, t.train, ctx.device, ctx.args.loss) for t in s.retain])


def forget_rows(ctx: Ctx, pairs: np.ndarray) -> Rows:
    return labelled_rows(ctx.setup, ctx.setup.forget, pairs, ctx.device, ctx.args.loss)


def test_accuracies(ctx: Ctx, logits_of) -> Dict[str, float]:
    """Held-out accuracy of every task (forget task first), from a callable x -> logits."""
    s, out = ctx.setup, {}
    for t in s.tasks:
        pairs = np.concatenate([t.test_fit, t.test_eval])
        x = tokens(s, t, pairs, ctx.device)
        out[f"acc_p{t.p}"] = masked_accuracy(logits_of(x), answers(t, pairs, ctx.device), t.p)
    out["acc_forget"] = out[f"acc_p{s.forget.p}"]
    out["acc_retain"] = float(np.mean([out[f"acc_p{t.p}"] for t in s.retain]))
    return out


# --------------------------------------------------------------------------- training

def train_full_batch(params, loss_fn, *, lr: float, wd: float, beta2: float, max_steps: int, eval_every: int,
                     check, opt_name: str = "adam") -> Dict:
    """Full-batch Adam(W) or SGD until `check()` returns True (checked every eval_every steps) or max_steps."""
    if opt_name == "adam":
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, beta2), weight_decay=wd)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    history, status, step, start = [], "step_cap", 0, time.time()
    while step < max_steps:
        loss = loss_fn()
        if not torch.isfinite(loss):
            status = "diverged"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % eval_every == 0 or step == max_steps:
            ok, point = check()
            history.append({"step": step, "loss": loss.item(), **point})
            if ok:
                status = "success"
                break
    return {"status": status, "steps": step, "history": history, "seconds": time.time() - start}


def stage_family(ctx: Ctx) -> None:
    a, s = ctx.args, ctx.setup
    for L in sorted(a.fam_L or []):
        if not 0 <= L <= len(s.forget.train):
            raise SystemExit(f"--fam_L {L} outside [0, {len(s.forget.train)}]")
        out = ctx.family_dir(L)
        if ctx.done(out / "result.json", MODEL_KEYS + PT_KEYS):
            continue
        log(f"=== family {a.exp} seed {a.seed} L={L}")
        rows = retain_rows(ctx)
        if L:
            rows = Rows.cat([rows, forget_rows(ctx, s.forget.train[:L])])
        model = ctx.new_model(derive_seed("family-init", a.exp, a.seed))

        def loss_fn():
            model.train()
            return row_loss(model(rows.x), rows.targets, rows.mask, a.loss)

        best = {"acc": -1.0, "at": 0, "n": 0}

        def check():
            # Stop when (i) every retain task generalises, (ii) the training rows are fitted (train loss below
            # pt_loss_thr; with weight decay the loss has a positive floor, so every training row classified correctly
            # instead), and (iii) the forget task has generalised or its test accuracy has not risen by 0.005 for
            # pt_patience evaluations (an A(L) below n_learn never generalises; it memorises its L rows).
            model.eval()
            with torch.no_grad():
                out = model(rows.x)
                tl = row_loss(out, rows.targets, rows.mask, a.loss).item()
                train_acc = (out.masked_fill(~rows.mask, -1e9).argmax(1) == rows.targets.argmax(1)).float().mean().item()
                accs = test_accuracies(ctx, model)
            best["n"] += 1
            if accs["acc_forget"] > best["acc"] + 0.005:
                best["acc"], best["at"] = accs["acc_forget"], best["n"]
            fitted = train_acc == 1.0 if a.pt_wd > 0 else tl < a.pt_loss_thr
            retain_ok = min(accs[f"acc_p{t.p}"] for t in s.retain) >= a.pt_acc
            forget_ok = (not L or accs["acc_forget"] >= a.pt_acc or best["n"] - best["at"] >= a.pt_patience)
            return fitted and retain_ok and forget_ok, {"train_loss": tl, "train_acc": train_acc, **accs}

        fit = train_full_batch(list(model.parameters()), loss_fn, lr=a.pt_lr, wd=a.pt_wd, beta2=a.pt_beta2,
                               max_steps=a.pt_max_steps, eval_every=a.pt_eval_every, check=check)
        log(f"family L={L}: {fit['status']} after {fit['steps']} steps, {fit['history'][-1]}")
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        xf = tokens(s, s.forget, np.concatenate([s.forget.test_fit, s.forget.test_eval]), ctx.device)
        torch.save({"f": model_forward(model, xf)[0].half().cpu()}, out / "logits.pt")  # for the seed-noise floor
        info = reference_readout(ctx, model)
        write_json(out / "result.json", {"stage": "family", "exp": a.exp, "seed": a.seed, "L": L,
                                         "settings": ctx.settings(MODEL_KEYS + PT_KEYS), "fit": fit,
                                         "metrics": test_accuracies(ctx, model), **info})


# --------------------------------------------------------------------------- read-outs

def probe_sets(ctx: Ctx):
    s, t = ctx.setup, ctx.setup.forget
    xf, xe = tokens(s, t, t.test_fit, ctx.device), tokens(s, t, t.test_eval, ctx.device)
    sf = torch.as_tensor(t.test_fit.sum(1) % t.p, device=ctx.device)
    se = torch.as_tensor(t.test_eval.sum(1) % t.p, device=ctx.device)
    return xf, sf, xe, se


def reference_readout(ctx: Ctx, model) -> Dict:
    """What a family member contributes to the read-outs of the stage-2 models: its key frequencies for the forget task
    (A(D)'s are the ones used), the loss rise when each is ablated (on the forget task and on the partner), and its
    answer-probe accuracy and Fourier R^2 per tap."""
    a, s = ctx.args, ctx.setup
    f_rows = forget_rows(ctx, s.forget.train)
    keys, rises = key_frequencies(model, f_rows, s.forget.p, a.loss, a.n_key)
    partner = s.retain[0]
    p_rows = labelled_rows(s, partner, partner.train, ctx.device, a.loss)
    partner_rises = frequency_rises(model, p_rows, s.forget.p, a.loss, range(1, s.forget.p // 2 + 1))
    return {"key_freqs": keys, "forget_rises": rises, "partner_rises": partner_rises,
            "probes": probes(ctx, lambda x: model_forward(model, x), keys, frozen=None)[0]}


def probes(ctx: Ctx, fwd, freqs: Sequence[int], frozen: Optional[Dict]):
    """Per tap: answer-probe accuracy ((x + y) mod p_f, softmax regression, standardised), Fourier R^2 of cos/sin
    2 pi k (x + y) / p_f for the given frequencies, and with `frozen` (A(D)'s ridge maps and standardisation) the R^2 of
    A(D)'s map applied to this model after refitting one gain and offset per target."""
    s = ctx.setup
    xf, sf, xe, se = probe_sets(ctx)
    _, ff = fwd(xf)
    _, fe = fwd(xe)
    tf, te = fourier_targets(sf.double(), s.forget.p, freqs), fourier_targets(se.double(), s.forget.p, freqs)
    out, maps = {}, {}
    for k in TAPS:
        if k not in ff:
            continue
        out[f"probe_{k}"] = softmax_probe_accuracy(ff[k], sf, fe[k], se, s.forget.p)
        (zf, ze), stats = standardise(ff[k], fe[k])
        w = ridge(zf, tf)
        pred = torch.cat([ze, torch.ones(len(ze), 1, device=ze.device)], 1).double() @ w
        out[f"fourier_r2_{k}"] = r2(pred, te).mean().item()
        for i, freq in enumerate(freqs):
            out[f"fourier_r2_{k}_k{freq}"] = r2(pred[:, 2 * i:2 * i + 2], te[:, 2 * i:2 * i + 2]).mean().item()
        maps[k] = (stats, w)
        if frozen is not None and k in frozen:
            (mean, std), w_ad = frozen[k]
            raw_f = torch.cat([(ff[k] - mean) / std, torch.ones(len(ff[k]), 1, device=xf.device)], 1).double() @ w_ad
            raw_e = torch.cat([(fe[k] - mean) / std, torch.ones(len(fe[k]), 1, device=xf.device)], 1).double() @ w_ad
            fitted = []
            for j in range(raw_f.shape[1]):  # one gain and offset per target
                A = torch.stack([raw_f[:, j], torch.ones_like(raw_f[:, j])], 1)
                coef = torch.linalg.lstsq(A, tf[:, j:j + 1].double()).solution
                fitted.append(torch.stack([raw_e[:, j], torch.ones_like(raw_e[:, j])], 1) @ coef)
            out[f"frozen_r2_{k}"] = r2(torch.cat(fitted, 1), te).mean().item()
    return out, maps


# --------------------------------------------------------------------------- stage 2

def rescale(model, mode: str, S: float) -> None:
    with torch.no_grad():
        if S == 1.0:
            return
        if mode == "head":
            model.unembed.weight.mul_(S)
        elif mode == "all":
            for p in model.parameters():
                p.mul_(S)
        else:
            raise SystemExit(f"unknown --rs_mode {mode!r} (head or all)")


def phase_dir(ctx: Ctx, l_full, n_s, method, target, arm, S) -> Path:
    rs = "none" if S == 1.0 else f"{ctx.args.rs_mode}-S{S:g}"
    return (ctx.out / "phase" / f"exp={ctx.args.exp}" / f"seed={ctx.args.seed}" / f"L={l_full}" / f"nS={n_s}"
            / f"m={method}" / f"t={target}" / f"arm={arm}" / f"rescale={rs}")


def stage_phase(ctx: Ctx) -> None:
    a, s = ctx.args, ctx.setup
    l_full = a.Lfull
    if l_full is None or not 0 < l_full <= len(s.forget.train):
        raise SystemExit(f"--Lfull must be in [1, {len(s.forget.train)}]")

    def ref(L):
        p = ctx.family_dir(L, read=True)
        if not (p / "model.pt").exists():
            raise SystemExit(f"{p}/model.pt missing: run the family stage with --fam_L {L} first")
        return ctx.load(p / "model.pt"), read_json(p / "result.json")

    m_ad, r_ad = ref(l_full)
    m_a0, _ = ref(0)
    freqs = r_ad["key_freqs"]
    _, frozen = probes(ctx, lambda x: model_forward(m_ad, x), freqs, frozen=None)
    xf_test = tokens(s, s.forget, np.concatenate([s.forget.test_fit, s.forget.test_eval]), ctx.device)
    mask_f = valid_mask(s.forget.p, len(xf_test), s.max_p, ctx.device)
    z = lambda lo: lo[:, :s.forget.p] if a.loss == "mse" else F.log_softmax(lo[:, :s.forget.p], 1)  # noqa: E731
    msd = lambda u, v: (u - v).pow(2).sum(1).mean().item()  # noqa: E731
    with torch.no_grad():
        z_ad, z_a0 = z(m_ad.eval()(xf_test)), z(m_a0.eval()(xf_test))
    norm = msd(z_ad, z_a0)

    for n_s in sorted(a.nS):
        if not 0 < n_s <= l_full:
            raise SystemExit(f"--nS {n_s} must be in [1, {l_full}]")
        m_ut, _ = ref(l_full - n_s)
        with torch.no_grad():
            z_ut = z(m_ut.eval()(xf_test))
        s_pairs, left = s.forget.train[l_full - n_s:l_full], s.forget.train[:l_full - n_s]
        for method in a.methods:
            for target in (["-"] if method == "R" else a.targets):
                for arm in a.arms:
                    for S in a.S:
                        out = phase_dir(ctx, l_full, n_s, method, target, arm, S)
                        if ctx.done(out / "result.json", MODEL_KEYS + PT_KEYS + UL_KEYS):
                            continue
                        log(f"=== phase {a.exp} seed {a.seed} L={l_full} nS={n_s} {method} {target} {arm} S={S:g}")
                        model = ctx.load(ctx.family_dir(l_full, read=True) / "model.pt")
                        rescale(model, a.rs_mode, S)
                        sources = [retain_rows(ctx)]
                        if method in ("FR", "R") and len(left):
                            sources = [Rows.cat([sources[0], forget_rows(ctx, left)])]
                        if method in ("F", "FR"):
                            xs = tokens(s, s.forget, s_pairs, ctx.device)
                            mask = valid_mask(s.forget.p, len(xs), s.max_p, ctx.device)
                            if target == "ybar":
                                t = mask.float() / s.forget.p
                            else:  # oracle
                                with torch.no_grad():
                                    lo = m_a0.eval()(xs)
                                t = lo if a.loss == "mse" else F.softmax(lo.masked_fill(~mask, -1e9), 1)
                            sources.append(Rows(xs, t, mask))
                        res = run_stage2(ctx, model, arm, sources,
                                         seed=derive_seed("phase", a.exp, a.seed, l_full, n_s, method, target, arm, S),
                                         ad_embed=m_ad.embed.weight.detach(), freqs=freqs)
                        fwd, logits_of, embed = res.pop("fwd"), res.pop("logits_of"), res.pop("embed")
                        with torch.no_grad():
                            zu = z(logits_of(xf_test))
                        metrics = {"norm": norm, "d_ut2": msd(zu, z_ut) / norm, "d_ul2": msd(zu, z_a0) / norm,
                                   "d_ref2": msd(z_ut, z_a0) / norm, "d_ad2": msd(zu, z_ad) / norm,
                                   **test_accuracies(ctx, logits_of),
                                   **amplitude_and_phase(embed, m_ad.embed.weight.detach(), s.forget.p, freqs),
                                   **{f"a0_{k}": v for k, v in amplitude_and_phase(  # the floor: A(0)'s embedding
                                       m_a0.embed.weight.detach(), m_ad.embed.weight.detach(), s.forget.p,
                                       freqs).items()},
                                   **probes(ctx, fwd, freqs, frozen)[0]}
                        out.mkdir(parents=True, exist_ok=True)
                        torch.save(res.pop("state"), out / ("delta.pt" if arm == "linear" else "model.pt"))
                        write_json(out / "result.json", {
                            "stage": "phase", "exp": a.exp, "seed": a.seed, "L_full": l_full, "n_S": n_s,
                            "L_left": l_full - n_s, "method": method, "target": target, "arm": arm, "S": S,
                            "rs_mode": a.rs_mode, "key_freqs": freqs,
                            "settings": ctx.settings(MODEL_KEYS + PT_KEYS + UL_KEYS), "fit": res, "metrics": metrics})
                        log(f"d_ut2 {metrics['d_ut2']:.3f} d_ul2 {metrics['d_ul2']:.3f} acc_forget "
                            f"{metrics['acc_forget']:.3f} amp {metrics['amp']:.3f} phase {metrics['phase']:.3f}")


def run_stage2(ctx: Ctx, model, arm: str, sources: List[Rows], seed: int, ad_embed=None, freqs=()) -> Dict:
    """Stage 2 on full batches; every --ul_eval_every steps it records each source's loss and (X1's dynamics) the
    forget-task accuracy and the surviving amplitude and phase of A(D)'s key frequencies."""
    a = ctx.args
    torch.manual_seed(seed)
    if arm == "linear":
        theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
        delta = {n: torch.zeros_like(p, requires_grad=True) for n, p in theta0.items()}
        fn = linear_function(model, theta0)
        params = list(delta.values())
        logits_of = lambda x: linear_forward(fn, {n: d.detach() for n, d in delta.items()}, x)[0]  # noqa: E731
        train_logits = lambda x: fn(delta, x)[0]  # noqa: E731
    else:
        for p in model.parameters():
            p.requires_grad_(arm == "full")
        if arm == "readout":
            model.unembed.weight.requires_grad_(True)
        params = [p for p in model.parameters() if p.requires_grad]
        logits_of = lambda x: model_forward(model, x)[0]  # noqa: E731

        def train_logits(x):
            model.train()
            return model(x)

    def loss_fn():
        return sum(row_loss(train_logits(r.x), r.targets, r.mask, a.loss) for r in sources)

    def check():
        with torch.no_grad():
            losses = [row_loss(logits_of(r.x), r.targets, r.mask, a.loss).item() for r in sources]
            point = {"losses": losses, "acc_forget": test_accuracies(ctx, logits_of)["acc_forget"]}
            if ad_embed is not None and len(freqs):
                emb = (theta0["embed.weight"] + delta["embed.weight"].detach()) if arm == "linear" \
                    else model.embed.weight.detach()
                point.update(amplitude_and_phase(emb, ad_embed, ctx.setup.forget.p, freqs))
        return a.ul_stop == "loss" and max(losses) < a.ul_loss_thr, point

    fit = train_full_batch(params, loss_fn, lr=a.ul_lr, wd=0.0, beta2=a.pt_beta2, max_steps=a.ul_max_steps,
                           eval_every=a.ul_eval_every, check=check, opt_name=a.ul_opt)
    log(f"stage 2: {fit['status']} after {fit['steps']} steps, losses {fit['history'][-1]['losses']}")
    if arm == "linear":
        d = {n: v.detach() for n, v in delta.items()}
        fwd = lambda x: linear_forward(fn, d, x)  # noqa: E731
        embed = theta0["embed.weight"] + d["embed.weight"]
        state = {n: v.half().cpu() for n, v in d.items()}
    else:
        fwd = lambda x: model_forward(model, x)  # noqa: E731
        embed = model.embed.weight.detach()
        state = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
    return {**fit, "fwd": fwd, "logits_of": logits_of, "embed": embed, "state": state}


# --------------------------------------------------------------------------- relearning

def relearn(ctx: Ctx, model, n: int, label: str) -> Dict:
    a, s = ctx.args, ctx.setup
    rows = forget_rows(ctx, s.forget.attack[:n])
    curve = []

    def loss_fn():
        model.train()
        return row_loss(model(rows.x), rows.targets, rows.mask, a.loss)

    def check():
        accs = test_accuracies(ctx, lambda x: model_forward(model, x)[0])
        curve.append(accs)
        return accs["acc_forget"] >= a.rl_acc, accs

    for p in model.parameters():
        p.requires_grad_(True)
    torch.manual_seed(derive_seed("relearn", a.exp, a.seed, label, n))
    fit = train_full_batch(list(model.parameters()), loss_fn, lr=a.rl_lr, wd=0.0, beta2=a.pt_beta2,
                           max_steps=a.rl_max_steps, eval_every=a.rl_eval_every, check=check)
    reached = next((h["step"] for h in fit["history"] if h["acc_forget"] >= a.rl_acc), None)
    return {**fit, "steps_to_acc": reached}


def stage_phase_relearn(ctx: Ctx) -> None:
    a, s = ctx.args, ctx.setup
    base = ctx.out / "phase" / f"exp={a.exp}" / f"seed={a.seed}" / f"L={a.Lfull}"
    srcs = sorted(p.parent for p in base.glob("nS=*/m=*/t=*/arm=*/rescale=*/model.pt"))
    lefts = sorted({read_json(p / "result.json")["L_left"] for p in srcs} | {0})
    for n in sorted(a.nRL):
        if not 0 < n <= len(s.forget.attack):
            raise SystemExit(f"--nRL {n} must be in [1, {len(s.forget.attack)}]")
        jobs = [(src / "relearn" / f"n={n}", lambda src=src: ctx.load(src / "model.pt"), str(src.relative_to(ctx.out)))
                for src in srcs]
        jobs += [(ctx.family_dir(L) / "relearn" / f"n={n}", lambda L=L: ctx.load(ctx.family_dir(L, read=True) / "model.pt"),
                  f"family/L={L}") for L in lefts]
        jobs.append((ctx.out / "family" / f"exp={a.exp}" / f"seed={a.seed}" / "scratch" / "relearn" / f"n={n}",
                     lambda: ctx.new_model(derive_seed("scratch", a.exp, a.seed)), "scratch"))
        for out, make, label in jobs:
            if ctx.done(out / "result.json", MODEL_KEYS + RL_KEYS):
                continue
            log(f"=== relearn {label} n={n}")
            res = relearn(ctx, make(), n, label)
            write_json(out / "result.json", {"stage": "phase_relearn", "exp": a.exp, "seed": a.seed, "source": label,
                                             "n_RL": n, "settings": ctx.settings(MODEL_KEYS + RL_KEYS), "relearn": res})


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--exp", required=True, help="p<forget modulus>-q<partner modulus>, e.g. p60-q30")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retain_extra", type=int, nargs="*", default=[59, 61])
    p.add_argument("--out_root", default="data/modular_unlearning/default")
    p.add_argument("--family_root", default=None, help="read the pretraining family from this root")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--loss", choices=("mse", "ce"), default="mse")
    # model
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d_mlp", type=int, default=512)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--ln", choices=("none", "pre"), default="none")
    p.add_argument("--act", choices=("relu", "gelu"), default="relu")
    p.add_argument("--kappa", type=float, default=1.0, help="initial token-embedding scale (balancedness)")
    # pretraining (family)
    p.add_argument("--fam_L", type=int, nargs="+", default=None)
    p.add_argument("--pt_lr", type=float, default=1e-3)
    p.add_argument("--pt_wd", type=float, default=1.0, help="AdamW weight decay in pretraining only (stage 2 and relearning use 0)")
    p.add_argument("--pt_beta2", type=float, default=0.98)
    p.add_argument("--pt_max_steps", type=int, default=1_000_000)
    p.add_argument("--pt_eval_every", type=int, default=1000)
    p.add_argument("--pt_acc", type=float, default=0.99, help="stop once every trained task reaches this test accuracy")
    p.add_argument("--pt_loss_thr", type=float, default=1e-3)
    p.add_argument("--pt_patience", type=int, default=20, help="evaluations without a forget-accuracy rise before stopping")
    # stage 2
    p.add_argument("--Lfull", type=int, default=None)
    p.add_argument("--nS", type=int, nargs="+", default=[400])
    p.add_argument("--methods", nargs="+", default=["F"])
    p.add_argument("--targets", nargs="+", default=["ybar"])
    p.add_argument("--arms", nargs="+", default=["full"])
    p.add_argument("--S", type=float, nargs="+", default=[1.0])
    p.add_argument("--rs_mode", choices=("head", "all"), default="head")
    p.add_argument("--ul_opt", choices=("adam", "sgd"), default="adam")
    p.add_argument("--ul_lr", type=float, default=1e-3)
    p.add_argument("--ul_max_steps", type=int, default=32000)
    p.add_argument("--ul_eval_every", type=int, default=500)
    p.add_argument("--ul_loss_thr", type=float, default=1e-3)
    p.add_argument("--ul_stop", choices=("loss", "none"), default="loss")
    p.add_argument("--n_key", type=int, default=5, help="key frequencies of the forget task")
    # relearning
    p.add_argument("--nRL", type=int, nargs="+", default=[30, 100, 300, 900])
    p.add_argument("--rl_lr", type=float, default=1e-3)
    p.add_argument("--rl_max_steps", type=int, default=32000)
    p.add_argument("--rl_eval_every", type=int, default=250)
    p.add_argument("--rl_acc", type=float, default=0.99)
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    for name, allowed in (("methods", METHODS), ("targets", TARGETS), ("arms", ARMS)):
        bad = [v for v in getattr(args, name) if v not in allowed]
        if bad:
            raise SystemExit(f"--{name}: {bad} not in {allowed}")
    ctx = Ctx(args)
    log(f"stage {args.stage} on {ctx.device}, out_root {ctx.out}, moduli {ctx.setup.moduli}")
    {"family": stage_family, "phase": stage_phase, "phase_relearn": stage_phase_relearn}[args.stage](ctx)


if __name__ == "__main__":
    main()
