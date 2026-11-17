"""Data, model, losses and read-outs for unlearning one modular-addition task in a one-layer Transformer.

Tasks. Each task is addition mod p on pairs x, y < p. The sequence is [task token, x, y, "="] and the answer is read at
"=" from one logit per number token (0 .. max_p - 1); only the first p logits count for a task with modulus p. Number
tokens are shared across tasks, so frequencies shared between moduli live in one embedding matrix.

An experiment "p60-q30" forgets addition mod 60 while retaining mod q (the partner) and the moduli of --retain_extra
(59 and 61 by default). gcd(60, q) - 1 is the most frequencies the partner can share with the forget task.

Splits (per task, seeded): 50% training pairs, 25% attack pool (relearning), 25% test; the test pairs are split again
into probe-fitting and evaluation halves. The forget task's training pairs come in a fixed nested order: S_full is
the first L_full of them and S the last n_S of S_full, so D minus S is the family member A(L_full - n_S).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- tasks and data

def parse_exp(name: str) -> Tuple[int, int]:
    """'p60-q30' -> (60, 30)."""
    try:
        a, b = name.split("-")
        assert a[0] == "p" and b[0] == "q"
        return int(a[1:]), int(b[1:])
    except Exception as err:  # noqa: BLE001
        raise SystemExit(f"experiment {name!r} must look like p60-q30") from err


@dataclass
class TaskData:
    p: int
    token: int
    train: np.ndarray  # (n, 2) pairs, nested random order
    attack: np.ndarray
    test_fit: np.ndarray  # probe fitting
    test_eval: np.ndarray  # probe evaluation; test = test_fit + test_eval


@dataclass
class Setup:
    moduli: List[int]  # forget modulus first
    tasks: List[TaskData]
    max_p: int
    eq_token: int
    vocab: int

    @property
    def forget(self) -> TaskData:
        return self.tasks[0]

    @property
    def retain(self) -> List[TaskData]:
        return self.tasks[1:]


def make_setup(exp: str, retain_extra: Sequence[int], seed: int) -> Setup:
    pf, q = parse_exp(exp)
    moduli = [pf, q] + [m for m in retain_extra if m not in (pf, q)]
    max_p = max(moduli)
    tasks = []
    for i, p in enumerate(moduli):
        rng = np.random.default_rng([seed, p, 7])
        pairs = np.stack(np.meshgrid(np.arange(p), np.arange(p), indexing="ij"), -1).reshape(-1, 2)
        pairs = pairs[rng.permutation(len(pairs))]
        n = len(pairs)
        a, b = n // 2, n // 2 + n // 4
        test = pairs[b:]
        tasks.append(TaskData(p, max_p + i, pairs[:a], pairs[a:b], test[:len(test) // 2], test[len(test) // 2:]))
    return Setup(moduli, tasks, max_p, max_p + len(moduli), max_p + len(moduli) + 1)


def tokens(setup: Setup, task: TaskData, pairs: np.ndarray, device) -> torch.Tensor:
    n = len(pairs)
    t = np.empty((n, 4), dtype=np.int64)
    t[:, 0], t[:, 1:3], t[:, 3] = task.token, pairs, setup.eq_token
    return torch.as_tensor(t, device=device)


def answers(task: TaskData, pairs: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor((pairs[:, 0] + pairs[:, 1]) % task.p, device=device)


@dataclass
class Rows:
    """A set of sequences with per-row targets over max_p outputs and a validity mask (first p entries)."""
    x: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor

    def __len__(self):
        return len(self.x)

    @staticmethod
    def cat(parts: List["Rows"]) -> "Rows":
        return Rows(torch.cat([r.x for r in parts]), torch.cat([r.targets for r in parts]),
                    torch.cat([r.mask for r in parts]))


def valid_mask(p: int, n: int, max_p: int, device) -> torch.Tensor:
    m = torch.zeros(n, max_p, dtype=torch.bool, device=device)
    m[:, :p] = True
    return m


def labelled_rows(setup: Setup, task: TaskData, pairs: np.ndarray, device, kind: str) -> Rows:
    """True answers: onehot over the task's p outputs (mse), or the same as a distribution (ce)."""
    y = answers(task, pairs, device)
    return Rows(tokens(setup, task, pairs, device), F.one_hot(y, setup.max_p).float(),
                valid_mask(task.p, len(pairs), setup.max_p, device))


# --------------------------------------------------------------------------- model

class Attention(nn.Module):
    """Causal multi-head self-attention written out with matmuls (no fused kernel), so that the linearised fine-tune
    (forward-mode AD through the network) works on every device."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.h, self.dh).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.masked_fill(mask, -1e9).softmax(-1)
        return self.o((att @ v).transpose(1, 2).reshape(b, n, d))


class Transformer(nn.Module):
    """One (or more) attention + MLP blocks on [task, x, y, =], logits read at "=". `ln="none"` follows the
    grokking analyses of Nanda et al. (no LayerNorm); `ln="pre"` puts a LayerNorm before attention, MLP and unembedding,
    as in the grokking codebase Anguita et al. used. `kappa` multiplies the initial token embedding (the balancedness knob
    of Kunin et al. and Anguita et al.)."""

    def __init__(self, vocab: int, n_out: int, d: int = 128, heads: int = 4, d_mlp: int = 512, layers: int = 1,
                 ln: str = "none", act: str = "relu", kappa: float = 1.0, seq: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(seq, d) / math.sqrt(d))
        nn.init.normal_(self.embed.weight, std=kappa / math.sqrt(d))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            blk = nn.Module()
            blk.attn = Attention(d, heads)
            blk.mlp = nn.Sequential(nn.Linear(d, d_mlp), nn.ReLU() if act == "relu" else nn.GELU(), nn.Linear(d_mlp, d))
            blk.ln1 = nn.LayerNorm(d) if ln == "pre" else nn.Identity()
            blk.ln2 = nn.LayerNorm(d) if ln == "pre" else nn.Identity()
            self.blocks.append(blk)
        self.ln_f = nn.LayerNorm(d) if ln == "pre" else nn.Identity()
        self.unembed = nn.Linear(d, n_out, bias=False)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1)
        self.register_buffer("causal", mask, persistent=False)

    def forward(self, x: torch.Tensor, taps: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        h = self.embed(x) + self.pos[: x.shape[1]]
        if taps is not None:
            taps["embed"] = h[:, -1]
        for i, blk in enumerate(self.blocks):
            z = blk.ln1(h)
            h = h + blk.attn(z, self.causal[: x.shape[1], : x.shape[1]])
            if taps is not None:
                taps[f"attn{i}"] = h[:, -1]
            h = h + blk.mlp(blk.ln2(h))
            if taps is not None:
                taps[f"mlp{i}"] = h[:, -1]
        return self.unembed(self.ln_f(h[:, -1]))


# --------------------------------------------------------------------------- losses

def row_loss(logits: torch.Tensor, rows_t: torch.Tensor, mask: torch.Tensor, kind: str) -> torch.Tensor:
    """Mean over rows. mse: 0.5 * squared error over each row's valid outputs. ce: KL(target || softmax over the
    valid outputs); a onehot target gives the plain cross-entropy."""
    if kind == "mse":
        return 0.5 * ((logits - rows_t).pow(2) * mask).sum(1).mean()
    logp = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=1)
    t = rows_t * mask
    return (t * (torch.log(t.clamp_min(1e-12)) - logp)).masked_fill(~mask, 0).sum(1).mean()


def masked_accuracy(logits: torch.Tensor, y: torch.Tensor, p: int) -> float:
    return (logits[:, :p].argmax(1) == y).float().mean().item()


# --------------------------------------------------------------------------- forward helpers (plain or linearised)

TAPS = ("embed", "attn0", "mlp0")


def model_forward(model: nn.Module, x: torch.Tensor, batch: int = 4096):
    model.eval()
    logits, feats = [], {}
    with torch.no_grad():
        for i in range(0, len(x), batch):
            taps: Dict[str, torch.Tensor] = {}
            logits.append(model(x[i:i + batch], taps))
            for k, v in taps.items():
                feats.setdefault(k, []).append(v)
    return torch.cat(logits), {k: torch.cat(v) for k, v in feats.items()}


def linear_function(model: nn.Module, theta0: Dict[str, torch.Tensor]):
    """f(delta, x) -> (logits, taps) of the linearisation around theta0 (reverse-over-forward differentiable)."""
    from torch.func import functional_call, jvp

    buffers = dict(model.named_buffers())

    def run(delta, x):
        names: List[str] = []

        def f(params):
            taps: Dict[str, torch.Tensor] = {}
            out = functional_call(model, {**params, **buffers}, (x, taps))
            names[:] = list(taps)
            return (out,) + tuple(taps[n] for n in names)

        primal, tangent = jvp(f, (theta0,), (delta,))
        outs = [p + t for p, t in zip(primal, tangent)]
        return outs[0], dict(zip(names, outs[1:]))

    return run


def linear_forward(fn, delta, x: torch.Tensor, batch: int = 4096):
    logits, feats = [], {}
    with torch.no_grad():
        for i in range(0, len(x), batch):
            lo, ta = fn(delta, x[i:i + batch])
            logits.append(lo)
            for k, v in ta.items():
                feats.setdefault(k, []).append(v)
    return torch.cat(logits), {k: torch.cat(v) for k, v in feats.items()}


# --------------------------------------------------------------------------- Fourier read-outs

def fourier_basis(p: int, k: int, device) -> torch.Tensor:
    """Orthonormal (p, 2) basis of cos and sin of frequency k over x = 0 .. p-1."""
    x = torch.arange(p, device=device, dtype=torch.float64)
    b = torch.stack([torch.cos(2 * math.pi * k * x / p), torch.sin(2 * math.pi * k * x / p)], 1)
    q, _ = torch.linalg.qr(b)
    return q.float()


def fourier_coeffs(embed: torch.Tensor, p: int) -> torch.Tensor:
    """c_k = sum_x W_E[x] exp(-2 pi i k x / p) for k = 1 .. p // 2, from the number-token rows 0 .. p-1: (p//2, d)."""
    w = embed[:p].double()
    x = torch.arange(p, device=w.device, dtype=torch.float64)
    ks = torch.arange(1, p // 2 + 1, device=w.device, dtype=torch.float64)
    phase = torch.exp(-2j * math.pi * ks[:, None] * x[None, :] / p)  # (K, p)
    return phase @ w.to(torch.complex128)


def ablate_frequency(embed: torch.Tensor, p: int, k: int) -> torch.Tensor:
    """The embedding with frequency k (of modulus p) projected out of the number-token rows 0 .. p-1."""
    out = embed.clone()
    b = fourier_basis(p, k, embed.device)
    out[:p] = out[:p] - b @ (b.T @ out[:p])
    return out


def key_frequencies(model: nn.Module, rows: Rows, p: int, kind: str, n_key: int) -> Tuple[List[int], List[float]]:
    """The n_key frequencies of modulus p whose removal from the embedding raises the loss on `rows` most.
    Returns (frequencies, the loss increase of every k = 1 .. p//2)."""
    base_w = model.embed.weight.detach().clone()
    model.eval()
    with torch.no_grad():
        base = row_loss(model(rows.x), rows.targets, rows.mask, kind).item()
        rises = []
        for k in range(1, p // 2 + 1):
            model.embed.weight.copy_(ablate_frequency(base_w, p, k))
            rises.append(row_loss(model(rows.x), rows.targets, rows.mask, kind).item() - base)
        model.embed.weight.copy_(base_w)
    order = sorted(range(len(rises)), key=lambda i: -rises[i])
    return [i + 1 for i in order[:n_key]], rises


def frequency_rises(model: nn.Module, rows: Rows, p: int, kind: str, freqs: Sequence[int]) -> List[float]:
    base_w = model.embed.weight.detach().clone()
    model.eval()
    with torch.no_grad():
        base = row_loss(model(rows.x), rows.targets, rows.mask, kind).item()
        out = []
        for k in freqs:
            model.embed.weight.copy_(ablate_frequency(base_w, p, k))
            out.append(row_loss(model(rows.x), rows.targets, rows.mask, kind).item() - base)
        model.embed.weight.copy_(base_w)
    return out


def amplitude_and_phase(embed_u: torch.Tensor, embed_ad: torch.Tensor, p: int, freqs: Sequence[int]) -> Dict[str, float]:
    """Surviving amplitude r = sum_K |c_k(U)|^2 / sum_K |c_k(A(D))|^2 and phase preservation
    |<c(U), c(A(D))>| / (|c(U)| |c(A(D))|) over the key frequencies (1: same frequencies and phases)."""
    cu, ca = fourier_coeffs(embed_u, p), fourier_coeffs(embed_ad, p)
    idx = torch.as_tensor([k - 1 for k in freqs], device=cu.device)
    u, a = cu[idx].flatten(), ca[idx].flatten()
    nu, na = u.abs().pow(2).sum().item(), a.abs().pow(2).sum().item()
    inner = (u * a.conj()).sum().abs().item()
    return {"amp": nu / na if na > 0 else float("nan"),
            "phase": inner / math.sqrt(nu * na) if nu > 0 and na > 0 else float("nan")}


# --------------------------------------------------------------------------- probes

def standardise(fit: torch.Tensor, *others: torch.Tensor):
    mean, std = fit.mean(0), fit.std(0).clamp_min(1e-8)
    return [(t - mean) / std for t in (fit,) + others], (mean, std)


def softmax_probe_accuracy(f_fit, y_fit, f_eval, y_eval, n_classes: int, l2_grid=(1e-3, 1e-2, 1e-1)) -> float:
    """Multinomial logistic regression on standardised features, L2 picked on a 20% hold-out of the fitting rows."""
    (z_fit, z_eval), _ = standardise(f_fit, f_eval)
    n = len(z_fit)
    cut = max(1, n // 5)

    def train(z, y, l2):
        w = torch.zeros(z.shape[1], n_classes, device=z.device, requires_grad=True)
        b = torch.zeros(n_classes, device=z.device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(z @ w + b, y) + 0.5 * l2 * w.pow(2).sum()
            loss.backward()
            return loss

        with torch.enable_grad():
            opt.step(closure)
        return w.detach(), b.detach()

    acc = lambda wb, z, y: ((z @ wb[0] + wb[1]).argmax(1) == y).float().mean().item()  # noqa: E731
    best = max(l2_grid, key=lambda l2: acc(train(z_fit[cut:], y_fit[cut:], l2), z_fit[:cut], y_fit[:cut]))
    return acc(train(z_fit, y_fit, best), z_eval, y_eval)


def ridge(z: torch.Tensor, t: torch.Tensor, l2: float = 1e-2) -> torch.Tensor:
    z1 = torch.cat([z, torch.ones(len(z), 1, device=z.device)], 1).double()
    a = z1.T @ z1 + l2 * len(z) * torch.eye(z1.shape[1], device=z.device, dtype=torch.float64)
    return torch.linalg.solve(a, z1.T @ t.double())


def r2(pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t = t.double()
    return 1 - (pred.double() - t).pow(2).sum(0) / (t - t.mean(0)).pow(2).sum(0).clamp_min(1e-12)


def fourier_targets(s: torch.Tensor, p: int, freqs: Sequence[int]) -> torch.Tensor:
    cols = []
    for k in freqs:
        cols += [torch.cos(2 * math.pi * k * s / p), torch.sin(2 * math.pi * k * s / p)]
    return torch.stack(cols, 1).float()
