"""
functions/unlearning.py
=======================
Infrastructure for a three-stage machine-unlearning pipeline on diagonal
linear networks (DLNs).

Pipeline overview
-----------------
  Stage 1 (Pretraining)  : Train a DLN on a sparse teacher β_PT.
                           Implemented here so that ``run_experiment`` is
                           self-contained; the existing pretrain scripts are
                           left unchanged.
  Stage 2 (Unlearning)   : Fine-tune against an *effective teacher* β_eff
                           that encodes the desired unlearning objective.
                           β_eff is computed *before* any training function
                           is called.
  Stage 3 (Relearning)   : Sweep over sample counts α₃·D, re-training on
                           forget-set data and recording how quickly the
                           network relearns β_f.

Weights representation
----------------------
All weight states are passed and returned as ``DiagWeights``, a NamedTuple
of four 1-D float Tensors (w_pos, w_neg, v_pos, v_neg), each of shape (D,).

Reset modes (A and B) are handled by strictly separated helper functions;
no interleaving occurs.

β_eff is computed by ``compute_effective_teacher`` and passed explicitly to
``run_stage2``; the training loop never inspects the loss type.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Weight container
# ---------------------------------------------------------------------------

class DiagWeights(NamedTuple):
    """
    Immutable snapshot of diagonal linear network weights.

    The network realises f(x) = x @ β where
        β_d = w_pos_d · v_pos_d − w_neg_d · v_neg_d   (elementwise).

    Each field is a detached 1-D float Tensor of shape (D,).
    """
    w_pos: torch.Tensor  # shape (D,)
    w_neg: torch.Tensor  # shape (D,)
    v_pos: torch.Tensor  # shape (D,)
    v_neg: torch.Tensor  # shape (D,)

    def beta(self) -> torch.Tensor:
        """Return the effective linear map β = w⁺ ∘ v⁺ − w⁻ ∘ v⁻."""
        return self.w_pos * self.v_pos - self.w_neg * self.v_neg


# ---------------------------------------------------------------------------
# Minimal DiagonalNet (local to this module)
# ---------------------------------------------------------------------------

class _DiagonalNet(nn.Module):
    """
    Two-layer diagonal linear network: f(x) = x @ (w_pos ∘ v_pos − w_neg ∘ v_neg).

    Constructed from a ``DiagWeights`` snapshot; parameters are never shared
    with external state.  Use ``get_weights()`` to extract a detached snapshot
    after training.
    """

    def __init__(self, weights: DiagWeights) -> None:
        super().__init__()
        self.w_pos = nn.Parameter(weights.w_pos.detach().clone().float())
        self.w_neg = nn.Parameter(weights.w_neg.detach().clone().float())
        self.v_pos = nn.Parameter(weights.v_pos.detach().clone().float())
        self.v_neg = nn.Parameter(weights.v_neg.detach().clone().float())

    def beta(self) -> torch.Tensor:
        """Effective linear map β = w⁺ ∘ v⁺ − w⁻ ∘ v⁻."""
        return self.w_pos * self.v_pos - self.w_neg * self.v_neg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute f(x) = x @ β for input batch x of shape (N, D)."""
        return x @ self.beta()

    def get_weights(self) -> DiagWeights:
        """Return current parameters as a detached ``DiagWeights`` snapshot."""
        return DiagWeights(
            w_pos=self.w_pos.detach().clone(),
            w_neg=self.w_neg.detach().clone(),
            v_pos=self.v_pos.detach().clone(),
            v_neg=self.v_neg.detach().clone(),
        )


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def _get_init_parameters(c: float, lmda: float) -> Tuple[float, float, float, float]:
    """
    Solve for uniform (v⁺, v⁻, u⁺, u⁻) from the (c, λ) parameterisation.

    At every coordinate d the invariants read:
        w_pos_d · v_pos_d − w_neg_d · v_neg_d = λ   (initial β)
        w_pos_d · w_neg_d + v_pos_d · v_neg_d = c   (geometry anchor)

    The symmetric solution (all four scalars equal per pair) is:
        v_pos = v_neg = √((c + λ)/2)
        w_pos = w_neg = √((c − λ)/2)

    Args:
        c:    Scale parameter (c ≥ |λ| required for real solution).
        lmda: Asymmetry parameter (λ = 0 gives symmetric init).

    Returns:
        (v_pos_val, v_neg_val, w_pos_val, w_neg_val) — scalar init values.

    Raises:
        ValueError: if c² < λ² (no real solution).
    """
    if c ** 2 < lmda ** 2:
        raise ValueError(
            f"Require c² ≥ λ²; got c={c}, λ={lmda}."
        )
    v = math.sqrt((c + lmda) / 2.0)
    u = math.sqrt((c - lmda) / 2.0)
    return v, v, u, u  # v_pos, v_neg, w_pos, w_neg


def make_init_weights(D: int, c: float, lmda: float) -> DiagWeights:
    """
    Construct uniform (c, λ)-initialisation weights for a D-dimensional DLN.

    All coordinates receive the same scalar values derived from
    ``_get_init_parameters(c, lmda)``.  This is the 'complex' initialisation
    used by the existing pretraining scripts.

    Args:
        D:    Input / parameter dimension.
        c:    Scale parameter (c² ≥ λ²).
        lmda: Asymmetry parameter (λ = 0 for symmetric init).

    Returns:
        DiagWeights with every coordinate set uniformly.
    """
    v_pos_val, v_neg_val, w_pos_val, w_neg_val = _get_init_parameters(c, lmda)
    return DiagWeights(
        w_pos=torch.full((D,), w_pos_val),
        w_neg=torch.full((D,), w_neg_val),
        v_pos=torch.full((D,), v_pos_val),
        v_neg=torch.full((D,), v_neg_val),
    )


# ---------------------------------------------------------------------------
# Teacher / data sampling helpers
# ---------------------------------------------------------------------------

def _sample_sparse_teacher(
    D: int,
    k: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample a k-sparse linear teacher β* ∈ ℝᴰ.

    Active coordinates are drawn uniformly at random from {0, …, D−1};
    each active coordinate receives a random sign ±1/√k.

    Args:
        D:         Input dimension.
        k:         Number of non-zero (active) coordinates.
        generator: Optional RNG for reproducibility.

    Returns:
        beta: shape (D,) float Tensor with exactly k non-zero entries.
    """
    if k <= 0 or k > D:
        raise ValueError(f"k must be in [1, D]; got k={k}, D={D}.")
    perm = torch.randperm(D, generator=generator)
    active = perm[:k]
    signs = torch.sign(torch.rand(k, generator=generator) - 0.5)
    beta = torch.zeros(D)
    beta[active] = signs / math.sqrt(k)
    return beta


def _sample_ft_teacher(
    beta_PT: torch.Tensor,
    D: int,
    k_shared: int,
    k_new: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample a fine-tuning teacher β_FT with controlled overlap with β_PT.

    β_FT has k_shared coordinates drawn from β_PT's active set and k_new
    coordinates drawn from the complement (inactive in β_PT).  Each active
    FT coordinate receives a fresh random ±1/√(k_shared + k_new) sign,
    independent of β_PT's signs.

    Args:
        beta_PT:   Pretraining teacher, shape (D,).
        D:         Input dimension.
        k_shared:  Number of β_FT active coords shared with β_PT's support.
        k_new:     Number of β_FT active coords NOT in β_PT's support.
        generator: Optional RNG.

    Returns:
        beta_FT: shape (D,) float Tensor.
    """
    k_FT = k_shared + k_new
    if k_FT == 0:
        return torch.zeros(D)

    pt_active = torch.nonzero(beta_PT, as_tuple=False).squeeze(-1)
    n_pt_active = pt_active.shape[0]

    if k_shared > n_pt_active:
        warnings.warn(
            f"k_shared={k_shared} > number of PT active coords ({n_pt_active}); "
            f"clamping k_shared to {n_pt_active}.",
            stacklevel=2,
        )
        k_shared = n_pt_active
        k_FT = k_shared + k_new

    # Shared coords: random subset of PT active set
    perm_shared = torch.randperm(n_pt_active, generator=generator)
    shared_coords = pt_active[perm_shared[:k_shared]]

    # New coords: random subset of PT inactive set
    inactive_mask = torch.ones(D, dtype=torch.bool)
    inactive_mask[pt_active] = False
    inactive_coords = torch.nonzero(inactive_mask, as_tuple=False).squeeze(-1)

    if k_new > inactive_coords.shape[0]:
        raise ValueError(
            f"k_new={k_new} exceeds number of PT-inactive coordinates "
            f"({inactive_coords.shape[0]})."
        )
    perm_new = torch.randperm(inactive_coords.shape[0], generator=generator)
    new_coords = inactive_coords[perm_new[:k_new]]

    all_active = torch.cat([shared_coords, new_coords])
    signs = torch.sign(torch.rand(k_FT, generator=generator) - 0.5)
    beta_FT = torch.zeros(D)
    beta_FT[all_active] = signs / math.sqrt(k_FT)
    return beta_FT


def _circular_sample(
    shape: Tuple[int, ...],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample rows drawn uniformly from the sphere defined by (1/D)||x||² = 1.

    Each row is drawn i.i.d. from N(0, I_D) then divided by
    sqrt(mean(x²)), matching the convention in the existing scripts.

    Args:
        shape:     (N, D) — number of samples × input dimension.
        generator: Optional RNG.

    Returns:
        X: shape (N, D) float Tensor.
    """
    X = torch.randn(*shape, generator=generator)
    norms = torch.sqrt(torch.mean(X ** 2, dim=-1, keepdim=True))
    return X / norms


# ---------------------------------------------------------------------------
# Core training primitive
# ---------------------------------------------------------------------------

def _train_to_convergence(
    weights_init: DiagWeights,
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.1,
    epochs: int = int(1e6),
    threshold: float = 1e-8,
    lr_tuning: bool = True,
    lr_decay: float = 0.1,
    lr_overflow_threshold: float = 100.0,
) -> Tuple[DiagWeights, List[float]]:
    """
    Run full-batch gradient descent on MSE loss until convergence or budget.

    Minimises L(β) = (1/N) ||X β − y||²  where β = w⁺∘v⁺ − w⁻∘v⁻.

    Adaptive LR: if the loss exceeds ``lr_overflow_threshold`` or is NaN,
    the learning rate is multiplied by ``lr_decay`` and training restarts from
    the same initial weights.  This mirrors the adaptive-LR logic in the
    existing ``train_one_task`` function.

    Args:
        weights_init:          Starting weight state.
        X:                     Data matrix, shape (N, D).
        y:                     Target vector, shape (N,).
        lr:                    Initial learning rate.
        epochs:                Maximum number of gradient steps.
        threshold:             Stop when train loss < threshold.
        lr_tuning:             Enable adaptive LR reduction on overflow.
        lr_decay:              Multiplicative LR reduction factor.
        lr_overflow_threshold: Loss value that triggers LR reduction.

    Returns:
        (final_weights, loss_curve) where ``loss_curve`` is a Python list of
        per-epoch float losses (length ≤ epochs).
    """
    min_lr = 1e-12
    current_lr = float(lr)

    # Outer loop: retry with reduced LR on overflow
    model: Optional[_DiagonalNet] = None
    loss_curve: List[float] = []

    while current_lr >= min_lr:
        model = _DiagonalNet(weights_init)
        optimizer = optim.SGD(list(model.parameters()), lr=current_lr, momentum=0.0)
        loss_curve = []
        final_loss = float("inf")
        overflow = False

        for _ in range(epochs):
            optimizer.zero_grad()
            loss_t = F.mse_loss(model(X), y)
            loss_t.backward()
            optimizer.step()

            final_loss = loss_t.item()
            loss_curve.append(final_loss)

            # Check for overflow before convergence test
            if lr_tuning and (math.isnan(final_loss) or final_loss > lr_overflow_threshold):
                current_lr *= lr_decay
                overflow = True
                break

            if final_loss < threshold:
                break

        if not overflow:
            # Exited via convergence or epoch budget — done
            break

    # If we exhausted the minimum LR without converging, return what we have
    assert model is not None
    return model.get_weights(), loss_curve


# ---------------------------------------------------------------------------
# Geometry diagnostic: k_d values
# ---------------------------------------------------------------------------

def compute_k_d(weights: DiagWeights) -> torch.Tensor:
    """
    Compute per-coordinate geometry parameter k_d from weight values.

    In a diagonal linear network the implicit bias of gradient descent is
    determined by the geometry of the level sets of the conserved quantities.
    k_d characterises this geometry at coordinate d:

        c_d       = w⁺_d · w⁻_d + v⁺_d · v⁻_d
        δ⁺_d      = (v⁺_d)² − (w⁺_d)²
        δ⁻_d      = (v⁻_d)² − (w⁻_d)²
        k_d       = (δ⁺_d − δ⁻_d)² + 4 c_d²

    These values are evaluated *after* reset-mode rescaling and *before*
    training begins; they are recorded as diagnostics that characterise
    the implicit prior on β introduced by the initialisation geometry.

    Args:
        weights: DiagWeights at the start of the stage (post-reset).

    Returns:
        k_d: shape (D,) float Tensor.
    """
    c_d = weights.w_pos * weights.w_neg + weights.v_pos * weights.v_neg
    delta_plus = weights.v_pos ** 2 - weights.w_pos ** 2
    delta_minus = weights.v_neg ** 2 - weights.w_neg ** 2
    return (delta_plus - delta_minus) ** 2 + 4.0 * c_d ** 2


# ---------------------------------------------------------------------------
# Reset modes A and B — strictly separated functions
# ---------------------------------------------------------------------------

def _apply_reset_mode_A(weights: DiagWeights, gamma: float) -> DiagWeights:
    """
    Mode A reset (Anguita-style): rebalance hidden representation, zero β.

    Transformation:
        w±(0) ← w⁺_prev + w⁻_prev   (hidden representation preserved)
        v±(0) ← γ                    (output reset ⟹ β_init = γ²−γ² = 0)

    This is the standard PT→FT transition used by the existing codebase.
    With γ = 0 the network function is exactly zero at the start of the stage.

    Args:
        weights: Weight state at end of the previous stage.
        gamma:   Scalar re-initialisation value for v parameters (γ ≥ 0).

    Returns:
        New DiagWeights with zeroed network function.
    """
    D = weights.w_pos.shape[0]
    rebalanced_w = weights.w_pos + weights.w_neg
    return DiagWeights(
        w_pos=rebalanced_w.clone(),
        w_neg=rebalanced_w.clone(),
        v_pos=torch.full((D,), float(gamma)),
        v_neg=torch.full((D,), float(gamma)),
    )


def _apply_reset_mode_B(
    weights: DiagWeights,
    a: float = 1.0,
    b: float = 1.0,
) -> DiagWeights:
    """
    Mode B reset (no reset): optional layerwise rescaling, β_init = a·b·β_prev.

    Applies:
        w± ← a · w±
        v± ← b · v±

    With a = b = 1 (defaults) this is an exact continuation: no modification.
    The network function at the start of the stage is a·b·β_prev ≠ 0 in general
    (unlike Mode A where it is identically zero).

    Args:
        weights: Weight state at end of the previous stage.
        a:       Multiplicative rescaling for the w-layer (a > 0).
        b:       Multiplicative rescaling for the v-layer (b > 0).

    Returns:
        New DiagWeights with rescaled parameters.

    Raises:
        ValueError: if a ≤ 0 or b ≤ 0.
    """
    if a <= 0 or b <= 0:
        raise ValueError(
            f"Rescaling factors must be strictly positive; got a={a}, b={b}."
        )
    return DiagWeights(
        w_pos=a * weights.w_pos,
        w_neg=a * weights.w_neg,
        v_pos=b * weights.v_pos,
        v_neg=b * weights.v_neg,
    )


def _apply_reset(weights: DiagWeights, reset_config: dict) -> DiagWeights:
    """
    Dispatch to Mode A or Mode B based on ``reset_config['mode']``.

    ``reset_config`` keys
    ---------------------
    mode  (str):   'A' or 'B' — selects the reset variant.
    gamma (float): Used only for Mode A.
    a     (float): Used only for Mode B (default 1.0).
    b     (float): Used only for Mode B (default 1.0).

    Args:
        weights:      Weight state from the end of the previous stage.
        reset_config: Dict as described above.

    Returns:
        Post-reset DiagWeights, ready for the next training stage.

    Raises:
        ValueError: for unknown mode string.
    """
    mode = reset_config["mode"]
    if mode == "A":
        return _apply_reset_mode_A(weights, gamma=float(reset_config["gamma"]))
    elif mode == "B":
        return _apply_reset_mode_B(
            weights,
            a=float(reset_config.get("a", 1.0)),
            b=float(reset_config.get("b", 1.0)),
        )
    else:
        raise ValueError(f"Unknown reset mode '{mode}'; expected 'A' or 'B'.")


# ---------------------------------------------------------------------------
# Retain / forget data generation
# ---------------------------------------------------------------------------

def make_retain_forget_data(
    beta_FT: torch.Tensor,
    X: torch.Tensor,
    forget_fraction: float,
    seed: int = 0,
    include_noise: bool = False,
) -> Dict:
    """
    Partition the active coordinates of β_FT into retain and forget sets.

    Only *active* coordinates (β_FT_d ≠ 0) are eligible; inactive coordinates
    belong to neither partition.  The forget set S_f is a random subset of the
    active coordinates of size ⌊forget_fraction · |active|⌋.

    Args:
        beta_FT:         Fine-tuning teacher, shape (D,).
        X:               Data matrix, shape (N, D).
        forget_fraction: Fraction of active coords assigned to the forget set,
                         ∈ (0, 1).  At least one coordinate is always retained.
        seed:            Integer seed for the forget-set RNG.
        include_noise:   If True, also sample β_noise ~ N(0, I/D) and
                         compute y_noise = X @ β_noise.

    Returns:
        Dict with keys:
            P_r    (Tensor D):  Binary retain mask (1 on retain coords, 0 elsewhere).
            P_f    (Tensor D):  Binary forget mask  (1 on forget coords, 0 elsewhere).
            beta_r (Tensor D):  P_r ∘ β_FT  (retain teacher).
            beta_f (Tensor D):  P_f ∘ β_FT  (forget teacher).
            y_r    (Tensor N):  X @ β_r.
            y_f    (Tensor N):  X @ β_f.
        And if include_noise=True:
            beta_noise (Tensor D): sample ~ N(0, I/D).
            y_noise    (Tensor N): X @ β_noise.

    Raises:
        ValueError: if β_FT has no active coordinates.
    """
    D = beta_FT.shape[0]
    active_coords = torch.nonzero(beta_FT, as_tuple=False).squeeze(-1)
    n_active = active_coords.shape[0]

    if n_active == 0:
        raise ValueError(
            "beta_FT has no active (non-zero) coordinates; cannot partition."
        )

    n_forget = max(1, int(math.floor(forget_fraction * n_active)))
    if n_forget >= n_active:
        warnings.warn(
            f"forget_fraction={forget_fraction} would forget all {n_active} active "
            f"coords; clamping to n_forget={n_active - 1} to preserve at least one "
            f"retain coordinate.",
            stacklevel=2,
        )
        n_forget = n_active - 1

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n_active, generator=gen)
    forget_idx = active_coords[perm[:n_forget]]
    retain_idx = active_coords[perm[n_forget:]]

    P_f = torch.zeros(D)
    P_f[forget_idx] = 1.0
    P_r = torch.zeros(D)
    P_r[retain_idx] = 1.0

    beta_r = P_r * beta_FT
    beta_f = P_f * beta_FT

    result: Dict = dict(
        P_r=P_r,
        P_f=P_f,
        beta_r=beta_r,
        beta_f=beta_f,
        y_r=X @ beta_r,
        y_f=X @ beta_f,
    )

    if include_noise:
        gen_noise = torch.Generator()
        gen_noise.manual_seed(seed + 10_000)
        beta_noise = torch.randn(D, generator=gen_noise) / math.sqrt(D)
        result["beta_noise"] = beta_noise
        result["y_noise"] = X @ beta_noise

    return result


# ---------------------------------------------------------------------------
# Effective teacher computation
# ---------------------------------------------------------------------------

def compute_effective_teacher(
    loss_type: str,
    beta_r: torch.Tensor,
    beta_f: torch.Tensor,
    c_r: float,
    c_f: float,
    beta_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute β_eff, the effective teacher for Stage 2 MSE training.

    Any unlearning loss of the form
        L(β) = Σⱼ (aⱼ/2) ||X⊤β − X⊤βⱼ||²,    A = Σⱼ aⱼ > 0
    reduces algebraically to standard MSE against β_eff = (Σⱼ aⱼ βⱼ) / A.
    This function encodes that reduction for each supported loss type.

    Supported loss types and their effective teachers
    --------------------------------------------------
    retain_only:
        β_eff = β_r
        Always stable.

    signed:
        β_eff = (c_r · β_r − c_f · β_f) / (c_r − c_f)
        Stable only when c_r > c_f; a warning is issued otherwise.

    forget_to_zero:
        β_eff = [c_r / (c_r + c_f)] · β_r
        Always stable (the forget objective pushes the forget coords to 0).

    forget_to_noise:
        β_eff = (c_r · β_r + c_f · β_noise) / (c_r + c_f)
        Always stable; requires ``beta_noise`` to be provided.

    Args:
        loss_type:  One of 'retain_only', 'signed', 'forget_to_zero',
                    'forget_to_noise'.
        beta_r:     Retain teacher P_r ∘ β_FT, shape (D,).
        beta_f:     Forget teacher P_f ∘ β_FT, shape (D,).
        c_r:        Retain coefficient (positive scalar).
        c_f:        Forget coefficient (positive scalar).
        beta_noise: Noise teacher ~ N(0, I/D); required only for
                    'forget_to_noise'.

    Returns:
        beta_eff: shape (D,) float Tensor.

    Raises:
        ValueError: for an unknown loss_type or missing beta_noise.
    """
    if loss_type == "retain_only":
        return beta_r.clone()

    elif loss_type == "signed":
        denom = c_r - c_f
        if denom <= 0:
            warnings.warn(
                f"'signed' loss requires c_r > c_f for stability, "
                f"but c_r={c_r} ≤ c_f={c_f}.  The effective teacher may be "
                f"ill-conditioned and results may be unreliable.",
                stacklevel=2,
            )
        return (c_r * beta_r - c_f * beta_f) / denom

    elif loss_type == "forget_to_zero":
        A = c_r + c_f
        return (c_r / A) * beta_r

    elif loss_type == "forget_to_noise":
        if beta_noise is None:
            raise ValueError(
                "'forget_to_noise' requires beta_noise to be provided."
            )
        A = c_r + c_f
        return (c_r * beta_r + c_f * beta_noise) / A

    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Expected one of: "
            "'retain_only', 'signed', 'forget_to_zero', 'forget_to_noise'."
        )


# ---------------------------------------------------------------------------
# Stage 2: Unlearning
# ---------------------------------------------------------------------------

def run_stage2(
    weights_prev: DiagWeights,
    X: torch.Tensor,
    beta_eff: torch.Tensor,
    reset_config: dict,
    lr: float = 0.1,
    epochs: int = int(1_000_000),
    threshold: float = 1e-8,
) -> Tuple[DiagWeights, torch.Tensor, dict]:
    """
    Stage 2: Unlearning via MSE training against a pre-computed effective teacher.

    ``beta_eff`` MUST be computed by ``compute_effective_teacher`` before this
    function is called.  The training loop contains no loss-type logic — it
    trains purely on MSE against X @ beta_eff.  This decoupling ensures that
    Stage 2 is independent of the unlearning loss choice.

    The reset mode (A or B) is applied to ``weights_prev`` before training;
    k_d geometry parameters are computed from the post-reset, pre-training
    weights and stored as diagnostics.

    Args:
        weights_prev:  DiagWeights at end of Stage 1.
        X:             Data matrix for Stage 2, shape (N_2, D).
        beta_eff:      Effective teacher β_eff, shape (D,).  Must be computed
                       externally via ``compute_effective_teacher``.
        reset_config:  Dict controlling the reset variant:
                           mode  (str):   'A' or 'B'
                           gamma (float): scalar γ for Mode A
                           a     (float): w-layer scale for Mode B (default 1.0)
                           b     (float): v-layer scale for Mode B (default 1.0)
        lr:            Learning rate for SGD.
        epochs:        Maximum training epochs.
        threshold:     Convergence criterion on training loss.

    Returns:
        (weights_final, beta_unlearn, diagnostics) where:
            weights_final  : DiagWeights at end of Stage 2.
            beta_unlearn   : Tensor (D,), effective β after Stage 2.
            diagnostics    : dict with:
                'k_d'        : Tensor (D,) — geometry params at stage start.
                'loss_curve' : list[float] — per-epoch training losses.
    """
    # Step 1: apply reset (Mode A or B) — strictly via separate helpers
    weights_init = _apply_reset(weights_prev, reset_config)

    # Step 2: compute k_d diagnostics BEFORE any gradient update
    k_d = compute_k_d(weights_init)

    # Step 3: build targets and train to convergence
    y = X @ beta_eff
    weights_final, loss_curve = _train_to_convergence(
        weights_init=weights_init,
        X=X,
        y=y,
        lr=lr,
        epochs=epochs,
        threshold=threshold,
    )

    # Step 4: extract effective β
    beta_unlearn = weights_final.beta()

    return weights_final, beta_unlearn, {"k_d": k_d, "loss_curve": loss_curve}


# ---------------------------------------------------------------------------
# Stage 3: Relearning sweep
# ---------------------------------------------------------------------------

def run_stage3(
    weights_prev: DiagWeights,
    X_forget: torch.Tensor,
    beta_f: torch.Tensor,
    P_f: torch.Tensor,
    alpha3_values: List[float],
    reset_config: dict,
    D: int,
    lr: float = 0.1,
    epochs: int = int(1_000_000),
    threshold: float = 1e-8,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Stage 3: Relearning sweep over forget-set sample counts.

    For each α₃ in ``alpha3_values``:
        1. Apply the Stage 3 reset to ``weights_prev`` (independent per α₃).
        2. Use the first N₃ = ⌊α₃ · D⌋ rows of ``X_forget``.
        3. Train to convergence on MSE against X_forget[:N₃] @ β_f.
        4. Record the forget relearning error:
               E_f_relearn(α₃) = ||P_f ∘ (β_relearn − β_f)||²

    k_d values are computed at the start of each sub-run (post-reset,
    pre-training) and stored as per-α₃ diagnostics.

    Args:
        weights_prev:   DiagWeights at end of Stage 2.
        X_forget:       All available forget-set rows, shape (N_max, D).
                        Must satisfy N_max ≥ ⌊max(alpha3_values) · D⌋.
        beta_f:         True forget teacher P_f ∘ β_FT, shape (D,).
        P_f:            Binary forget mask, shape (D,).
        alpha3_values:  List of α₃ = N₃/D values to sweep.
        reset_config:   Same structure as Stage 2's reset_config.
        D:              Input dimension (used to compute N₃ = ⌊α₃ · D⌋).
        lr:             Learning rate.
        epochs:         Maximum epochs per α₃.
        threshold:      Convergence threshold.

    Returns:
        (relearn_errors, per_alpha_diagnostics) where:
            relearn_errors (np.ndarray, shape (len(alpha3_values),)):
                E_f_relearn(α₃) for each α₃; NaN where N₃ = 0.
            per_alpha_diagnostics (list[dict], one entry per α₃):
                Each dict has keys:
                    'alpha_3'    : float — the α₃ value.
                    'N_3'        : int — number of forget samples used.
                    'k_d'        : Tensor (D,) or None — geometry params.
                    'loss_curve' : list[float] — training loss curve.
    """
    relearn_errors = np.full(len(alpha3_values), float("nan"))
    per_alpha_diagnostics: List[dict] = []

    for idx, alpha_3 in enumerate(alpha3_values):
        N_3 = int(math.floor(alpha_3 * D))

        if N_3 == 0:
            warnings.warn(
                f"alpha_3={alpha_3} gives N_3=0 for D={D}; "
                f"skipping this entry (relearn_error set to NaN).",
                stacklevel=2,
            )
            per_alpha_diagnostics.append(
                {"alpha_3": alpha_3, "N_3": 0, "k_d": None, "loss_curve": []}
            )
            continue

        if N_3 > X_forget.shape[0]:
            raise ValueError(
                f"alpha_3={alpha_3} requires N_3={N_3} forget samples, "
                f"but X_forget has only {X_forget.shape[0]} rows. "
                f"Increase the pre-allocated X_forget size."
            )

        # Apply reset independently for each α₃ (same weights_prev every time)
        weights_init = _apply_reset(weights_prev, reset_config)

        # k_d diagnostic BEFORE any gradient update
        k_d = compute_k_d(weights_init)

        # Slice data and build targets
        X_sub = X_forget[:N_3]
        y_sub = X_sub @ beta_f

        # Train to convergence
        weights_final, loss_curve = _train_to_convergence(
            weights_init=weights_init,
            X=X_sub,
            y=y_sub,
            lr=lr,
            epochs=epochs,
            threshold=threshold,
        )

        # Forget relearning error: ||P_f ∘ (β_relearn − β_f)||²
        beta_relearn = weights_final.beta()
        err = torch.sum(P_f * (beta_relearn - beta_f) ** 2).item()
        relearn_errors[idx] = err

        per_alpha_diagnostics.append(
            {
                "alpha_3": alpha_3,
                "N_3": N_3,
                "k_d": k_d,
                "loss_curve": loss_curve,
            }
        )

    return relearn_errors, per_alpha_diagnostics


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class UnlearningConfig:
    """
    Complete specification of a single three-stage unlearning experiment.

    All ratio fields (rho_*, alpha_*) are dimensionless fractions of D;
    integer sample counts are computed at runtime as floor(ratio × D).

    Teacher / dimension parameters
    --------------------------------
    D              : Input dimension.
    rho_PT         : Fraction of D active in the pretraining teacher β_PT.
    rho_shared_FT  : Fraction of D shared between β_PT and β_FT.
    rho_new_FT     : Fraction of D exclusive to β_FT (not in β_PT).
    forget_fraction: Fraction of β_FT active coords assigned to the forget set.

    Stage 1 parameters (Pretraining on β_PT)
    -----------------------------------------
    c_PT           : Scale parameter c for (c, λ) initialisation (c² ≥ λ²).
    lambda_PT      : Asymmetry parameter λ for (c, λ) initialisation.
    alpha_PT       : N_PT / D — sample ratio for Stage 1.

    Stage 2 parameters (Unlearning)
    --------------------------------
    loss_type      : 'retain_only' | 'signed' | 'forget_to_zero' |
                     'forget_to_noise'.
    c_r            : Retain coefficient for β_eff computation (positive).
    c_f            : Forget coefficient for β_eff computation (positive).
    alpha_2        : N_2 / D — sample ratio for Stage 2.
    stage2_reset_mode : 'A' or 'B'.
    stage2_gamma   : γ for Mode A reset in Stage 2.
    stage2_a       : w-layer rescaling for Mode B in Stage 2 (default 1.0).
    stage2_b       : v-layer rescaling for Mode B in Stage 2 (default 1.0).

    Stage 3 parameters (Relearning)
    --------------------------------
    alpha3_values  : List of N₃/D values to sweep.
    stage3_reset_mode : 'A' or 'B'.
    stage3_gamma   : γ for Mode A reset in Stage 3.
    stage3_a       : w-layer rescaling for Mode B in Stage 3 (default 1.0).
    stage3_b       : v-layer rescaling for Mode B in Stage 3 (default 1.0).

    Training hyperparameters (shared across all stages)
    ----------------------------------------------------
    lr             : SGD learning rate (default 0.1).
    epochs         : Maximum epochs per stage (default 1e6).
    threshold      : Convergence threshold on training loss (default 1e-8).

    Reproducibility
    ----------------
    n_seeds        : Number of seeds for ``run_experiment_over_seeds``.
    seed           : Base random seed for this run.
    """

    # Teacher / dimension
    D: int
    rho_PT: float
    rho_shared_FT: float
    rho_new_FT: float
    forget_fraction: float

    # Stage 1
    c_PT: float
    lambda_PT: float
    alpha_PT: float

    # Stage 2
    loss_type: str
    c_r: float
    c_f: float
    alpha_2: float
    stage2_reset_mode: str
    stage2_gamma: float
    stage2_a: float = 1.0
    stage2_b: float = 1.0

    # Stage 3
    alpha3_values: List[float] = field(default_factory=list)
    stage3_reset_mode: str = "A"
    stage3_gamma: float = 0.0
    stage3_a: float = 1.0
    stage3_b: float = 1.0

    # Training hyperparameters
    lr: float = 0.1
    epochs: int = int(1_000_000)
    threshold: float = 1e-8

    # Reproducibility
    n_seeds: int = 1
    seed: int = 0

    def stage2_reset_config(self) -> dict:
        """Return the reset_config dict consumed by run_stage2 / _apply_reset."""
        return {
            "mode": self.stage2_reset_mode,
            "gamma": self.stage2_gamma,
            "a": self.stage2_a,
            "b": self.stage2_b,
        }

    def stage3_reset_config(self) -> dict:
        """Return the reset_config dict consumed by run_stage3 / _apply_reset."""
        return {
            "mode": self.stage3_reset_mode,
            "gamma": self.stage3_gamma,
            "a": self.stage3_a,
            "b": self.stage3_b,
        }


# ---------------------------------------------------------------------------
# Top-level runners
# ---------------------------------------------------------------------------

def run_experiment(config: UnlearningConfig) -> dict:
    """
    Execute the full three-stage unlearning pipeline for a single (config, seed).

    Stages
    ------
    Stage 1 — Pretrain on β_PT with N_PT = ⌊α_PT · D⌋ samples.
    Stage 2 — Unlearn via MSE against β_eff with N_2 = ⌊α_2 · D⌋ samples.
    Stage 3 — Relearn on forget-set data, sweeping α₃ values.

    Teacher construction
    --------------------
    β_PT is a ρ_PT-sparse teacher with k_PT = ⌊ρ_PT · D⌋ active coords.
    β_FT is built with k_shared = ⌊ρ_shared_FT · D⌋ coords from β_PT's support
    and k_new = ⌊ρ_new_FT · D⌋ new coords.
    P_r / P_f partition the active coords of β_FT according to forget_fraction.
    β_eff is computed *once* from (loss_type, β_r, β_f, c_r, c_f) before any
    training function is called.

    Metrics recorded
    ----------------
    beta_unlearn   : β after Stage 2.
    E_r_unlearn    : ||P_r ∘ (β_unlearn − β_r)||²  (retain error after unlearning).
    E_f_unlearn    : ||P_f ∘ β_unlearn||²           (forget suppression).
    E_f_relearn    : np.ndarray of relearning errors indexed by α₃.
    k_d_stage2     : k_d geometry params at start of Stage 2.
    k_d_stage3     : k_d geometry params at start of Stage 3 (common to all α₃).

    Args:
        config: An ``UnlearningConfig`` fully specifying the experiment.

    Returns:
        dict with keys:
            'beta_PT'           : Tensor (D,) — pretraining teacher.
            'beta_FT'           : Tensor (D,) — fine-tuning teacher.
            'beta_r'            : Tensor (D,) — retain teacher P_r ∘ β_FT.
            'beta_f'            : Tensor (D,) — forget teacher P_f ∘ β_FT.
            'P_r'               : Tensor (D,) — binary retain mask.
            'P_f'               : Tensor (D,) — binary forget mask.
            'beta_eff'          : Tensor (D,) — effective teacher for Stage 2.
            'weights_stage1'    : DiagWeights — weights at end of Stage 1.
            'weights_stage2'    : DiagWeights — weights at end of Stage 2.
            'beta_unlearn'      : Tensor (D,) — β after Stage 2.
            'E_r_unlearn'       : float — retain error after unlearning.
            'E_f_unlearn'       : float — forget suppression after unlearning.
            'E_f_relearn'       : np.ndarray (len(alpha3_values),).
            'k_d_stage2'        : Tensor (D,) — geometry params at Stage 2 start.
            'k_d_stage3'        : Tensor (D,) or None — geometry params at Stage 3 start.
            'loss_curve_stage1' : list[float].
            'loss_curve_stage2' : list[float].
            'stage3_diagnostics': list[dict] — per-α₃ diagnostics.
    """
    D = config.D
    seed = config.seed

    # Independent generators — one role each for reproducibility
    gen_pt_teacher   = torch.Generator().manual_seed(seed)
    gen_ft_teacher   = torch.Generator().manual_seed(seed + 1)
    gen_pt_data      = torch.Generator().manual_seed(seed + 2)
    gen_s2_data      = torch.Generator().manual_seed(seed + 3)
    gen_s3_data      = torch.Generator().manual_seed(seed + 4)
    seed_forget_split = seed + 5  # int seed passed to make_retain_forget_data

    # ------------------------------------------------------------------
    # Teacher construction
    # ------------------------------------------------------------------
    k_PT     = max(1, int(math.floor(config.rho_PT * D)))
    k_shared = max(0, int(math.floor(config.rho_shared_FT * D)))
    k_new    = max(0, int(math.floor(config.rho_new_FT * D)))

    beta_PT = _sample_sparse_teacher(D, k_PT, generator=gen_pt_teacher)
    beta_FT = _sample_ft_teacher(
        beta_PT=beta_PT,
        D=D,
        k_shared=k_shared,
        k_new=k_new,
        generator=gen_ft_teacher,
    )

    # ------------------------------------------------------------------
    # Stage 1: Pretraining on β_PT
    # ------------------------------------------------------------------
    N_PT = max(1, int(math.floor(config.alpha_PT * D)))
    X_PT = _circular_sample((N_PT, D), generator=gen_pt_data)
    y_PT = X_PT @ beta_PT

    weights_init_s1 = make_init_weights(D, c=config.c_PT, lmda=config.lambda_PT)
    weights_stage1, loss_curve_s1 = _train_to_convergence(
        weights_init=weights_init_s1,
        X=X_PT,
        y=y_PT,
        lr=config.lr,
        epochs=config.epochs,
        threshold=config.threshold,
    )

    # ------------------------------------------------------------------
    # Retain / forget partition and Stage 2 data
    # ------------------------------------------------------------------
    N_2 = max(1, int(math.floor(config.alpha_2 * D)))
    X_2 = _circular_sample((N_2, D), generator=gen_s2_data)

    include_noise = (config.loss_type == "forget_to_noise")
    retain_forget = make_retain_forget_data(
        beta_FT=beta_FT,
        X=X_2,
        forget_fraction=config.forget_fraction,
        seed=seed_forget_split,
        include_noise=include_noise,
    )
    P_r = retain_forget["P_r"]
    P_f = retain_forget["P_f"]
    beta_r = retain_forget["beta_r"]
    beta_f = retain_forget["beta_f"]
    beta_noise = retain_forget.get("beta_noise", None)

    # ------------------------------------------------------------------
    # Effective teacher — computed BEFORE calling any training function
    # ------------------------------------------------------------------
    beta_eff = compute_effective_teacher(
        loss_type=config.loss_type,
        beta_r=beta_r,
        beta_f=beta_f,
        c_r=config.c_r,
        c_f=config.c_f,
        beta_noise=beta_noise,
    )

    # ------------------------------------------------------------------
    # Stage 2: Unlearning
    # ------------------------------------------------------------------
    weights_stage2, beta_unlearn, diag_s2 = run_stage2(
        weights_prev=weights_stage1,
        X=X_2,
        beta_eff=beta_eff,
        reset_config=config.stage2_reset_config(),
        lr=config.lr,
        epochs=config.epochs,
        threshold=config.threshold,
    )

    E_r_unlearn = torch.sum(P_r * (beta_unlearn - beta_r) ** 2).item()
    E_f_unlearn = torch.sum(P_f * beta_unlearn ** 2).item()

    # ------------------------------------------------------------------
    # Stage 3: Relearning sweep
    # ------------------------------------------------------------------
    if not config.alpha3_values:
        relearn_errors    = np.array([], dtype=float)
        stage3_diagnostics: List[dict] = []
        k_d_stage3: Optional[torch.Tensor] = None
    else:
        max_N3 = max(int(math.floor(a * D)) for a in config.alpha3_values)
        max_N3 = max(max_N3, 1)
        X_forget_all = _circular_sample((max_N3, D), generator=gen_s3_data)

        relearn_errors, stage3_diagnostics = run_stage3(
            weights_prev=weights_stage2,
            X_forget=X_forget_all,
            beta_f=beta_f,
            P_f=P_f,
            alpha3_values=config.alpha3_values,
            reset_config=config.stage3_reset_config(),
            D=D,
            lr=config.lr,
            epochs=config.epochs,
            threshold=config.threshold,
        )
        # All α₃ sub-runs share the same weights_prev and same reset, so k_d
        # is identical across runs; take from the first non-None entry.
        k_d_stage3 = next(
            (d["k_d"] for d in stage3_diagnostics if d["k_d"] is not None),
            None,
        )

    return {
        "beta_PT":            beta_PT,
        "beta_FT":            beta_FT,
        "beta_r":             beta_r,
        "beta_f":             beta_f,
        "P_r":                P_r,
        "P_f":                P_f,
        "beta_eff":           beta_eff,
        "weights_stage1":     weights_stage1,
        "weights_stage2":     weights_stage2,
        "beta_unlearn":       beta_unlearn,
        "E_r_unlearn":        E_r_unlearn,
        "E_f_unlearn":        E_f_unlearn,
        "E_f_relearn":        relearn_errors,
        "k_d_stage2":         diag_s2["k_d"],
        "k_d_stage3":         k_d_stage3,
        "loss_curve_stage1":  loss_curve_s1,
        "loss_curve_stage2":  diag_s2["loss_curve"],
        "stage3_diagnostics": stage3_diagnostics,
    }


def run_experiment_over_seeds(
    config: UnlearningConfig,
    n_seeds: int,
) -> dict:
    """
    Run ``run_experiment`` for n_seeds consecutive seeds and aggregate results.

    Seed i uses ``config.seed + i`` as the effective seed; all other fields
    of ``config`` are unchanged.

    Scalar metrics (E_r_unlearn, E_f_unlearn) and array metrics (E_f_relearn)
    are aggregated as mean ± standard error across seeds.  The raw output from
    the last seed is stored under 'last_seed_raw' for reference (teachers,
    masks, weight snapshots, etc.).

    Args:
        config:  Base ``UnlearningConfig``; config.seed is the starting seed.
        n_seeds: Number of independent seeds to run.

    Returns:
        dict with the following structure:
            'E_r_unlearn'       : {'mean': float, 'se': float}
            'E_f_unlearn'       : {'mean': float, 'se': float}
            'E_f_relearn'       : {'mean': np.ndarray, 'se': np.ndarray}
                                  (element-wise over alpha3_values)
            'n_seeds'           : int
            'last_seed_raw'     : dict — full result from the last seed run.
    """
    results: List[dict] = []
    for i in range(n_seeds):
        seed_config = replace(config, seed=config.seed + i)
        results.append(run_experiment(seed_config))

    aggregated: dict = {"n_seeds": n_seeds}

    # Scalar metrics
    for key in ("E_r_unlearn", "E_f_unlearn"):
        vals = np.array([r[key] for r in results], dtype=float)
        aggregated[key] = {
            "mean": float(vals.mean()),
            "se":   float(vals.std(ddof=1) / math.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
        }

    # Array metric: E_f_relearn — shape (n_seeds, len(alpha3_values))
    if config.alpha3_values:
        mat = np.stack([r["E_f_relearn"] for r in results], axis=0)
        aggregated["E_f_relearn"] = {
            "mean": mat.mean(axis=0),
            "se":   mat.std(axis=0, ddof=1) / math.sqrt(n_seeds) if n_seeds > 1
                    else np.zeros(mat.shape[1]),
        }
    else:
        aggregated["E_f_relearn"] = {"mean": np.array([]), "se": np.array([])}

    aggregated["last_seed_raw"] = results[-1]
    return aggregated
