# Phase 1 Validation Figures

This document explains the three validation overlay figures generated from the Phase 1 sweeps with the updated fixed-point stopping logic.

---

## Background: The Diagonal Network

The experiments train a **diagonal network** to learn a sparse teacher vector β* ∈ ℝ^d (d=1000).

The network parameterizes the learnable weights as:
```
β̂ᵢ = w_pos_i · v_pos_i - w_neg_i · v_neg_i
```

The **initialization** is controlled by two parameters per coordinate:
- **λ (lambda):** Controls the initial "bias" (w_pos² - v_pos²)
- **c:** Controls the initial "scale" (w_pos · w_neg + v_pos · v_neg)

The key insight is that **smaller c leads to faster convergence** toward the correct solution, while larger c can cause the network to get stuck.

### What is k?

The quantity **k = (2c)²** appears in the replica theory. It controls how "mobile" each coordinate is during gradient descent:
- **Small k (small c):** Coordinate can move quickly → good for learning
- **Large k (large c):** Coordinate is "sluggish" → slow to learn

---

## Figure 1: Step 1 Validation (Mixture-k)

**File:** `figures/validation_step1_phase1/step1_validation_overlay.png`

### What it plots

| Axis | Variable | Description |
|------|----------|-------------|
| X | α = n_train / d | Sample complexity ratio (training samples / dimension) |
| Y | Parameter MSE (dB) | 10·log₁₀(‖β̂ - β*‖²) — lower is better |

### Lines

| Line | Description |
|------|-------------|
| **Empirical π_A=0.1** | 10% of coordinates use c_A=0.001, 90% use c_B=0.5 |
| **Empirical π_A=0.5** | 50% use c_A, 50% use c_B |
| **Empirical π_A=0.9** | 90% use c_A, 10% use c_B |
| **Replica (dashed)** | Theoretical prediction |

### Experimental Setup

- **Teacher:** Bernoulli-Gaussian with sparsity ρ=0.04
- **Initialization:** "Mixture mode" — each coordinate randomly assigned to group A or B
  - c_A = 0.001 (small → fast learning)
  - c_B = 0.5 (large → slow learning)
  - π_A = probability of being in group A

### Key Insight

Higher π_A (more coordinates with small c) → better performance at high α because more coordinates can converge quickly.

---

## Figure 2: Step 2 Validation (Support-Conditioned k)

**File:** `figures/validation_step2_phase1/step2_validation_overlay.png`

### What it plots

| Axis | Variable | Description |
|------|----------|-------------|
| X | α = n_train / d | Sample complexity ratio |
| Y | Parameter MSE (dB) | 10·log₁₀(‖β̂ - β*‖²) |

### Lines

| Line | Case | c_nz | c_z | Description |
|------|------|------|-----|-------------|
| **Green (good)** | good | 0.001 | 0.5 | Small c on support, large c off-support |
| **Red (bad)** | bad | 0.5 | 0.001 | Large c on support, small c off-support |

### Variable Definitions

| Variable | Meaning |
|----------|---------|
| **c_nz** | c value for coordinates where β*ᵢ ≠ 0 (the **support** / nonzero entries) |
| **c_z** | c value for coordinates where β*ᵢ = 0 (the **off-support** / zero entries) |
| **Support** | The set of coordinates where the true teacher is nonzero: S = {i : β*ᵢ ≠ 0} |

### Experimental Setup

- **Teacher:** Bernoulli-Gaussian with sparsity ρ=0.04 (so ~40 nonzero coordinates out of 1000)
- **Initialization:** "Support mode" — c value depends on whether β*ᵢ is zero or not
  - This is an **oracle** initialization that knows the teacher's support

### Key Insight

The "good" case (c_nz < c_z) dramatically outperforms the "bad" case because:
- Small c on support → the ~40 signal coordinates can move quickly to match β*
- Large c off-support → the ~960 zero coordinates stay near zero (which is correct)

The "bad" case has it backwards: signal coordinates are sluggish while zero coordinates are mobile (and can drift to wrong values).

---

## Figure 3: Step 3 Validation (PT+FT Oracle)

**File:** `figures/validation_step3_phase1/step3_validation_overlay.png`

### What it plots

| Axis | Variable | Description |
|------|----------|-------------|
| X | α = n_train / d | Fine-tuning sample complexity |
| Y | Parameter MSE (dB) | 10·log₁₀(‖β̂ - β*_ft‖²) |

### Lines

| Line | ω (omega) | Description |
|------|-----------|-------------|
| **ω=0.0** | 0% overlap | FT support completely disjoint from PT support |
| **ω=0.5** | 50% overlap | Half of FT support overlaps with PT support |
| **ω=1.0** | 100% overlap | FT support is subset of PT support |

### Variable Definitions

| Variable | Meaning |
|----------|---------|
| **ω (omega)** | Overlap fraction = \|S_pt ∩ S_ft\| / \|S_ft\| |
| **S_pt** | Pre-training teacher support (where β*_pt ≠ 0) |
| **S_ft** | Fine-tuning teacher support (where β*_ft ≠ 0) |
| **ρ_pt = 0.10** | Pre-training sparsity (100 nonzero coordinates) |
| **ρ_ft = 0.04** | Fine-tuning sparsity (40 nonzero coordinates) |
| **a_pt = 1.0** | Pre-training teacher amplitude |

### Experimental Setup

- **Pre-training teacher:** Deterministic with ρ_pt=0.10 sparsity, amplitude a_pt=1.0
- **Fine-tuning teacher:** Bernoulli-Gaussian with ρ_ft=0.04 sparsity
- **Initialization:** Oracle PT+FT mapping — network is initialized as if it perfectly learned β*_pt, then mapped to FT initialization using the Cosyne formula

### Key Insight

Higher ω (more overlap) → better FT performance because:
- Overlapping coordinates were already learned during PT
- The initialization "remembers" useful information from pre-training
- ω=1.0 (full overlap) gives the best transfer

---

## Validation Criteria

Each figure checks acceptance criteria comparing empirical results to replica theory:

| Figure | Criterion | What it checks |
|--------|-----------|----------------|
| Step 1 | Ordering | Higher π_A → lower MSE at high α |
| Step 1 | Smoothness | Low coefficient of variation across seeds |
| Step 2 | Directionality | "good" case < "bad" case at low α |
| Step 3 | Monotonicity | Higher ω → lower MSE at low α |
| Step 3 | Qualitative match | High correlation with replica curves |

---

## File Locations

| Output | Path |
|--------|------|
| Step 1 plot | `figures/validation_step1_phase1/step1_validation_overlay.png` |
| Step 2 plot | `figures/validation_step2_phase1/step2_validation_overlay.png` |
| Step 3 plot | `figures/validation_step3_phase1/step3_validation_overlay.png` |
| Step 1 CSV | `experiment_results_step1_mixture_phase1.csv` |
| Step 2 CSV | `experiment_results_step2_support_phase1.csv` |
| Step 3 CSV | `experiment_results_step3_omega_phase1.csv` |


