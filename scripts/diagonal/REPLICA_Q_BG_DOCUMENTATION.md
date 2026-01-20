# Documentation: Replica Theory Q-Function Curves for Diagonal Network

## Overview

This document explains the implementation of replica theory curves for the implicit bias function `q` induced by diagonal network initialization, and how they are overlaid on empirical generalization error curves.

## Background and Motivation

### Experimental Setup

We have empirical generalization curves from training diagonal networks on a Bernoulli-Gaussian teacher-student regression problem:

- **Teacher**: `beta_star` follows a Bernoulli-Gaussian distribution with sparsity `rho`
  - Each component `beta_i^* = 0` with probability `1-ρ`
  - Each component `beta_i^* ~ N(0, 1/ρ)` with probability `ρ`
  - This ensures `E[(beta_i^*)^2] = 1`
- **Design Matrix**: Gaussian with RS scaling
  - Train: `X_{ij} ~ N(0, 1/n_train)`
  - Test: `X_{ij} ~ N(0, 1/n_test)`
- **Outputs**: Noiseless, `y = X @ beta_star` (so `sigma_0^2 = 0`)
- **Model**: Diagonal network with parameters `(w_pos, v_pos, w_neg, v_neg)` where `beta_hat = w_pos*v_pos - w_neg*v_neg`
- **Initialization**: Controlled by `c` and `lmda` parameters (complex initialization method)

### Goal

Generate theoretical generalization error curves using replica theory that correspond to the implicit bias function `q` induced by the diagonal network initialization. The optimization problem is:

```
minimize: data_fit + λ * q(x)
```

where `λ` is a very small regularization parameter (e.g., `1e-6`), and `q` is the implicit bias function parameterized by `k_q`.

## Key Theoretical Concepts

### 1. Diagonal Network Initialization → q Function

The diagonal network uses "complex initialization" with parameters `c` and `lmda`:

```python
def get_parameters(c, lmda):
    v = sqrt((c + lmda) / 2)  # w_pos, v_pos
    u = sqrt((c - lmda) / 2)  # w_neg, v_neg
    return v, v, u, u
```

At initialization:
- `w_pos = v_pos = sqrt((c + lmda)/2)`
- `w_neg = v_neg = sqrt((c - lmda)/2)`
- `beta(0) = w_pos*v_pos - w_neg*v_neg = (c + lmda)/2 - (c - lmda)/2 = lmda`

The implicit bias function `q` is parameterized by `k_q`. Based on the codebase documentation (`postprocess_diagonal_st_k.py` line 204-206), for uniform initialization:

```
sqrt(k) = c
k = c^2
```

**Therefore: `k_q = c^2`**

This mapping connects the initialization parameter `c` to the q-function parameter `k_q` used in replica theory.

### 2. Replica Theory Framework

Replica theory provides a framework to compute generalization error for optimization problems of the form:

```
minimize: ||y - X*beta||^2 + λ * q(beta)
```

where:
- `y = X @ beta_star` (noiseless case: `sigma_0^2 = 0`)
- `q` is a regularization function parameterized by `k_q`
- `λ` is the regularization strength (very small in our case)

The replica theory computation uses:
- **Measurement ratio**: `beta = n / d = inp_dim / n_train = 1 / alpha`
  - Where `alpha = n_train / inp_dim` is the empirical x-axis
- **Fixed point iteration**: Solves coupled equations for effective noise variance and regularization strength
- **Proximal operator**: The q-function's proximal operator is used in the fixed point equations

### 3. The q-Function Proximal Operator

The q-function has a specific form with parameter `k_q`. Its proximal operator (from `ReplicaExperiments/fixed_lambda_all.py`) is:

```python
def prox_qk_safeguarded(z, lam, k):
    sk = sqrt(k)
    # Solves: x - z + 0.5 * lam * arcsinh(2*x / sqrt(k)) = 0
    # Uses Newton's method with bracketing
```

The regularization term in the optimization is: `0.5 * λ * arcsinh(2*x / sqrt(k_q))`, which means the effective regularization strength depends on the ratio `λ / sqrt(k_q)`.

### 4. gamma_ext Scaling

Because the effective regularization depends on `λ / sqrt(k_q)`, we need to scale the regularization parameter `λ` based on `k_q` to get comparable regularization strength across different `k_q` values.

The scaling functions (from `ReplicaExperiments/fixed_lambda_all.py`):

- **For small k** (`k < 1.0`): `gamma_ext = lambda * (4.0 / log(1/k))`
- **For large k** (`k >= 1.0`): `gamma_ext = lambda * sqrt(k)`

For our values:
- `c = 0.001` → `k_q = 1e-6` (very small) → uses small scaling: `gamma_ext ≈ λ * 0.29`
- `c = 0.5` → `k_q = 0.25` (moderate, < 1.0) → uses small scaling: `gamma_ext ≈ λ * 2.89`

## Implementation Details

### File: `scripts/diagonal/plot_replica_q_bg.py`

#### Main Components

1. **Configuration Setup** (`build_config`):
   - Sets up replica theory parameters: `rho`, `sigma0_2 = 0.0` (noiseless), `var_nonzero = 1/rho`
   - Creates beta range from alpha range: `beta = 1/alpha`
   - Configures fixed point iteration parameters

2. **Monte Carlo Sampling**:
   - Generates `x_mc`: Bernoulli-Gaussian samples using `sample_bg(n, rng, rho, var_nonzero)`
   - Generates `v_mc`: Standard Gaussian noise samples
   - Uses same seed as experiments for reproducibility

3. **Replica Curve Computation** (`compute_replica_curve`):
   - For each `c` value:
     - Computes `k_q = c^2`
     - Computes `gamma_ext` with appropriate scaling based on `k_q`
     - Converts alpha range to beta range (reversed, since beta = 1/alpha)
     - Calls `solve_rspmap_qk_curve_best_of_forward_backward` to compute MSE curve in beta space
     - Converts results back to alpha space for plotting

4. **Coordinate Conversion**:
   - **Empirical**: Uses `alpha = n_train / inp_dim` (x-axis: 0.008 to 1.0)
   - **Replica**: Uses `beta = inp_dim / n_train = 1/alpha` (x-axis: 1.0 to 125.0)
   - Conversion process:
     - Start with `alpha_range` in increasing order: `[0.008, ..., 1.0]`
     - Reverse to get decreasing: `[1.0, ..., 0.008]`
     - Compute `beta_range = 1.0 / alpha_reversed`: `[1.0, ..., 125.0]` (increasing)
     - Solve replica theory in beta space (increasing order)
     - Reverse `mse_beta` to match original `alpha_range` order

5. **Plotting** (`plot_overlay`):
   - Loads empirical results from CSV files
   - Plots empirical curves: mean (solid), median (dashed), IQR (shaded)
   - Overlays replica theory curves for each `c` value
   - X-axis: `alpha = n_train / inp_dim`
   - Y-axis: MSE in dB (`10*log10(mse + 1e-15)`)
   - Saves as PNG and PDF

#### Key Functions Used from Replica Theory

- `solve_rspmap_qk_curve_best_of_forward_backward`: Main solver that computes MSE curve
  - Uses forward and backward continuation over beta
  - Takes the best (minimum MSE) branch pointwise to handle phase transitions
- `gamma_ext_for_q_small`: Scaling function for small `k_q` values
- `gamma_ext_for_q_big`: Scaling function for large `k_q` values
- `sample_bg`: Generates Bernoulli-Gaussian samples

### File: `experiments/diagonal/diagonal_network_pretrain_bg.py`

This is the empirical experiment script that generates the data we're comparing against.

#### Key Features

1. **Bernoulli-Gaussian Teacher Sampling** (`sample_beta_star_bg`):
   - Samples mask: `mask ~ Bernoulli(rho)`
   - Samples Gaussian values: `gaussian_vals ~ N(0, 1/rho)`
   - Returns: `beta_star = mask * gaussian_vals`

2. **Design Matrix Generation**:
   - Train: `X ~ N(0, 1/n_train)` (RS scaling)
   - Test: `X ~ N(0, 1/n_test)` (separate scaling)
   - Uses separate random generators for train X, test X, and teacher to avoid coupling

3. **Training**:
   - Trains diagonal network to minimize MSE: `||y - X @ beta_hat||^2`
   - Logs: train/test prediction MSE and parameter MSE (`||beta_hat - beta_star||^2`)
   - Saves results to `df.feather` with columns: `epoch`, `split`, `pred_mse`, `param_mse`

4. **Output Files**:
   - `df.feather`: Training history with pred_mse and param_mse
   - `norm_df.feather`: L1/L2 norms of beta_hat over time
   - `beta_star.pt`: Ground truth teacher parameters
   - `model.pt`: Trained model state

### File: `scripts/diagonal/plot_generalization_bg.py`

This is the aggregator script that runs multiple experiments and creates empirical plots.

#### Key Features

1. **Experiment Execution**:
   - Runs experiments across a grid of `n_train` values (creating alpha values)
   - For each `(n_train, seed)` combination, calls `diagonal_network_pretrain_bg.py` via subprocess
   - Skips already-completed experiments

2. **Aggregation**:
   - Extracts final test `pred_mse` and `param_mse` from each experiment
   - Computes statistics across seeds: mean, median, 25th/75th percentiles
   - Saves aggregated results to CSV

3. **Plotting**:
   - Creates generalization curves: alpha vs test prediction MSE (in dB)
   - Shows mean (solid), median (dashed), and IQR (shaded)
   - Saves as PNG and PDF

## What Was Plotted

### Empirical Curves

The empirical curves show generalization error from actual training runs:

- **X-axis**: `alpha = n_train / inp_dim` (ranging from ~0.008 to 1.0)
- **Y-axis**: Test prediction MSE in dB (`10*log10(mse + 1e-15)`)
- **Data points**: Mean and median across multiple seeds (typically 10-20 seeds)
- **Uncertainty**: IQR (25th-75th percentiles) shown as shaded region
- **Two curves**: One for `c = 0.001` and one for `c = 0.5`

### Replica Theory Curves

The replica theory curves show the theoretical prediction for the same setup:

- **Same x-axis**: `alpha = n_train / inp_dim`
- **Same y-axis**: Test prediction MSE in dB
- **Computation**: Uses replica theory fixed point equations with:
  - `k_q = c^2` (mapping from initialization parameter)
  - `gamma_ext` scaled appropriately based on `k_q`
  - Very small `λ = 1e-6` regularization parameter
  - Noiseless case (`sigma_0^2 = 0`)
  - Same `rho = 0.04` sparsity

### Overlay Plot

The final plot (`replica_overlay.png/pdf`) shows:

- Empirical curves (blue/green) with error bars
- Replica theory curves (red/orange) overlaid
- Allows comparison between theory and experiment

## Logic and Reasoning

### Why k_q = c^2?

From the codebase documentation and the initialization:
- At initialization: `w_pos = v_pos = sqrt((c + lmda)/2)`, `w_neg = v_neg = sqrt((c - lmda)/2)`
- For uniform initialization (when `lmda = 0`): all parameters start at `sqrt(c/2)`
- The induced scale `sqrt(k)` is related to the initialization magnitude
- Documentation explicitly states: `sqrt(k) = c` for uniform STL initialization, so `k = c^2`

### Why Scale gamma_ext?

The q-function's proximal operator has the form:
```
prox: x - z + 0.5 * λ * arcsinh(2*x / sqrt(k)) = 0
```

The effective regularization strength is `λ / sqrt(k)`. To get comparable regularization behavior across different `k` values, we scale `λ` by a factor that depends on `k`:
- For small `k`: The `arcsinh` term dominates, so we use logarithmic scaling
- For large `k`: The linear approximation applies, so we use `sqrt(k)` scaling

### Why Very Small λ?

The goal is to see the implicit bias of the initialization, not strong explicit regularization. With `λ = 1e-6`, the regularization term is very small compared to the data fit term, so the solution is primarily determined by the implicit bias structure encoded in the q-function.

### Coordinate Conversion Logic

The conversion between alpha and beta is critical:

1. **Empirical uses alpha**: `alpha = n_train / inp_dim` (increasing from 0.008 to 1.0)
2. **Replica uses beta**: `beta = inp_dim / n_train = 1/alpha` (increasing from 1.0 to 125.0)
3. **Conversion process**:
   - Start: `alpha_range = [0.008, 0.016, ..., 1.0]` (increasing)
   - Reverse: `[1.0, ..., 0.016, 0.008]` (decreasing)
   - Compute beta: `[1.0, 1.33, ..., 62.5, 125.0]` (increasing) ✓
   - Solve replica theory (expects increasing beta)
   - Get `mse_beta[i]` corresponding to `beta_range[i]`
   - Reverse: `mse_alpha = mse_beta[::-1]` to match original `alpha_range` order

Verification:
- `alpha_range[0] = 0.008` should pair with `beta = 125.0`
- `alpha_reversed[-1] = 0.008`, `beta_range[-1] = 125.0`, `mse_beta[-1]` is the value
- After reversal: `mse_alpha[0] = mse_beta[-1]` ✓

## Parameters Used

### Default Values in Script

- `rho = 0.04`: Sparsity of Bernoulli-Gaussian teacher
- `sigma0_2 = 0.0`: Noiseless case
- `ft_regulariser_scale = 1e-6`: Very small regularization parameter
- `c_values = [0.001, 0.5]`: Initialization parameters to match experiments
- `alpha_min = 0.008`, `alpha_max = 1.0`: Range matching empirical data
- `alpha_points = 100`: Number of points for smooth curves
- `mc_samples = 50000`: Monte Carlo samples for replica expectations
- `max_fp_iters = 900`: Maximum fixed point iterations
- `tol_fp = 1e-10`: Fixed point convergence tolerance
- `damp = 0.25`: Damping factor for fixed point iteration

### Mapping Summary

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `c` | 0.001 or 0.5 | Diagonal network initialization parameter |
| `k_q` | `c^2` | q-function parameter (1e-6 or 0.25) |
| `lambda` | 1e-6 | Regularization strength (very small) |
| `gamma_ext` | Scaled `lambda` | Effective regularization after scaling |
| `rho` | 0.04 | Teacher sparsity |
| `alpha` | 0.008 to 1.0 | Measurement ratio (n_train / inp_dim) |
| `beta` | 1.0 to 125.0 | Inverse measurement ratio (inp_dim / n_train) |

## Output Files

### Generated by `plot_replica_q_bg.py`

- `figures/diagonal/bg_generalization/replica_overlay.png`: Overlay plot (PNG)
- `figures/diagonal/bg_generalization/replica_overlay.pdf`: Overlay plot (PDF)

### Input Files (Required)

- `figures/diagonal/bg_generalization/aggregated_results_rho=0.040000.csv`: Empirical results for c=0.001
- `figures/diagonal/bg_generalization/aggregated_results_rho=0.040000--c=0.500000.csv`: Empirical results for c=0.5

## Usage

```bash
conda run -n mtl_ft python scripts/diagonal/plot_replica_q_bg.py \
    --rho 0.04 \
    --ft_regulariser_scale 1e-6 \
    --c_values 0.001 0.5 \
    --alpha_min 0.008 \
    --alpha_max 1.0 \
    --alpha_points 100 \
    --mc_samples 50000 \
    --empirical_dir figures/diagonal/bg_generalization \
    --output_dir figures/diagonal/bg_generalization
```

## Key Insights

1. **Implicit Bias**: The diagonal network initialization induces an implicit bias function `q` parameterized by `k_q = c^2`

2. **Scaling Matters**: The regularization parameter must be scaled based on `k_q` to get comparable behavior across different initialization scales

3. **Very Small Regularization**: Using `λ = 1e-6` allows the implicit bias to dominate while still being in the regularized regime

4. **Coordinate Mapping**: Careful conversion between empirical `alpha` space and replica `beta` space is essential for correct plotting

5. **Theory-Experiment Comparison**: The overlay plot allows direct comparison between theoretical predictions (replica theory) and empirical observations (actual training)

## References

- Replica theory implementation: `ReplicaExperiments/fixed_lambda_all.py`
- Empirical experiment: `experiments/diagonal/diagonal_network_pretrain_bg.py`
- Aggregator script: `scripts/diagonal/plot_generalization_bg.py`
- Documentation on k mapping: `experiments/diagonal/postprocess_diagonal_st_k.py` (lines 204-206)









