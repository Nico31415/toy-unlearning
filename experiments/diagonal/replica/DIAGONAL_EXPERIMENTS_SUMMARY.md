# Diagonal Experiments Sweep: Empirical vs. Replica Curves

This document provides a deep dive into the full sweep of diagonal experiments conducted to compare empirical neural network performance with theoretical replica-symmetric (RS) curves. It covers the experimental setup, the four main experiment categories, and the specific hyperparameters used.

## 1. Core Experimental Framework

The experiments are designed to test a **Diagonal Linear Network** under two main settings:
1.  **Single-Task Learning (SLT):** Training from scratch on a single task.
2.  **Pretrain-Finetune (PTFT):** Starting from an "infinite pretraining" state and finetuning on a new task with controlled feature overlap.

### 1.1 The Diagonal Network Model
The model is a diagonal linear network where the effective coefficient $\beta_i$ for coordinate $i$ is:
$$\beta_i = w_{+,i} v_{+,i} - w_{-,i} v_{-,i}$$
The initialization is governed by two key parameters:
- $c_i = w_{+,i} w_{-,i} + v_{+,i} v_{-,i}$ (corresponds to the "richness" or "feature strength")
- $\lambda_i = w_{\pm,i}^2 - v_{\pm,i}^2$ (corresponds to the "asymmetry" or "pretraining memory")

### 1.2 The "Infinite PT" Limit
For the PTFT experiments, we don't explicitly run pretraining. Instead, we construct a parameter state $(w, v)$ that exactly represents a model that has seen infinite data on a Pretraining (PT) task. This state is defined by:
- $\beta_{PT, i}$: The PT teacher coefficients.
- $c_{PT}, \lambda_{PT}$: Homogeneous initialization parameters for the PT phase.
- $\gamma_{reinit}$: A reinitialization scale used at the start of finetuning to set $\beta(0) \approx 0$ while preserving the learned $c_i$ structure.

---

## 2. Global Hyperparameters

Across all empirical experiments, the following default settings were used unless specified otherwise:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `inp_dim` ($d$) | 5000 | Input dimensionality |
| `n_test` | 10,000 | Number of test samples |
| `lr` | 0.5 | Learning rate (constant) |
| `epochs` | 5,000,000 | Max epochs (usually converges much earlier) |
| `threshold` | 1e-4 | Convergence threshold for gradient norm/loss |
| `alpha` ($\alpha$) | 0.01 to 0.5 | Sweep of 11 values ($n/d$) |
| `seeds` | 6 to 19 | 14 random seeds per point |
| `a_pt` | 1.0 | Signal scale for teachers |
| `rho_pt` ($\rho_{pt}$) | 0.1 | Sparsity of the PT task |

---

## 3. The Four Main Experiments

### Experiment 1: Benefit from Existing Features
**Goal:** Differentiate between pretraining dependence and independence.
- **Fixed:** $\rho_{pt} = \rho_{ft} = 0.1$.
- **Baseline:** $\omega=0.5, c_{pt}=10^{-3}, \lambda_{pt}=0, \gamma_{reinit}=0$.
- **Sweeps:**
    - `omega` ($\omega$): $\{0.0, 0.5, 1.0\}$ (Overlap between PT and FT supports).
    - `c_pt`: $\{10^{-6}, 10^{-3}, 1.0\}$.
    - `lambda_pt`: $\{-10^{-3}, -0.99 \cdot 10^{-3}, 0, 0.99 \cdot 10^{-3}\}$.
    - `gamma_reinit`: $\{0, 1.0, 10.0\}$.

### Experiment 2: Learning New Features
**Goal:** Differentiate between rich and lazy learning in *new* (non-pretrained) features.
- **Fixed:** $\omega = 0$ (no overlap), $\rho_{pt} = 0.1$.
- **Sweeps:**
    - `rho_ft` ($\rho_{ft}$): $\{0.1, 0.9\}$.
    - `c_pt`, `lambda_pt`, `gamma_reinit`: Same ranges as Exp 1.

### Experiment 3: Nested Feature Regime
**Goal:** Show rich vs. lazy learning on *pretrained* features.
- **Fixed:** $\rho_{pt} = 0.1$.
- **Sweeps:**
    - `omega`: $\{0, 1\}$.
    - `rho_ft`: $\{0.01, 0.04\}$.
    - `c_pt`, `lambda_pt`, `gamma_reinit`: Same ranges as Exp 1.

### Experiment 4: Single Task Learning (SLT)
**Goal:** Provide a baseline comparison and show that $\lambda_{pt}$ is irrelevant for training from scratch.
- **Sweeps:**
    - `rho_pt`: $\{0.01, 0.04, 0.1, 0.9\}$.
    - `c_pt`: $\{10^{-6}, 10^{-3}, 1.0\}$.
    - `lambda_pt`: $\{0, -c_{pt}, -0.99c_{pt}, 0.99c_{pt}\}$.

---

## 4. Mapping to Replica Curves

The replica theory (RS) predicts the Mean Squared Error (MSE) by solving coupled fixed-point equations for:
1.  $s^2$: Effective scalar-channel noise variance.
2.  $g_p$: Effective "prox noise" parameter.

The empirical results are compared against these curves by mapping the network parameters to the replica $K$ parameter:
$$K_i = 4 c_{ft, i}^2$$
where $c_{ft, i}$ is the effective richness at the start of finetuning, derived from the PT state.

### The Four Theoretical Regimes
By choosing specific parameter combinations, we explore four distinct learning regimes:

| Regime | Parameters | Setup |
| :--- | :--- | :--- |
| **Rich PT-Independent** | $c_{pt}=10^{-6}, \lambda_{pt} \approx -c_{pt}, \gamma=0$ | $\omega=0, \rho_{ft}=0.01$ |
| **Rich PT-Dependent** | $c_{pt}=10^{-6}, \lambda_{pt} \approx -0.99c_{pt}, \gamma=0$ | $\omega=1, \rho_{ft}=0.01$ |
| **Lazy PT-Dependent** | $c_{pt}=10^{-6}, \lambda_{pt} \approx 0.99c_{pt}, \gamma=0$ | $\omega=1, \rho_{ft}=0.1$ |
| **Lazy PT-Independent** | $c_{pt}=10^{-6}, \lambda_{pt}=0, \gamma=1$ | $\omega=0, \rho_{ft}=0.9$ |

---

## 5. Summary of Files
- `ptft_empirical_finetune_df.py`: The core engine for running empirical experiments.
- `ptft_replica_qk.py`: The replica solver that generates the theoretical curves.
- `compute_emp_curves_worker_exp[1-4].py`: SLURM worker scripts for each experiment.
- `ExperimentSetup.md`: A concise summary of the parameter grids.
