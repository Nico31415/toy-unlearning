# PTFT replica \(q_k\) solver — complete reimplementation guide
This document explains, in full detail, how `experiments/diagonal/replica/ptft_replica_qk.py` works: how it samples “ground truths”, how it constructs heterogeneous \(k_i\), what the \(q_k\) proximal operator is, and how the replica-symmetric (RS) fixed-point equations are solved (including continuation, homotopy, and reliability scoring).

The goal is that you can **reimplement the entire file from scratch** from this explanation alone.

---

## 1) What the code computes

The file produces **curves** indexed by \(\alpha\) (sample efficiency). Internally it uses

\[
\beta \;=\; \frac{1}{\alpha}.
\]

For each \(\beta\), it solves a pair of coupled RS fixed-point equations for two scalar state variables:

- \(s^2\) (`s2`): effective scalar-channel noise variance in the observation \(z = X + \sqrt{s^2}\,V\)
- \(g_p\) (`gp`): effective “prox noise” parameter used inside the denoiser \(\mathrm{prox}_{q_k}\)

Given a generative model for \((X, K)\), it returns per-\(\alpha\) estimates of:

- `mse_best`: \(\mathbb E[(X-\hat X)^2]\) (chosen from best of forward/backward continuation branches)
- `active_best`: \(\mathbb P(|\hat X|>\varepsilon)\) with a tiny \(\varepsilon\) (a sparsity proxy)
- `fp_residual`: fixed-point residual at termination
- standard-error diagnostics for MSE, and branch mismatch diagnostics in dB

Two problem settings are supported:

- **PTFT oracle** (`ptft_qk_curve`): heterogeneous \(K\) derived from a PT→FT mapping and a 4-group overlap model.
- **Single-task baseline** (`single_task_qk_curve`): constant \(K = 4c^2\) with Bernoulli–Gaussian \(X\).

---

## 2) Notation (exactly matching code variables)

- `mc` — number of Monte Carlo samples used to approximate expectations (arrays of length `mc`).
- `x` — Monte Carlo samples of the teacher coefficient \(X\): shape `(mc,)`.
- `v` — Monte Carlo samples of \(V\sim\mathcal N(0,1)\): shape `(mc,)`.
- `k_mc` — Monte Carlo samples of per-coordinate \(K\): shape `(mc,)`.
- `g_mc` — integer group labels in `{0,1,2,3}` for the PTFT oracle; optional.
- `alpha` — sample efficiency values; `betas = 1/alphas`.
- `sigma0_2` — \(\sigma_0^2\), a base additive noise term in the RS equation for \(s^2\).
- `gamma_ext` — \(\gamma_{\mathrm{ext}}\), a base additive term in the RS equation for \(g_p\) (also used for homotopy schedules).

---

## 3) The PTFT “oracle” sampling: how ground truths and \(k_i\) are initialized

This is implemented by `sample_ptft_oracle_mc(mc, seed, p, ft_teacher_norm=...)`.

### 3.1) PTFT parameters and feasibility

The dataclass `PTFTOracleParams` stores:

- \(\rho_{\mathrm{pt}}\in(0,1)\): fraction of PT-active coordinates
- \(\rho_{\mathrm{ft}}\in(0,1)\): fraction of FT-nonzero coordinates
- \(\omega\in[0,1]\): *overlap parameter* controlling what fraction of FT-nonzeros were also PT-active
- plus PT hyperparameters used to create \(K\): `a_pt`, `c_pt`, `lambda_pt`, `gamma_reinit`

The model partitions coordinates into four mutually exclusive groups:

| group label | meaning | probability |
|---:|---|---:|
| 0 | overlap: PT-active **and** FT-nonzero | \(p_{\mathrm{ov}}=\omega\rho_{\mathrm{ft}}\) |
| 1 | FT-new: FT-nonzero but PT-inactive | \(p_{\mathrm{new}}=(1-\omega)\rho_{\mathrm{ft}}\) |
| 2 | PT-only: PT-active but FT-zero | \(p_{\mathrm{ptonly}}=\rho_{\mathrm{pt}}-p_{\mathrm{ov}}\) |
| 3 | none: PT-inactive and FT-zero | \(p_{\mathrm{none}}=1-\rho_{\mathrm{pt}}-p_{\mathrm{new}}\) |

Feasibility constraints enforced by the code:

- \(p_{\mathrm{ptonly}}\ge 0\;\Longleftrightarrow\;\omega\rho_{\mathrm{ft}}\le\rho_{\mathrm{pt}}\)
- \(p_{\mathrm{none}}\ge 0\;\Longleftrightarrow\;\rho_{\mathrm{pt}}+(1-\omega)\rho_{\mathrm{ft}}\le 1\)

### 3.2) Sampling group labels

With RNG `np.random.default_rng(seed)`, sample i.i.d. group labels

\[
G_i \sim \mathrm{Categorical}(p_{\mathrm{ov}}, p_{\mathrm{new}}, p_{\mathrm{ptonly}}, p_{\mathrm{none}}),\quad i=1,\dots,\texttt{mc}.
\]

These are stored in `g_mc` as integers 0–3.

### 3.3) Sampling the FT ground truth \(X\)

Initialize `x = zeros(mc)`.

Define FT-nonzero indices:

\[
\texttt{ft\_nonzero}_i \equiv (G_i\in\{0,1\}).
\]

For those indices, sample Gaussian teacher values

\[
X_i \sim \mathcal N(0,\sigma_{\mathrm{ft}}^2).
\]

The code supports two conventions:

- `ft_teacher_norm="unit_total_var"` (default):
  \[
  \sigma_{\mathrm{ft}}=\frac{1}{\sqrt{\rho_{\mathrm{ft}}}}.
  \]
  This makes \(\mathbb E[X^2]=1\) regardless of \(\rho_{\mathrm{ft}}\), since
  \(\mathbb E[X^2]=\rho_{\mathrm{ft}}\cdot(1/\rho_{\mathrm{ft}})=1\).
- `ft_teacher_norm="unit_nonzero_var"`:
  \[
  \sigma_{\mathrm{ft}}=1.
  \]

All other indices remain exactly zero.

### 3.4) Sampling the PT “oracle” activity \(\beta_{\mathrm{pt},i}\)

Initialize `beta_pt = zeros(mc)`.

Define PT-active indices:

\[
\texttt{pt\_active}_i \equiv (G_i\in\{0,2\}).
\]

Set

\[
\beta_{\mathrm{pt},i}=
\begin{cases}
a_{\mathrm{pt}} & \text{if } G_i\in\{0,2\},\\
0 & \text{otherwise}.
\end{cases}
\]

### 3.5) Mapping PT activity to FT-scale \(c_{\mathrm{ft},i}\), then to \(K_i\)

The file defines:

\[
c_{\mathrm{ft},i}
\;=\;
(\lambda_{\mathrm{pt}} + c_{\mathrm{pt}})\left(1+\sqrt{1+\left(\frac{\beta_{\mathrm{pt},i}}{c_{\mathrm{pt}}}\right)^2}\right)
\;+\;\frac{1}{2}\gamma_{\mathrm{reinit}}^2,
\]

implemented by `compute_c_ft_from_pt(...)`.

Then it sets the \(q_k\) parameter

\[
K_i = 4\,c_{\mathrm{ft},i}^2,
\]

stored as `k_mc`.

**Important implementation detail:** since \(\beta_{\mathrm{pt},i}\in\{0,a_{\mathrm{pt}}\}\), the resulting \(K_i\) typically takes only **two values** (PT-active vs PT-inactive), but the solver supports arbitrary discrete or even per-sample \(K_i\).

### 3.6) Sampling the scalar-channel Gaussian noise \(V\)

Independently sample `v ~ Normal(0,1)` length `mc`.

### 3.7) Diagnostic info returned

The sampler returns an `info` dict containing:

- target group probabilities
- empirical group fractions in the drawn sample
- empirical \(\rho_{\mathrm{ft}}\) and \(\rho_{\mathrm{pt}}\)
- mean \(K\) by group label

---

## 4) Single-task ground truth initialization (baseline)

`single_task_qk_curve(...)` uses a Bernoulli–Gaussian teacher:

- choose `active_i ~ Bernoulli(rho)`
- set \(X_i=0\) if inactive
- if active, sample \(X_i\sim\mathcal N(0, 1/\rho)\), i.e. std \(1/\sqrt{\rho}\)

As in the PTFT `"unit_total_var"` case, this makes \(\mathbb E[X^2]=1\).

The \(K\) parameter is constant:

\[
K_i \equiv K_0 = 4c^2,
\]

stored as `k_mc = full(mc, K0)`.

---

## 5) The \(q_k\) proximal operator: definition and numerical solver

### 5.1) The implicit penalty \(q_k\)

The prox is defined as:

\[
\hat x
\;=\;
\arg\min_x\left\{
\frac{(x-z)^2}{2g_p} + q_k(x)
\right\},
\]

where \(g_p>0\) and \(k>0\) are parameters.

The code is written in terms of the stationarity condition

\[
0 = \frac{\partial}{\partial x}\left[\frac{(x-z)^2}{2g_p} + q_k(x)\right]
 = \frac{x-z}{g_p} + q_k'(x),
\]

and it uses the specific derivative

\[
q_k'(x)=\frac{1}{2}\operatorname{asinh}\!\left(\frac{2x}{\sqrt{k}}\right).
\]

Therefore the prox equation becomes:

\[
F(x) \equiv x - z + \frac{1}{2}g_p\,\operatorname{asinh}\!\left(\frac{2x}{\sqrt{k}}\right) = 0,
\]

which is exactly what `prox_qk` solves.

### 5.2) Curvature and uniqueness

Differentiate \(F\):

\[
F'(x)=1+\frac{g_p}{\sqrt{k+4x^2}} > 0,
\]

so \(F\) is strictly increasing and has at most one root; in fact it has exactly one root because \(F(x)\to\pm\infty\) as \(x\to\pm\infty\). Thus the prox solution is unique.

Also,

\[
q_k''(x) = \frac{1}{\sqrt{k+4x^2}} > 0,
\]

so \(q_k\) is strictly convex.

### 5.3) The numerical method used in the code

For each element of `z`, the code uses a **safeguarded Newton method**:

1. **Bracket initialization**:
   - `lo = min(z, 0)`, `hi = max(z, 0)` elementwise.
   - initialize `x = (lo+hi)/2`.
2. **Newton step**:
   \[
   x_{\text{new}} = x - \frac{F(x)}{F'(x)}.
   \]
3. **Bracket update** (monotone root bracketing):
   - if \(F(x)<0\), the root is above \(x\) so set `lo = x`
   - else set `hi = x`
4. **Safeguard**:
   - if `x_new` leaves \([lo,hi]\) or is non-finite, replace it with midpoint `(lo+hi)/2`.
5. Stop when `max(|F(x)|) < tol` or after a fixed number of iterations.

This converges reliably because \(F\) is monotone and smooth.

---

## 6) Local variance \(\sigma^2\): what `sigma2_qk` computes

After computing \(\hat x\), the code computes a scalar “local variance”

\[
\sigma^2(\hat x; g_p,k)
 \;=\;
\frac{1}{\frac{1}{g_p}+q_k''(\hat x)}
 \;=\;
\frac{1}{\frac{1}{g_p}+\frac{1}{\sqrt{k+4\hat x^2}}}.
\]

In the file this is used only through its Monte Carlo mean \(\mathbb E[\sigma^2]\).

---

## 7) The RS fixed-point equations and how they’re solved

### 7.1) Scalar-channel model

Given arrays `x` and `v`, and a current guess \(s^2\), form:

\[
z_i = x_i + \sqrt{s^2}\,v_i.
\]

Then compute denoised estimates \(\hat x_i\) via the prox:

\[
\hat x_i = \mathrm{prox}_{q_{k_i}}(z_i; g_p).
\]

Also compute \(\sigma_i^2 = \sigma^2(\hat x_i; g_p, k_i)\).

### 7.2) Fixed-point equations (exact implementation)

Define Monte Carlo estimators:

\[
\widehat{\mathrm{MSE}} = \frac{1}{\texttt{mc}}\sum_i (x_i-\hat x_i)^2,
\qquad
\widehat{S} = \frac{1}{\texttt{mc}}\sum_i \sigma_i^2.
\]

The code enforces:

\[
\boxed{
\begin{aligned}
s^2 &= \sigma_0^2 + \beta\,\widehat{\mathrm{MSE}},\\[2mm]
g_p &= \gamma_{\mathrm{ext}} + \beta\,\widehat{S}.
\end{aligned}}
\]

These are updated iteratively until convergence.

### 7.3) Damped fixed-point iteration

Let \((s^2_{\text{new}}, g_{p,\text{new}})\) be the right-hand side values from the current iterate.

The code uses damping `damp ∈ (0,1]`:

\[
s^2 \leftarrow (1-\lambda)s^2 + \lambda s^2_{\text{new}},\quad
g_p \leftarrow (1-\lambda)g_p + \lambda g_{p,\text{new}},
\]

with \(\lambda=\texttt{damp}\).

### 7.4) Convergence criterion and outputs

Residual:

\[
\mathrm{res} = \max\big(|s^2_{\text{new}}-s^2|,\;|g_{p,\text{new}}-g_p|\big).
\]

Stop if `res < tol` or after `max_iters`.

Returned per \(\beta\):

- `mse`: current \(\widehat{\mathrm{MSE}}\)
- `active_frac`: \(\frac{1}{\texttt{mc}}\sum_i \mathbf 1\{|\hat x_i|>\varepsilon\}\) with `eps_active`
- `(s2, gp)`: the final (or last-iterate) states
- `fp_residual`: final residual
- `mse_mc_se`: batch-means SE estimate for the mean of \((x-\hat x)^2\)

### 7.5) Handling heterogeneous \(k_i\)

There are two execution paths:

1. **Grouped-by-label path** (`use_grouped_k=True` and `g_mc` provided):
   - For each group label \(\ell\), it computes a single `k_lab = mean(k_mc[g==ℓ])` and applies `prox_qk` with that \(k\) to all samples in the group.
   - This assumes \(k_i\) is (approximately) constant within each label (true for the PTFT construction in this file).
2. **General discrete-\(k\) path**:
   - It enumerates unique values in `k_mc` and applies the prox per unique \(k\) (fast if the number of unique values is small).
   - If there are too many unique \(k\) values, it falls back to a slower per-sample loop.

---

## 8) Continuation (forward/backward) and gamma-homotopy

Replica fixed points can be multi-valued (multiple stable branches). This file uses two stabilizers:

### 8.1) Forward/backward continuation in \(\beta\)

Given `alphas`, define `betas = 1/alphas`.

- **Forward pass:** iterate `betas` in array order, warm-starting each \(\beta_i\) with the converged state from \(\beta_{i-1}\).
- **Backward pass:** iterate in reverse order, warm-starting from the next point on the reverse path.

This yields two candidate MSE curves: `mse_fwd`, `mse_bwd`.

The code defines:

- `mse_best[i] = min(mse_fwd[i], mse_bwd[i])` (pointwise)
- `diff_db[i] = |10 log10(mse_fwd[i]) - 10 log10(mse_bwd[i])|`

### 8.2) Gamma-homotopy (warm-start across \(\gamma_{\mathrm{ext}}\))

When \(\gamma_{\mathrm{ext}}\) is very small, direct solving can be unstable. The file creates a decreasing schedule down to `gamma_target`:

- If `gamma_target >= 1e-6`: schedule is `[gamma_target]` only.
- If `0 < gamma_target < 1e-6`: schedule is `[1e-6, 1e-8, 1e-10, 1e-12, gamma_target]` truncated so all elements are strictly decreasing and above the target.
- If `gamma_target == 0`: schedule is `[1e-6, 1e-8, 1e-10, 1e-12, 0]`.

Crucially, the solver maintains **per-beta index** warm-start states across gamma steps:

- `fwd_states[i]` caches the converged `(s2, gp)` for beta index `i` from the previous gamma.
- Same for `bwd_states[i]` on the backward branch.

At each gamma, each beta solve chooses initialization:

1) if a per-index state exists from the previous gamma, use it; else  
2) fall back to continuation warm-start from the previous beta on the same gamma.

This is why gamma homotopy is effective here: you solve an easier problem at higher gamma first, then gradually decrease gamma while reusing the closest available solution as initialization.

---

## 9) Estimating Monte Carlo uncertainty and “reliability”

### 9.1) Batch-means SE for mean MSE

Let \(e_i^2 = (x_i-\hat x_i)^2\). The code estimates the SE of \(\frac{1}{mc}\sum e_i^2\) using batch means:

1. Choose number of batches:
   \[
   B=\max\{5,\;\min(\texttt{n\_batches\_se},\;\lfloor mc/200\rfloor)\}.
   \]
   If \(B<5\), return NaN.
2. Truncate to multiple of \(B\), reshape into `(B, m)` with \(m=\lfloor mc/B\rfloor\).
3. Compute batch means \(\bar e_b^2\) and return:
   \[
   \mathrm{SE} = \frac{\mathrm{std}(\bar e_b^2)}{\sqrt{B}}.
   \]

### 9.2) SE in dB and the “no explosion near MSE→0” fix

The code reports an SE for \(10\log_{10}(\mathrm{MSE})\) using the delta method:

\[
\mathrm{SE}_{\mathrm{dB}}
\approx
\left(\frac{10}{\ln 10}\right)\frac{\mathrm{SE}(\mathrm{MSE})}{\mathrm{MSE}}.
\]

To prevent blow-up when \(\mathrm{MSE}\to 0\), it caps the slope by flooring MSE at
`mse_floor_for_db_se` (and at least \(10^{-300}\)):

\[
\mathrm{SE}_{\mathrm{dB}}
 =
\left(\frac{10}{\ln 10}\right)\frac{\mathrm{SE}(\mathrm{MSE})}{\max(\mathrm{MSE},\mathrm{floor})}.
\]

It also computes a “relative SE” channel:

\[
\mathrm{rel\_se}
 = \frac{\mathrm{SE}(\mathrm{MSE})}{\max(\mathrm{MSE},\mathrm{floor})}.
\]

### 9.3) Reliability score (single-number “penalty in dB”)

For each \(\alpha\), define three penalties:

1. **Branch mismatch**: `diff_db`
2. **MC uncertainty**: \(z\cdot \mathrm{SE}_{\mathrm{dB}}\) (with NaNs treated as 0)
3. **Fixed-point convergence penalty**:
   \[
   s_{\mathrm{fp,db}}\cdot \log_{10}\!\left(1+\frac{\max(\mathrm{fp\_residual},0)}{\max(\mathrm{tol},10^{-300})}\right)
   \]

Per-\(\alpha\) score is the max of those; curve score is `p95`/`max`/`median` across alphas (default `p95`).

---

## 10) Exact outputs of `ptft_qk_curve` and `single_task_qk_curve`

Both return `(curve, reliability, info)`.

### 10.1) `curve` dict fields

- `alpha`: the alpha grid (float array)
- `mse_best`: pointwise min of forward/backward MSE
- `active_best`: active fraction corresponding to whichever branch won at that alpha
- `mse_fwd`, `mse_bwd`: MSE from forward and backward continuation
- `diff_db`: branch mismatch in dB as defined above
- `fp_residual`: residual from the selected branch
- `mse_se`: batch-means SE of MSE from selected branch
- `mse_rel_se`: relative SE (with floor) from selected branch
- `mse_se_db`: SE in dB (with slope floor) from selected branch
- `gamma_schedule`: the gamma schedule actually used (float array)

### 10.2) `reliability` dict fields

- `score_db`: aggregated penalty score (bigger = worse)
- `score_alpha_db`: per-alpha penalty (max of the three channels)
- `score_parts`: dict with `branch_db`, `mc_db`, `fp_db` arrays
- plus configuration values used: `agg`, `z`, `s_fp_db`, `tol`, `mse_floor_for_db_se`

### 10.3) `info` dict

- PTFT: sampling diagnostics from the oracle (group fracs, empirical rhos, mean k by group).
- Single-task: `{"rho", "rho_emp", "c", "k0"}`.

---

## 11) Full reimplementation pseudocode (faithful to the file)

Below is “structured pseudocode” that matches the exact control flow and default behaviors of `ptft_replica_qk.py`.

### 11.1) Utility

```text
to_db(x):
  return 10 * log10(max(x, 1e-15))

batch_means_se(values, n_batches=50):
  n = len(values)
  B = max(5, min(n_batches, floor(n/200)))
  if B < 5: return NaN
  m = floor(n/B)
  reshape first (m*B) elements to (B, m)
  xb = mean over axis=1
  return std(xb, ddof=1) / sqrt(B)
```

### 11.2) PTFT oracle sampler

```text
sample_ptft_oracle_mc(mc, seed, params, ft_teacher_norm):
  compute group probs:
    p_ov = omega * rho_ft
    p_new = (1-omega) * rho_ft
    p_ptonly = rho_pt - p_ov
    p_none = 1 - rho_pt - p_new
  check p_ov,p_new,p_ptonly,p_none >= 0 (with tiny slack)
  g = categorical({0,1,2,3}, probs) size mc

  beta_pt = zeros(mc)
  beta_pt[g in {0,2}] = a_pt

  x = zeros(mc)
  ft_nonzero = (g in {0,1})
  if any(ft_nonzero):
    if ft_teacher_norm == "unit_total_var": sigma_ft = 1/sqrt(rho_ft)
    else if "unit_nonzero_var": sigma_ft = 1
    else error
    x[ft_nonzero] = Normal(0, sigma_ft) iid

  c_ft = (lambda_pt + c_pt) * (1 + sqrt(1 + (beta_pt/c_pt)^2)) + 0.5*gamma_reinit^2
  k = 4 * c_ft^2

  v = Normal(0,1) iid length mc

  return x, k, g, v, info_dict
```

### 11.3) Prox and local variance

```text
prox_qk(z, gp, k):
  solve for x:  x - z + 0.5*gp*asinh(2x/sqrt(k)) = 0
  using safeguarded Newton + bracketing between 0 and z

sigma2_qk(xhat, gp, k):
  qpp = 1/sqrt(k + 4*xhat^2)
  return 1 / (1/gp + qpp)
```

### 11.4) Solve one fixed point at one beta

```text
solve_fp_qk_one(beta, x, v, k, g_or_None, sigma0_2, gamma_ext, init_state, ...):
  initialize:
    if init_state:
      s2 = max(sigma0_2, init_state.s2)
      gp = max(gamma_ext, init_state.gp, 1e-14)
    else:
      s2 = sigma0_2
      gp = max(gamma_ext, 1e-14)

  repeat up to max_iters:
    z = x + sqrt(max(s2,1e-15)) * v

    compute xhat and sig2:
      if use_grouped_k and g provided:
        for each label ℓ in unique(g):
          k_lab = mean(k[g==ℓ])
          xhat[g==ℓ] = prox_qk(z[g==ℓ], gp, k_lab)
          sig2[g==ℓ] = sigma2_qk(xhat[g==ℓ], gp, k_lab)
      else:
        uniq_k = unique(k)
        if len(uniq_k) <= 64:
          for each k_val in uniq_k:
            idx = (k == k_val)
            xhat[idx] = prox_qk(z[idx], gp, k_val)
            sig2[idx] = sigma2_qk(xhat[idx], gp, k_val)
        else:
          slow per-sample loop

    err2 = (x - xhat)^2
    mse = mean(err2)
    mean_sig2 = mean(sig2)

    s2_new = sigma0_2 + beta*mse
    gp_new = gamma_ext + beta*mean_sig2

    res = max(|s2_new - s2|, |gp_new - gp|)
    if res < tol:
      active = mean(|xhat| > eps_active)
      mse_se = batch_means_se(err2, n_batches_se)
      return mse, active, (s2_new, gp_new), res, mse_se

    s2 = (1-damp)*s2 + damp*s2_new
    gp = (1-damp)*gp + damp*gp_new
    s2 = max(s2, sigma0_2)
    gp = max(gp, gamma_ext, 1e-14)

  (not converged)
  active = mean(|xhat| > eps_active)
  mse_se = batch_means_se(err2, n_batches_se)
  return mse, active, (s2, gp), res, mse_se
```

### 11.5) Curve solver with gamma homotopy + forward/backward continuation

```text
solve_curve_with_gamma_homotopy_best_of_fwd_bwd(alphas, ..., gamma_target, gamma_schedule=None):
  betas = 1/alphas
  if gamma_schedule is None: gamma_schedule = default_gamma_schedule(gamma_target)
  else: require gamma_schedule[-1] == gamma_target and all gammas >= 0

  fwd_states = [None]*len(betas)   # per-beta warm-start across gammas
  bwd_states = [None]*len(betas)

  for gamma in gamma_schedule:
    # forward branch
    prev_state = None
    for i, beta in enumerate(betas):
      init = fwd_states[i] if fwd_states[i] else prev_state
      solve_fp_qk_one(..., beta=beta, gamma_ext=gamma, init_state=init, ...)
      store mse_fwd[i], act_fwd[i], res_fwd[i], se_fwd[i]
      fwd_states[i] = solved_state
      prev_state = solved_state

    # backward branch
    prev_state = None
    for j, beta in enumerate(reverse(betas)):
      i = len(betas)-1-j
      init = bwd_states[i] if bwd_states[i] else prev_state
      solve_fp_qk_one(..., beta=beta, gamma_ext=gamma, init_state=init, ...)
      store mse_bwd[i], act_bwd[i], res_bwd[i], se_bwd[i]
      bwd_states[i] = solved_state
      prev_state = solved_state

  diff_db = abs(to_db(mse_fwd) - to_db(mse_bwd))
  choose_fwd = mse_fwd <= mse_bwd
  mse_best = choose_fwd ? mse_fwd : mse_bwd
  act_best = choose_fwd ? act_fwd : act_bwd
  res_best = choose_fwd ? res_fwd : res_bwd
  se_best  = choose_fwd ? se_fwd  : se_bwd

  mse_floor = max(mse_floor_for_db_se, 1e-300)
  rel_se = se_best / max(mse_best, mse_floor)
  slope = (10/log(10)) / max(mse_best, mse_floor)
  se_db = slope * se_best

  return dict of arrays (alpha, mse_best, ..., gamma_schedule)
```

---

## 12) Where this file ends (no CLI)

In the repository state you’re using, `ptft_replica_qk.py` ends after the two dataframe builders:

- `build_ptft_curves_dataframe(...)`
- `build_single_task_curves_dataframe(...)`

There is **no** `if __name__ == "__main__":` CLI block in this actual file.

---

## 13) Practical reimplementation checklist

If you are re-coding this from scratch in a new environment, ensure you match:

- **Randomness**: use `np.random.default_rng(seed)`; do not use legacy global RNG.
- **Teacher normalization**: default PTFT uses `"unit_total_var"` and single-task uses variance \(1/\rho\) on active entries, so \(\mathbb E[X^2]=1\).
- **PT→FT mapping**: implement `compute_c_ft_from_pt` and \(K=4c_{\mathrm{ft}}^2\) exactly.
- **Prox equation**: solve \(x-z+\tfrac12 g_p\,\mathrm{asinh}(2x/\sqrt{k})=0\) with a safe method.
- **Fixed-point updates**: use the exact RS equations for \(s^2\) and \(g_p\), with damping and clamps.
- **Continuation and homotopy**: implement both forward/backward beta continuation and the gamma schedule with per-beta state caching.
- **Reliability score**: reproduce the three penalty channels and the aggregation, including the dB slope floor.

