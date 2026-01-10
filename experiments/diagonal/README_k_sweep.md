### Diagonal PT→FT k-sweep experiments (gamma / lambda / c)

This repo’s diagonal finetuning experiment (`experiments/diagonal/diagonal_network_finetune.py`) implements the PT→FT setup.

This k-sweep suite computes/logs \(k_i\) and \(r_i\) **without modifying the training code**, by running finetune and then postprocessing the run directory.

#### Theory ↔ repo parameter map

- **Effective predictor**: `beta()` of the diagonal net (elementwise), i.e. `w_pos*v_pos - w_neg*v_neg`.
- **PT stage**: `experiments/diagonal/diagonal_network_pretrain.py`
  - **\(\lambda_{PT}\)**: `--lmda`
  - **\(c_{PT}\)**: `--c`
- **FT stage**: `experiments/diagonal/diagonal_network_finetune.py`
  - **\(\gamma\)** (FT readout reinit magnitude): `--scaling` (this is how the existing code controls the readout reinit magnitude)
  - In one-task PT+FT mode (`--one_task`), the readout parameters `v_pos/v_neg` are reinitialized to magnitude `scaling` (interpreted as \(\gamma\) here).

#### Induced \(k_i\) used in this codebase

Consistent with `fig-penalties.Rmd`, the postprocessor uses:

- \(\sqrt{k_i} = |\beta_{PT,i}| + \gamma^2\)
- \(k_i = (|\beta_{PT,i}| + \gamma^2)^2\)
- \(r_i = 2|\beta_{FT,i}| / \sqrt{k_i}\)  (also logs a legacy `r_code = |\beta|/\sqrt{k}` for comparison)

All summaries are appended to the repo-root `experiment_results_k.csv`, and per-coordinate arrays are saved to each run directory as `k_r_arrays.npz`.

---

### Minimal base experiment (recommended first)

#### 0) Pretrain a single model (required once)

This creates the pretrained checkpoint expected by the base sweep:

```bash
python experiments/diagonal/diagonal_network_pretrain.py \
  --seed 0 --inp_dim 1000 --active_dim 40 --n_train 1024 \
  --scaling 1e-3 --c 1e-3 --lmda 0.0 --init_method complex \
  --lr 0.5 --epochs 1000000 --threshold 1e-10 \
  --save_folder "data/diagonal/pretrain/seed=0--active_dim=40--c=1.0e-03--lmda=0.0000000000--init_method=complex/"
```

#### 1) Run a single FT run (smoke test)

```bash
python experiments/diagonal/diagonal_k_base.py 0
```

#### 2) Run the full base sweep (20 runs: 5 n_train2 × 4 gamma)

```bash
for i in $(seq 0 19); do
  python experiments/diagonal/diagonal_k_base.py $i
done
```

#### 3) Plot (gen vs delta/gamma/lambda + k/r histograms)

```bash
python experiments/diagonal/plot_diagonal_k_sweep.py \
  --results_csv experiment_results_k.csv \
  --filter_contains "data/diagonal/k_base/" \
  --out_dir "figures/diagonal_k_base"
```

---

### Full factorial sweep (bigger)

`experiments/diagonal/diagonal_k_sweep_1.py` runs a larger grid:

- seeds: 3
- c: 3
- lmda_frac: 3  (lmda = c * lmda_frac)
- gamma: 4
- n_train2: 5
- active_dim_2: 4
- overlap_frac: 3  (pretrain_overlap = round(overlap_frac * active_dim_2))

Total: 6480 runs.

Run as a SLURM array or a parallel job runner; locally you probably want to subset first.

---

### What to check (sanity)

- **k changes with gamma**: in `experiment_results.csv`, check `sqrt_k_mean` increases with `gamma`.
- **r distribution moves**: check `r_theory_frac_le_0.1`, `r_theory_frac_le_1`, `r_theory_frac_le_10` vary across settings.
- **Overlap axis works**: `supp_overlap_pt_teacher_frac_of_teacher` should increase with `pretrain_overlap`.


