# c=0.05 Experiment: Testing the Transition Point

## Goal
Determine where the transition happens between "matches theory" and "doesn't match theory" by testing c=0.05 with lr=0.5.

## Current Status
- ✅ c=0.5, lr=0.5 → ratio ≈ 1.0x (perfect match)
- ❌ c=0.001, lr=0.5 → ratio ≈ 2-6x (poor match)
- ❓ c=0.05, lr=0.5 → **TO BE DETERMINED**

## How to Run

### Step 1: Submit Empirical Experiments (22 jobs)
```bash
cd /home/na658/multi-task2
sbatch experiments/diagonal/diagonal_bg_c005_lr05.sh
```

This will:
- Run 11 alpha values (0.05 to 1.0)
- 2 seeds per alpha
- Total: 22 SLURM array jobs
- Time: ~3 hours per job
- Results saved to: `experiment_results_bg_c005_lr05.csv`

### Step 2: Generate Replica Curve (1 job)
```bash
sbatch experiments/diagonal/replica_curve_c005.sh
```

This will:
- Generate replica theory curve for c=0.05, rho=0.04
- Save to: `figures/diagonal/bg_generalization/replica_cache/replica_curve_rho=0.040000--c=0.050000--*.csv`
- Time: ~30-60 minutes

### Step 3: Plot Results
After both jobs complete:
```bash
python scripts/diagonal/plot_c005_vs_replica.py
```

This will:
- Compare empirical vs replica theory
- Generate ratio plot
- Print summary statistics
- Save plots to: `figures/diagonal/bg_generalization/empirical_c0.05_vs_replica.png`

## Expected Outcomes

### Scenario A: Ratio ≈ 1.0-1.2x (Good Match)
**Conclusion:** Transition is between c=0.001 and c=0.05
**Action:** Use c≥0.05 with lr=0.5 for all future experiments

### Scenario B: Ratio ≈ 1.5-2.0x (Intermediate)
**Conclusion:** Smooth transition, c=0.05 is borderline
**Action:** Test c=0.1 next to find cleaner threshold

### Scenario C: Ratio ≈ 3-5x (Poor Match)
**Conclusion:** Transition is between c=0.05 and c=0.5
**Action:** Test c=0.1, c=0.2 to narrow down threshold

## Files Created

### Experiment Scripts
- `experiments/diagonal/diagonal_bg_c005_lr05.py` - Python array job handler
- `experiments/diagonal/diagonal_bg_c005_lr05.sh` - SLURM submission script

### Replica Curve Script
- `experiments/diagonal/replica_curve_c005.sh` - SLURM script to generate theory curve

### Plotting Script
- `scripts/diagonal/plot_c005_vs_replica.py` - Analysis and visualization

### Output Files
- `experiment_results_bg_c005_lr05.csv` - Empirical results
- `figures/diagonal/bg_generalization/replica_cache/replica_curve_rho=0.040000--c=0.050000--*.csv` - Theory curve
- `figures/diagonal/bg_generalization/empirical_c0.05_vs_replica.png` - Comparison plot
- `figures/diagonal/bg_generalization/empirical_c0.05_vs_replica.pdf` - Comparison plot (PDF)

## Monitoring Progress

### Check experiment progress:
```bash
squeue -u $USER
```

### Check if results are being saved:
```bash
tail -f experiment_results_bg_c005_lr05.csv
```

### Check logs:
```bash
tail -f logs/<jobid>_<arrayid>.out
```

## Next Steps After Results

Depending on the outcome, you may need to:
1. If good match → Proceed with science using c≥0.05
2. If intermediate → Run additional c value (e.g., c=0.1)
3. If poor match → Map the full transition curve

## Key Parameters
- c = 0.05 (test value)
- lr = 0.5 (standard learning rate)
- rho = 0.04 (sparsity)
- threshold = 1e-12 (convergence criterion)
- epochs = 5,000,000 (max)
- inp_dim = 1000
- alpha = 0.05, 0.1, 0.2, ..., 1.0 (11 values)
- seeds = 0, 1 (2 per alpha)

