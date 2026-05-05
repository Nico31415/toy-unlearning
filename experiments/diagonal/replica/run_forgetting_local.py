"""
Local sequential runner for the forgetting experiment (reduced grid).

Grid: 2 regimes x 4 alphas x 5 seeds = 40 tasks, ~25 min total.
Skips tasks that already have results saved.
"""
import itertools
import subprocess
import sys
import time
from pathlib import Path

ALPHAS  = [0.01, 0.059, 0.108, 0.206]
SEEDS   = list(range(5))
REGIMES = ["regime_III"]   # II and IV already done

OUTPUT_DIR = "results/forgetting"

# Map (regime, seed, alpha) -> task_id in the full worker grid
# (matches the ordering in compute_emp_forgetting_worker.py)
FULL_REGIMES = [
    ("regime_II",  1e-3,  0.0,     0.0),
    ("regime_IV",  1e-3, -0.99e-3, 0.0),
    ("regime_III", 1e-3,  0.0,    10.0),
]
FULL_ALPHAS = list(__import__('numpy').linspace(0.01, 0.5, 11))
FULL_SEEDS  = list(range(10))

def get_task_id(regime_name, seed, alpha):
    regime_idx = [r[0] for r in FULL_REGIMES].index(regime_name)
    seed_idx   = FULL_SEEDS.index(seed)
    alpha_idx  = min(range(len(FULL_ALPHAS)), key=lambda i: abs(FULL_ALPHAS[i] - alpha))
    return regime_idx * len(FULL_SEEDS) * len(FULL_ALPHAS) + seed_idx * len(FULL_ALPHAS) + alpha_idx

tasks = list(itertools.product(REGIMES, SEEDS, ALPHAS))
print(f"Total tasks: {len(tasks)}")

done = 0
skipped = 0
for i, (regime, seed, alpha) in enumerate(tasks):
    folder = Path(OUTPUT_DIR) / regime / f"seed{seed}_alpha{alpha:.4f}"
    if (folder / "model.pt").exists():
        print(f"[{i+1}/{len(tasks)}] SKIP (already done): {folder}")
        skipped += 1
        continue

    task_id = get_task_id(regime, seed, alpha)
    print(f"[{i+1}/{len(tasks)}] Running task_id={task_id} | {regime} seed={seed} alpha={alpha:.4f}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, "experiments/diagonal/replica/compute_emp_forgetting_worker.py",
         "--task-id", str(task_id),
         "--output-dir", OUTPUT_DIR],
        capture_output=False,
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        done += 1
        print(f"    Done in {elapsed:.1f}s\n")
    else:
        print(f"    FAILED (returncode={result.returncode}) after {elapsed:.1f}s\n")

print(f"\nFinished: {done} ran, {skipped} skipped.")
