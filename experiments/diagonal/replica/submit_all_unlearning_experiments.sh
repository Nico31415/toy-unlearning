#!/bin/bash
set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this script on a Slurm login node." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

submit() {
  local script="$1"
  echo "[submit] $script"
  sbatch "$SCRIPT_DIR/$script"
}

# Experiment 1: main empirical forgetting sweep.
submit submit_emp_forgetting.sh

# Experiment 2: empirical sanity checks and omega/lambda variants.
submit submit_emp_sanity_check_omega00.sh
submit submit_emp_sanity_check_omega01.sh
submit submit_emp_sanity_check_omega05.sh
submit submit_emp_sanity_check_omega09.sh
submit submit_emp_sanity_check_omega1.sh
submit submit_emp_sanity_check_omega05_lam099.sh

# Experiment 3: replica sanity curves.
submit submit_replica_sanity_check_omega05.sh

# Experiment 4: correlated-overlap q sweep.
submit submit_emp_correlated_overlap_q_sweep.sh

# Experiment 5: scratch FT baseline.
submit submit_emp_correlated_overlap_scratch.sh

# Experiment 6: PT recovery assays.
submit submit_readout_recovery_correlated_overlap.sh
submit submit_gd_recovery_correlated_overlap.sh

# Post-hoc metric generation. This should run after the correlated-overlap and
# scratch arrays have completed; submit it now if your Slurm dependency policy
# or manual workflow handles ordering externally.
echo
echo "Not auto-submitting submit_correlated_overlap_with_scratch_metrics.sh because it depends on q-sweep and scratch artifacts."
echo "Run it after those arrays complete:"
echo "  sbatch $SCRIPT_DIR/submit_correlated_overlap_with_scratch_metrics.sh"
