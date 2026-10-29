#!/bin/bash
# One SLURM template for every stage, following the conventions of the repo's other
# CSD3 scripts (absolute conda Python, repo root from the submit directory, logs/),
# switched from the icelake CPU partition to the Wilkes3 A100 GPU partition.
# Usage, from the repository root:
#   sbatch --array=0-$(( $($PY experiments/cifar_unlearning/launch.py STAGE --count) - 1 )) \
#       experiments/cifar_unlearning/launch.sh STAGE [extra flags passed to run.py]
# See experiments/cifar_unlearning/README.md for the full order and dependencies.
#
# CSD3 GPU rules: a project account ending in -GPU (the icelake jobs ran without -A,
# but GPU hours come from a separate -GPU project; `mybalance` lists it), partition
# ampere, at most 12 hours on SL3. Slurm gives each GPU 32 CPUs and matching memory,
# so no --cpus-per-task or --mem is requested.
#SBATCH --job-name=cifar_unlearn
#SBATCH -A CHANGEME-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/cifar_unlearn_%A_%a.out
#SBATCH --error=logs/cifar_unlearn_%A_%a.err

# CSD3's base GPU environment (as in its Wilkes3 template); loaded before strict mode.
if [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
    module purge
    module load rhel8/default-amp
fi

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch --array=..., not bash" >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Fail fast if this Python's torch cannot see the GPU (for example a CPU-only build):
# otherwise every stage would silently run on CPU and hit the time limit.
if [[ "${REQUIRE_GPU:-1}" == "1" ]]; then
  "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: torch in $PY cannot see a GPU (CPU-only build, or no GPU allocated)" >&2
    exit 1
  }
fi

STAGE=$1
shift
echo "[cifar_unlearn] job ${SLURM_JOB_ID:-?} task $SLURM_ARRAY_TASK_ID host $(hostname) stage $STAGE python $PY"
"$PY" experiments/cifar_unlearning/launch.py "$STAGE" "$SLURM_ARRAY_TASK_ID" "$@"
echo "[cifar_unlearn] task $SLURM_ARRAY_TASK_ID done"
