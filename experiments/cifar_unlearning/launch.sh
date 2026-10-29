#!/bin/bash
# One SLURM template for every stage, set up for Cambridge CSD3 (Wilkes3 A100 nodes).
# Usage, from the repository root:
#   sbatch --array=0-$(( $(python experiments/cifar_unlearning/launch.py STAGE --count) - 1 )) \
#       experiments/cifar_unlearning/launch.sh STAGE [extra flags passed to run.py]
# See experiments/cifar_unlearning/README.md for the full order and dependencies.
#
# CSD3 rules this follows: GPU jobs need a -GPU project account and partition
# "ampere"; SL3 jobs may run at most 12 hours; Slurm allocates 32 CPUs and the
# matching memory per GPU itself, so no --cpus-per-task or --mem is requested.
#
# EDIT 1: your GPU project account (run `mybalance` on a login node to see it).
#SBATCH -A CHANGEME-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=cifar-unlearn
#SBATCH --output=slurm/cifar-unlearn-%A_%a.out

# Load the CSD3 base environment (as in CSD3's own Wilkes3 template).
if [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
    module purge
    module load rhel8/default-amp
fi

# EDIT 2: activate the Python environment that has torch and torchvision, e.g.
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate <env>

# Strict mode only after activation: conda's activation scripts are not safe under `set -u`.
set -euo pipefail

STAGE=$1
shift
python experiments/cifar_unlearning/launch.py "$STAGE" "$SLURM_ARRAY_TASK_ID" "$@"
