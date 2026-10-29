#!/bin/bash
# One SLURM template for every stage. Usage, from the repository root:
#   sbatch --array=0-$(( $(python experiments/cifar_unlearning/launch.py STAGE --count) - 1 )) \
#       experiments/cifar_unlearning/launch.sh STAGE [extra flags passed to run.py]
# See experiments/cifar_unlearning/README.md for the full order and dependencies.
#SBATCH --job-name=cifar-unlearn
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=slurm/cifar-unlearn-%A_%a.out
# EDIT FOR YOUR CLUSTER: partition/account lines, e.g.
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail

# EDIT FOR YOUR CLUSTER: activate the Python environment, e.g.
# source ~/.bashrc
# conda activate <env>

STAGE=$1
shift
python experiments/cifar_unlearning/launch.py "$STAGE" "$SLURM_ARRAY_TASK_ID" "$@"
