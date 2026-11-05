#!/bin/bash
# One experiment round on the CUED blackwell node, in one command (from anywhere in the repo):
#   bash experiments/cifar_unlearning/round.sh <name>
# 1. pulls the current code (git pull --ff-only),
# 2. sources rounds/<name>.sh, whose `submit ...` lines submit the round's SLURM jobs,
# 3. submits a final job that waits for all of them (afterany) and runs publish.sh: aggregate every
#    output root, write a job report and push both to the `results` branch on GitHub.
# Needs a blackwell shell (the repo lives on its local disk) and a git credential that can push.
set -euo pipefail
name=${1:?usage: round.sh <name>  (a file experiments/cifar_unlearning/rounds/<name>.sh)}
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only
export PY=${PY:-/var/tmp/na658/unlearn-env/bin/python}
mkdir -p logs
round_file=experiments/cifar_unlearning/rounds/$name.sh
[[ -f $round_file ]] || { echo "no round file $round_file" >&2; exit 1; }

JOBS=()
# submit <sbatch args...>: submit one job (array) on cbl and remember its id for the final publish job.
# The id is also left in $LAST, so a later submit can wait for it (--dependency=afterok:$LAST).
submit() {
  local id
  id=$(sbatch --parsable -p cbl -A divf "$@")
  LAST=${id%%;*}
  JOBS+=("$LAST")
  echo "submitted $LAST: $*"
}
L=experiments/cifar_unlearning/launch.sh
source "$round_file"

dep=()
if (( ${#JOBS[@]} )); then dep=(--dependency="afterany:$(IFS=:; echo "${JOBS[*]}")"); fi
pub=$(sbatch --parsable -p cbl -A divf "${dep[@]}" --job-name=publish --time=00:30:00 \
      --output="logs/publish_%j.out" \
      --wrap "bash experiments/cifar_unlearning/publish.sh $name ${JOBS[*]:-}")
echo "publish job ${pub%%;*} runs when the jobs above have finished; results go to the 'results' branch"
