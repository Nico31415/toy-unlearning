#!/bin/bash
# One modular-arithmetic round on blackwell, as for CIFAR (from anywhere in the repo):
#   bash experiments/modular_unlearning/round.sh <name>
# pulls the code, sources rounds/<name>.sh (its `submit ...` lines are sbatch calls on cbl), then queues a final job that
# waits for them and runs publish.sh: every root under data/modular_unlearning/ is aggregated and plotted, and the CSVs,
# figures and a job report are pushed to the `results` branch under modular/.
set -euo pipefail
name=${1:?usage: round.sh <name>  (a file experiments/modular_unlearning/rounds/<name>.sh)}
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only
export PY=${PY:-/var/tmp/na658/unlearn-env/bin/python}
mkdir -p logs
round_file=experiments/modular_unlearning/rounds/$name.sh
[[ -f $round_file ]] || { echo "no round file $round_file" >&2; exit 1; }
JOBS=()
submit() {
  local id
  id=$(sbatch --parsable -p cbl -A divf "$@")
  LAST=${id%%;*}
  JOBS+=("$LAST")
  echo "submitted $LAST: $*"
}
L=experiments/modular_unlearning/launch.sh
# count <stage> <launch flags...>: number of array tasks
count() { "$PY" experiments/modular_unlearning/launch.py "$@" --count; }
source "$round_file"
dep=()
if (( ${#JOBS[@]} )); then dep=(--dependency="afterany:$(IFS=:; echo "${JOBS[*]}")"); fi
pub=$(sbatch --parsable -p cbl -A divf "${dep[@]}" --job-name=publish_mod --time=00:30:00 \
      --output="logs/publish_mod_%j.out" \
      --wrap "bash experiments/modular_unlearning/publish.sh $name ${JOBS[*]:-}")
echo "publish job ${pub%%;*} runs when the jobs above have finished; results go to the 'results' branch (modular/)"
