#!/bin/bash
# One experiment round on the CUED blackwell node, in one command (from anywhere in the repo):
#   bash experiments/cifar_unlearning/round.sh <name>
# 1. pulls the current code (git pull --ff-only),
# 2. sources rounds/<name>.sh, whose `submit ...` lines submit the round's SLURM jobs,
# 3. submits a final job that waits for all of them (afterany) and runs publish.sh: aggregate every
#    output root, write a job report and push both to the `results` branch on GitHub.
# Needs a blackwell shell (the repo lives on its local disk) and a git credential that can push.
# Before submitting, a preflight checks imports, runs the tests, parses every task's flags and trains each
# from-scratch configuration for warm-up + 300 steps on real CIFAR; any failure stops the round before sbatch.
set -euo pipefail
name=${1:?usage: round.sh <name>  (a file experiments/cifar_unlearning/rounds/<name>.sh)}
cd "$(git rev-parse --show-toplevel)"
git pull --ff-only
export PY=${PY:-/var/tmp/na658/unlearn-env/bin/python}
mkdir -p logs
round_file=experiments/cifar_unlearning/rounds/$name.sh
[[ -f $round_file ]] || { echo "no round file $round_file" >&2; exit 1; }

L=experiments/cifar_unlearning/launch.sh

# ---- preflight: nothing is submitted unless every check passes (PREFLIGHT=0 skips it, SKIP_TESTS=1 skips the tests)
if [[ "${PREFLIGHT:-1}" == "1" ]]; then
  echo "== preflight (python $PY)"
  "$PY" experiments/cifar_unlearning/preflight.py imports
  if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    if ! "$PY" -c "import pytest" 2>/dev/null; then
      echo "pytest missing in $PY; installing it"; "$PY" -m pip install -q pytest
    fi
    "$PY" -m pytest -q -x -p no:cacheprovider experiments/cifar_unlearning/tests
  fi
  CALLS=()
  submit() { CALLS+=("$(printf '%q ' "$@")"); LAST=0; }   # record the round's submits instead of running them
  source "$round_file"
  declare -A TRAINED=()
  for call in "${CALLS[@]}"; do
    eval "set -- $call"
    arr=""
    while (( $# )) && [[ $1 != "$L" ]]; do [[ $1 == --array=* ]] && arr=${1#--array=}; shift; done
    (( $# )) || continue                                   # not a launch.sh job
    shift; stage=$1; shift
    "$PY" experiments/cifar_unlearning/launch.py "$stage" 0 --check "$@"
    n=$("$PY" experiments/cifar_unlearning/launch.py "$stage" 0 --count "$@")
    if [[ -n $arr && $arr != "0-$((n - 1))" ]]; then
      echo "preflight: $stage submitted with --array=$arr but launch.py builds $n tasks" >&2; exit 1
    fi
    key="$stage $*"
    if [[ $stage =~ ^(family|pretrain|retrain)$ && -z "${TRAINED[$key]:-}" ]]; then
      TRAINED[$key]=1
      "$PY" experiments/cifar_unlearning/preflight.py train "$stage" "$@"
    fi
  done
  unset -f submit
  echo "== preflight passed"
fi

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
source "$round_file"

dep=()
if (( ${#JOBS[@]} )); then dep=(--dependency="afterany:$(IFS=:; echo "${JOBS[*]}")"); fi
pub=$(sbatch --parsable -p cbl -A divf "${dep[@]}" --job-name=publish --time=00:30:00 \
      --output="logs/publish_%j.out" \
      --wrap "bash experiments/cifar_unlearning/publish.sh $name ${JOBS[*]:-}")
echo "publish job ${pub%%;*} runs when the jobs above have finished; results go to the 'results' branch"
