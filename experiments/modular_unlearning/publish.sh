#!/bin/bash
# Aggregate and plot every modular root, push CSVs, figures and a job report to the `results` branch under modular/.
# Called by round.sh's final job:  publish.sh <round name> [job ids...]. Uses the same results worktree as CIFAR.
set -uo pipefail
name=$1; shift
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-/var/tmp/na658/unlearn-env/bin/python}
WT=${RESULTS_WORKTREE:-/var/tmp/na658/results-wt}
if ! git -C "$WT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git ls-remote --exit-code --heads origin results >/dev/null 2>&1; then
    git fetch origin results:results && git worktree add "$WT" results
  else
    git worktree add --detach "$WT" && git -C "$WT" checkout --orphan results && git -C "$WT" rm -rfq .
  fi
fi
git -C "$WT" pull --ff-only origin results 2>/dev/null || true
for root in data/modular_unlearning/*/; do
  root=${root%/}; tag=$(basename "$root")
  [[ -d $root/family ]] || continue
  dest="$WT/modular/$tag"; mkdir -p "$dest"
  "$PY" experiments/modular_unlearning/aggregate.py --out_root "$root" --dest "$dest" > "$dest/aggregate.log" 2>&1 \
    || echo "aggregate failed for $root" >> "$dest/aggregate.log"
  rm -rf "$dest/figures"
  MPLBACKEND=Agg "$PY" experiments/modular_unlearning/plot.py --src "$dest" --dest "$dest/figures" > "$dest/figures.log" 2>&1 \
    || echo "plot failed for $root" >> "$dest/figures.log"
done
mkdir -p "$WT/modular/rounds"
report="$WT/modular/rounds/$name.txt"
{
  echo "round $name, published $(date -Is), code $(git rev-parse --short HEAD)"
  for job in "$@"; do sacct -j "$job" -X --format=JobID%20,JobName%20,State,Elapsed,ExitCode -n; done
  for job in "$@"; do
    sacct -j "$job" -X -n --format=JobID%30,State | awk '$2 != "COMPLETED" {print $1}' | while read -r id; do
      for f in logs/*_"$id".out logs/*_"$id".err; do [[ -f $f ]] && { echo "== $f"; tail -n 20 "$f"; }; done
    done
  done
} > "$report" 2>&1
git -C "$WT" add -A
git -C "$WT" -c user.name="blackwell" -c user.email="na658@blackwell.local" commit -qm "results (modular): $name" || true
git -C "$WT" push -u origin results
