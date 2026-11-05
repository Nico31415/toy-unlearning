#!/bin/bash
# Aggregate every output root, draw its figures, and push the CSVs, figures and a job report to the `results` branch.
# Called by round.sh's final job:  publish.sh <round name> [job ids...]
# The results branch is kept in a separate worktree, so the code checkout is never touched.
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

# every output root that exists: data/cifar_unlearning/<root> -> results/<root>/*.csv
for root in data/cifar_unlearning/*/; do
  root=${root%/}; tag=$(basename "$root")
  [[ -d $root/pretrain || -d $root/unlearn || -d $root/references ]] || continue
  loss=ce; [[ -f $(ls "$root"/pretrain/seed=*/result.json 2>/dev/null | head -1) ]] && \
    loss=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['loss'])" "$(ls "$root"/pretrain/seed=*/result.json | head -1)")
  mkdir -p "$WT/$tag"
  "$PY" experiments/cifar_unlearning/aggregate.py --loss "$loss" --out_root "$root" --dest "$WT/$tag" \
    > "$WT/$tag/aggregate.log" 2>&1 || echo "aggregate failed for $root" >> "$WT/$tag/aggregate.log"
  # figures from the same tables, made here so they come back with the CSVs
  rm -rf "$WT/$tag/figures"
  MPLBACKEND=Agg "$PY" experiments/cifar_unlearning/plot.py --src "$WT/$tag" --dest "$WT/$tag/figures" \
    > "$WT/$tag/figures.log" 2>&1 || echo "plot failed for $root" >> "$WT/$tag/figures.log"
done

# job report: state of every job of the round, and the tail of the log of every task that did not complete
mkdir -p "$WT/rounds"
report="$WT/rounds/$name.txt"
{
  echo "round $name, published $(date -Is), code $(git rev-parse --short HEAD)"
  for job in "$@"; do
    sacct -j "$job" -X --format=JobID%20,JobName%20,State,Elapsed,ExitCode -n
  done
  for job in "$@"; do
    sacct -j "$job" -X -n --format=JobID%30,State | awk '$2 != "COMPLETED" {print $1}' | while read -r id; do
      for f in logs/*_"$id".out logs/*_"$id".err; do
        [[ -f $f ]] && { echo "== $f"; tail -n 20 "$f"; }
      done
    done
  done
} > "$report" 2>&1

git -C "$WT" add -A
git -C "$WT" -c user.name="blackwell" -c user.email="na658@blackwell.local" commit -qm "results: $name" || true
git -C "$WT" push -u origin results
