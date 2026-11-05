# Sourced by rounds that rerun stages 2 and 3 with other settings in a new output root. Stage 1, Retrain and the
# stage-1 size curve do not depend on those settings, so they are copied from the old root instead of retrained.
# reuse <from root> <to root> <seed>...: copies pair-1-7's pretrain, retrain, split and pretrain_curve of each seed
# that has finished (result.json present) in <from root>; existing files in <to root> are never overwritten.
reuse() {
  local from=$1 to=$2 s p; shift 2
  for s in "$@"; do
    [[ -f $from/pretrain/seed=$s/result.json && -f $from/retrain/exp=pair-1-7/seed=$s/result.json ]] || {
      echo "reuse: seed $s is not finished in $from, skipped (stage 2 will fail for it)"; continue; }
    for p in pretrain/seed=$s retrain/exp=pair-1-7/seed=$s splits/exp=pair-1-7/seed=$s.npz pretrain_curve/seed=$s; do
      if [[ -e $from/$p && ! -e $to/$p ]]; then mkdir -p "$to/$(dirname "$p")"; cp -r "$from/$p" "$to/$p"; fi
    done
  done
}
