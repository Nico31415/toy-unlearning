# X4: geometry. Kappa 0.1, 1, 10 (pretraining embedding scale), None and High conditions, full and linearised removal,
# methods F and R, n_S 400 and 1800, S 1 and 0.1. Needs mod_x0.
source experiments/modular_unlearning/rounds/_mod.sh
for k in 1 0.1 10; do
  r=$ROOT; extra=""; [[ $k != 1 ]] && { r=${ROOT}_k$k; extra="--kappa $k"; }
  submit --array=0-$((2 * 2 * $(nseeds) - 1)) $L phase --exps $NONE p60-q30 --seeds $SEEDS --Lfulls 1800 \
    --methods_array F R --out_root $r $extra --nS 400 1800 --targets ybar --arms full linear --S 1 0.1
done
