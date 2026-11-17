# X2: phase diagram, None condition, method F target ybar, full fine-tune, every (L_full, n_S) cell. Needs mod_x0.
source experiments/modular_unlearning/rounds/_mod.sh
for f in $LFULLS; do
  submit --array=0-$(( $(nseeds) - 1 )) $L phase --exps $NONE --seeds $SEEDS --Lfulls $f --methods_array F $A \
    --nS $(ns_for $f) --targets ybar --arms full --S 1
done
