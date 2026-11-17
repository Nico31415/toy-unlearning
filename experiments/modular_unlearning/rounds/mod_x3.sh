# X3: shared features and the forget target. All four overlap conditions, L_full 1800, n_S 400 and 1800, methods F
# (targets ybar and oracle), FR and R; S 1 and 0.1. Needs mod_x0.
source experiments/modular_unlearning/rounds/_mod.sh
submit --array=0-$((4 * 3 * $(nseeds) - 1)) $L phase --exps $ALL --seeds $SEEDS --Lfulls 1800 --methods_array F FR R $A \
  --nS 400 1800 --targets ybar oracle --arms full --S 1 0.1
