# X1 (headline): hidden, not deleted. None condition, L_full 1800, n_S 50, 400 and 1800, method F with target ybar,
# S 1 and 0.3; every stage-2 run records accuracy and the surviving amplitude and phase every 500 steps. Then relearning
# from every U and from the references. Needs mod_x0.
source experiments/modular_unlearning/rounds/_mod.sh
submit --array=0-$(( $(nseeds) - 1 )) $L phase --exps $NONE --seeds $SEEDS --Lfulls 1800 --methods_array F $A \
  --nS 50 400 1800 --targets ybar --arms full --S 1 0.3
submit --array=0-$(( $(nseeds) - 1 )) --dependency=afterok:$LAST $L phase_relearn --exps $NONE --seeds $SEEDS \
  --Lfulls 1800 $A --nRL 30 100 300 900
