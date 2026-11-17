# X5: rescaling. None condition, two cells (n_S 50 and 400 at L_full 1800), S 0.1 .. 5 on the unembedding (the logits
# scale exactly), methods F and R, then relearning. Needs mod_x0. (X6, the probes, is part of every phase run.)
source experiments/modular_unlearning/rounds/_mod.sh
submit --array=0-$((2 * $(nseeds) - 1)) $L phase --exps $NONE --seeds $SEEDS --Lfulls 1800 --methods_array F R $A \
  --nS 50 400 --targets ybar --arms full --S 0.1 0.3 0.5 1 1.5 2 5
submit --array=0-$(( $(nseeds) - 1 )) --dependency=afterok:$LAST $L phase_relearn --exps $NONE --seeds $SEEDS \
  --Lfulls 1800 $A --nRL 30 100 300 900
