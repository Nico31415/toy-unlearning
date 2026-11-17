# E6: removal geometry. Full, readout-only and linearised fine-tunes from the same A(D), methods F and R, S 1 and 0.1, at
# L_full 5000 with n_S 100 and 1000. The linearised arm costs about 3x. Needs ph_e0.
source experiments/cifar_unlearning/rounds/_phase.sh
submit --array=0-$((2 * $(echo $SEEDS | wc -w) - 1)) $L phase $A --exps $EXP --seeds $SEEDS --Lfulls 5000 --methods F R \
  --ph_nS 100 1000 --ph_targets ybar --ph_arms full readout linear --ph_S 1 0.1 $UL
