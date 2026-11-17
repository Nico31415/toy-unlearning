# E3: which loss. Methods F, FR, R (target ybar), no rescaling and S = 0.1, at L_full 1000 and 5000. Needs ph_e0.
source experiments/cifar_unlearning/rounds/_phase.sh
for f in 1000 5000; do
  submit --array=0-$(( 3 * $(echo $SEEDS | wc -w) - 1 )) $L phase $A --exps $EXP --seeds $SEEDS --Lfulls $f \
    --methods F FR R --ph_nS $(ns_for $f) --ph_targets ybar --ph_arms full --ph_S 1 0.1 $UL
done
