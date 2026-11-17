# E5: shared features and the forget target. Three single forgotten classes (automobile 1, cat 3, frog 6; the sharing
# proxy is the probe accuracy of A(0), in family.csv), targets ybar, oracle and approx, n_S 300 and 3000 at L_full 3000.
# Own root (mseaug_ph_cls) with --n_probe 1000, since a single class has 5000 training images.
EXP_CLS="class-1 class-3 class-6"
ROOT=data/cifar_unlearning/mseaug_ph_cls
source experiments/cifar_unlearning/rounds/_phase.sh
A="$A --n_probe 1000"
ns=$(echo $SEEDS | wc -w)
submit --array=0-$((3 * ns * 7 - 1)) $L family $A --exps $EXP_CLS --seeds $SEEDS --fam_Ls 0 100 300 1000 2700 2900 3000
submit --array=0-$((3 * ns - 1)) --dependency=afterok:$LAST $L phase $A --exps $EXP_CLS --seeds $SEEDS --Lfulls 3000 \
  --methods F --ph_nS 100 300 3000 --ph_targets ybar oracle approx --ph_arms full --ph_S 1 0.1 $UL
