# E9: from iid deletion to class removal. mix-7-ALPHA: the forget pool interleaves horse images (class 7, absent from the
# rest of the training set) and iid images of the other nine classes at ratio ALPHA, so S is an ALPHA fraction horse.
# ALPHA = 0: iid deletion only (untraining is the only meaningful goal); ALPHA = 1: the class experiment.
# Fixed L_full = 3000 and |S| in {300, 1000, 3000} across ALPHA: only the composition changes.
# Read-outs on S's own training images: membership AUC, loss gap, r_S_ut (0 = retrained, 1 = A(D)), per part.
# Own root (mseaug_ph_mix) with --n_probe 1000 and --n_forget 3000 (the iid part of the pool).
EXP_MIX="mix-7-0 mix-7-0.25 mix-7-0.5 mix-7-1"
ROOT=data/cifar_unlearning/mseaug_ph_mix
source experiments/cifar_unlearning/rounds/_phase.sh
A="$A --n_probe 1000 --n_forget 3000"
ns=$(echo $SEEDS | wc -w)
FAM="0 100 300 1000 2000 2700 3000"
submit --array=0-$((4 * ns * 7 - 1)) $L family $A --exps $EXP_MIX --seeds $SEEDS --fam_Ls $FAM
submit --array=0-$((4 * ns * 3 - 1)) --dependency=afterok:$LAST $L phase $A --exps $EXP_MIX --seeds $SEEDS \
  --Lfulls 3000 --methods F FR R --ph_nS 300 1000 3000 --ph_targets ybar oracle --ph_arms full --ph_S 1 0.1 $UL
