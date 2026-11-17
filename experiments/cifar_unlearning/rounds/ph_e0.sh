# E0 (and E1, which needs no new training): the pretraining family A(L) for every L in FAM_LS, 3 seeds.
# Gives n_learn, the seed-noise floor (eta) and every reference of E2-E8. 16 L x 3 seeds = 48 pretraining runs.
source experiments/cifar_unlearning/rounds/_phase.sh
n=$(echo $FAM_LS | wc -w); s=$(echo $SEEDS | wc -w)
submit --array=0-$((n * s - 1)) $L family $A --exps $EXP --seeds $SEEDS --fam_Ls $FAM_LS
