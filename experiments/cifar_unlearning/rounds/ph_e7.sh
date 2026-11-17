# E7: relearning attacks from every saved stage-2 model of mseaug_ph and from the references (A(L_left), A(0), a fresh
# network), n_RL in {10, 30, 100, 300, 1000}. Run after the stage-2 rounds; E8 (probes) is part of every phase run.
source experiments/cifar_unlearning/rounds/_phase.sh
submit --array=0-$(( 4 * $(echo $SEEDS | wc -w) - 1 )) $L phase_relearn $A --exps $EXP --seeds $SEEDS --Lfulls $LFULLS \
  --ph_nRL 10 30 100 300 1000 $RL
