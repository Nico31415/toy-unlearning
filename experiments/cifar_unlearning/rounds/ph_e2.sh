# E2: phase diagram of the forget loss: method F, target ybar, full fine-tune, no rescaling, every (L_full, n_S) cell.
# Two cells also record read-outs every 500 steps (E8 dynamics). Needs ph_e0.
source experiments/cifar_unlearning/rounds/_phase.sh
for f in $LFULLS; do
  submit --array=0-$(( $(echo $SEEDS | wc -w) - 1 )) $L phase $A --exps $EXP --seeds $SEEDS --Lfulls $f --methods F \
    --ph_nS $(ns_for $f) --ph_targets ybar --ph_arms full --ph_S 1 $UL
done
submit --array=0-0 $L phase $A --exps $EXP --seeds 0 --Lfulls 5000 --methods F --ph_nS 100 1000 --ph_targets ybar \
  --ph_arms full --ph_S 1 --ph_track_every 500 --out_root ${ROOT}_track --family_root $ROOT $UL
