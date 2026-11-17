# Quick check of the phase pipeline before the full rounds: seed 0, three family members, one stage-2 cell per method,
# every target, the three arms, two rescalings, then relearning. About 4 GPU-hours. Root mseaug_phq.
ROOT=data/cifar_unlearning/mseaug_phq
source experiments/cifar_unlearning/rounds/_phase.sh
E="--exps $EXP --seeds 0"
submit --array=0-2 $L family $A $E --fam_Ls 0 900 1000
FAM=$LAST
submit --array=0-2 --dependency=afterok:$FAM $L phase $A $E --Lfulls 1000 --methods F FR R \
  --ph_nS 100 --ph_targets ybar oracle --ph_arms full readout linear --ph_S 1 0.1 --ph_track_every 2000 $UL
submit --array=0-0 --dependency=afterok:$LAST $L phase_relearn $A $E --Lfulls 1000 --ph_nRL 10 100 1000 $RL
