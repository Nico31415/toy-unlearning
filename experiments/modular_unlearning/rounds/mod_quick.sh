# Quick check before the full rounds: High condition, seed 0, three family members, one stage-2 cell for every method,
# both targets, the three arms and two rescalings, then relearning. Tells you pretraining time and whether every task
# generalises. Pretraining uses AdamW weight decay 1.0 (the first quick round, without it, memorised and never
# generalised in 1M steps); stage 2 and relearning use none.
ROOT=data/modular_unlearning/quick_wd
source experiments/modular_unlearning/rounds/_mod.sh
E="--exps p60-q30 --seeds 0"
submit --array=0-2 $L family $E --fam_Ls 0 1400 1800 $A
submit --array=0-2 --dependency=afterok:$LAST $L phase $E --Lfulls 1800 --methods_array F FR R $A \
  --nS 400 --targets ybar oracle --arms full readout linear --S 1 0.3
submit --array=0-0 --dependency=afterok:$LAST $L phase_relearn $E --Lfulls 1800 $A --nRL 30 100 300 900
