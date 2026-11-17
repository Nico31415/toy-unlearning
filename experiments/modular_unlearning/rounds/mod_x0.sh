# X0: pretraining families. None condition: the full grid (n_learn, seed floor, every reference). Other conditions: the
# members X3 needs (0, 1400, 1800). Kappa variants (0.1 and 10) for X4, None and High, in their own roots.
source experiments/modular_unlearning/rounds/_mod.sh
n=$(echo $FAM_LS | wc -w)
submit --array=0-$((n * $(nseeds) - 1)) $L family --exps $NONE --seeds $SEEDS --fam_Ls $FAM_LS $A
submit --array=0-$((9 * $(nseeds) - 1)) $L family --exps p60-q36 p60-q40 p60-q30 --seeds $SEEDS --fam_Ls 0 1400 1800 $A
for k in 0.1 10; do
  submit --array=0-$((6 * $(nseeds) - 1)) $L family --exps $NONE p60-q30 --seeds $SEEDS --fam_Ls 0 1400 1800 \
    --out_root ${ROOT}_k$k --kappa $k
done
