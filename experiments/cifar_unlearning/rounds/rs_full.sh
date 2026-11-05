# The S-rescaling sweep (Lippl & Lindsey's finetune_scaling: every parameter times S, `--rs_mode all`), pair-1-7, 5 seeds,
# mu_f 0.75, both arms, BatchNorm frozen (roots ${arm}aug_rsf) and unfrozen (${arm}aug_rsu). Everything else as in the
# unscaled runs (r8/ul100 for MSE, ce_theory for CE); S = 1 is the unscaled run (no rs flags). Scaled runs also scale the BN
# running statistics exactly (--rs_scale_bn_stats true): needed with frozen BN, harmless with unfrozen (rs_quick).
# Grids from rs_quick: MSE {0.125 .. 4}; CE {0.5 .. 1.4} (outside it CE collapses or cannot unlearn), CE capped at 5000 steps.
# Stage 1, Retrain and the stage-1 curve are copied from mseaug / ceaug. Figures: comparison_S/<loss>/ (frozen dashed,
# unfrozen solid, colour = S) and <root>/figures/scale_sweep.png.
source experiments/cifar_unlearning/rounds/_reuse.sh
SIZES="2 4 10 20 50 100 300 1000"
E="--exps pair-1-7"
MU="--mu_fs 0.75"

for arm in mse ce; do
  if [[ $arm == mse ]]; then SCALES="0.125 0.25 0.5 1 2 4"; else SCALES="0.5 0.7 1 1.4"; fi
  for bn in f u; do
    to=data/cifar_unlearning/${arm}aug_rs$bn
    reuse data/cifar_unlearning/${arm}aug $to 0 1 2 3 4
    if [[ $bn == u ]]; then BN="--ul_freeze_bn false --rl_freeze_bn false"; else BN=""; fi
    RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000 $BN"
    if [[ $arm == mse ]]; then
      A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $to"
      UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000 $BN"
    else
      A="--loss ce --augment --pt_epochs 100 --out_root $to"
      UL="--ul_forget_clip 5 --ul_lr 1e-3 --ul_max_steps 5000 --ul_eval_every 25 $BN"
    fi
    submit --array=0-4 $L references $A $E --theory --forget_sizes $SIZES $RL
    for S in $SCALES; do
      if [[ $S == 1 ]]; then RS=""; else RS="--rs_mode all --rs_S $S --rs_scale_bn_stats true"; fi
      submit --array=0-4 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL $RS
      submit --array=0-4 --dependency=afterok:$LAST $L relearn_sweep $A $E $MU --theory --ul_from 100 --forget_sizes $SIZES $UL $RL $RS
    done
  done
done
