# What part of the rescaling matters (see rs_modes_quick): MSE, pair-1-7, 5 seeds, BatchNorm unfrozen, S in {0.125 .. 4}.
#   head  only the final linear layer times S (logits scale by exactly S; the toy's function scale)   roots mseaug_rsh (mu_f .75), mseaug_rsh0 (mu_f 0)
#   conv  only the conv weights times S, with exact BN stats (function unchanged; effective lr of the convs ~ 1/S^2; the control)  mseaug_rsc, mseaug_rsc0
# `all` (every parameter) is in mseaug_rsu / mseaug_rsu0 (rs_full / rs_mu0) for reference. S = 1 is the unscaled run. Same
# settings as rs_full; stage 1 and Retrain are copied from mseaug. Figures per root: <root>/figures/S_*.png and scale_sweep.png.
source experiments/cifar_unlearning/rounds/_reuse.sh
SIZES="2 4 10 20 50 100 300 1000"
E="--exps pair-1-7"
BN="--ul_freeze_bn false --rl_freeze_bn false"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000 $BN"
UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000 $BN"

for mode in h c; do
  for mu in 0.75 0; do
    root=mseaug_rs$mode; [[ $mu == 0 ]] && root=mseaug_rs${mode}0
    to=data/cifar_unlearning/$root
    A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $to"
    MU="--mu_fs $mu"
    reuse data/cifar_unlearning/mseaug $to 0 1 2 3 4
    submit --array=0-4 $L references $A $E --theory --forget_sizes $SIZES $RL
    for S in 0.125 0.25 0.5 1 2 4; do
      if [[ $S == 1 ]]; then RS=""
      elif [[ $mode == h ]]; then RS="--rs_mode body_head --rs_alpha_head $S"
      else RS="--rs_mode body_head --rs_alpha_body $S --rs_scale_bn_stats true"; fi
      submit --array=0-4 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL $RS
      submit --array=0-4 --dependency=afterok:$LAST $L relearn_sweep $A $E $MU --theory --ul_from 100 --forget_sizes $SIZES $UL $RL $RS
    done
  done
done
