# Quick check of the S-rescaling of Pretrained before unlearning (Lippl & Lindsey's finetune_scaling: every parameter times S,
# `--rs_mode all`), BatchNorm unfrozen, seed 0, mu_f 0.75, sizes 10 and 100, both arms (settings as bn.sh).
# Each S runs without and with the exact BN running-stat fix (--rs_scale_bn_stats true), plus the unscaled run (S = 1, no flags)
# and S = 1 with the fix (must match the unscaled one). Roots mseaug_rs / ceaug_rs; stage 1 is copied from mseaug / ceaug.
# plot_scale.py (called by publish.sh) draws the accuracies against S.
source experiments/cifar_unlearning/rounds/_reuse.sh
BN="--ul_freeze_bn false --rl_freeze_bn false"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000 $BN"
SIZES="10 100"
E="--exps pair-1-7 --seeds 0"
MU="--mu_fs 0.75"

for arm in mse ce; do
  from=data/cifar_unlearning/${arm}aug; to=${from}_rs
  reuse $from $to 0
  if [[ $arm == mse ]]; then
    A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $to"
    UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000 $BN"
  else
    A="--loss ce --augment --pt_epochs 100 --out_root $to"
    UL="--ul_forget_clip 5 --ul_lr 1e-3 --ul_max_steps 20000 --ul_eval_every 25 $BN"
  fi
  submit --array=0 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL
  submit --array=0 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL --rs_mode all --rs_S 1 --rs_scale_bn_stats true
  for S in 0.125 0.5 2; do
    submit --array=0 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL --rs_mode all --rs_S $S
    submit --array=0 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL --rs_mode all --rs_S $S --rs_scale_bn_stats true
  done
done
