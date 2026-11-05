# The two theory arms (MSE: mseaug, CE: ceaug) again with BatchNorm NOT frozen in stages 2 and 3 (--ul_freeze_bn false
# --rl_freeze_bn false): the models train in ordinary train mode (batch statistics, running statistics updated), as
# in plain finetuning code (experiments/cifar). The from-scratch reference trained BN normally already.
# Roots mseaug_bn2 / ceaug_bn2 (the first version, *_bn, passed retain and forget images through separate BN batches and
# is wrong). One forward pass for both kinds of image; CE with lr 1e-3 as in ce_theory. Everything else as in r8/ul100 (MSE)
# and ce_theory (CE). Stage 1, Retrain and the stage-1 curve are copied from
# the frozen roots (they do not depend on this). Output roots data/cifar_unlearning/mseaug_bn2 and ceaug_bn2, pair-1-7, 5 seeds, mu_f 0 and 0.75 only.
# Compared with the frozen runs (mseaug, ceaug_lr3) in the figures bn_comparison_*: frozen dashed, unfrozen solid.
source experiments/cifar_unlearning/rounds/_reuse.sh
BN="--ul_freeze_bn false --rl_freeze_bn false"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000 $BN"
SIZES="2 4 10 20 50 100 300 1000"
E="--exps pair-1-7"
MU="--mu_fs 0 0.75"

for arm in mse ce; do
  from=data/cifar_unlearning/${arm}aug; to=${from}_bn2   # stage 1 is the same for CE in ceaug and ceaug_lr3
  reuse $from $to 0 1 2 3 4
  if [[ $arm == mse ]]; then
    A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $to"
    UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000 $BN"
  else
    A="--loss ce --augment --pt_epochs 100 --out_root $to"
    UL="--ul_forget_clip 5 --ul_lr 1e-3 --ul_max_steps 20000 --ul_eval_every 25 $BN"
  fi
  submit --array=0-9 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL
  UR=$LAST
  submit --array=0-4 $L references $A $E --theory --forget_sizes $SIZES $RL
  submit --array=0-9 --dependency=afterok:$UR $L relearn_sweep $A $E $MU --theory --ul_from 100 --forget_sizes $SIZES $UL $RL
done
