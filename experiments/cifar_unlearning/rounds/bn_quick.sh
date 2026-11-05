# Quick check of bn (see bn.sh): seed 0, sizes 10 100 1000, mu_f 0 and 0.75, both arms.
source experiments/cifar_unlearning/rounds/_reuse.sh
BN="--ul_freeze_bn false --rl_freeze_bn false"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000 $BN"
SIZES="10 100 1000"
E="--exps pair-1-7 --seeds 0"
MU="--mu_fs 0 0.75"

for arm in mse ce; do
  from=data/cifar_unlearning/${arm}aug; to=${from}_bn2
  reuse $from $to 0
  if [[ $arm == mse ]]; then
    A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $to"
    UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000 $BN"
  else
    A="--loss ce --augment --pt_epochs 100 --out_root $to"
    UL="--ul_forget_clip 5 --ul_lr 1e-3 --ul_max_steps 20000 --ul_eval_every 25 $BN"
  fi
  submit --array=0-1 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL
  UR=$LAST
  submit --array=0 $L references $A $E --theory --forget_sizes $SIZES $RL
  submit --array=0-1 --dependency=afterok:$UR $L relearn_sweep $A $E $MU --theory --ul_from 10 --forget_sizes $SIZES $UL $RL
  submit --array=0-1 --dependency=afterok:$UR $L relearn_sweep $A $E $MU --theory --ul_from 100 --forget_sizes $SIZES $UL $RL
done
