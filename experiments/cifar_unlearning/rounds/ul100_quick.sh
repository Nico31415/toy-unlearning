# Quick check of ul100 (see ul100.sh), for both arms: seed 0, sizes 10 100 1000, mu_f 0 and 0.75, from n_UL = 100.
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000"
SIZES="10 100 1000"

A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root data/cifar_unlearning/mseaug"
UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000"
submit --array=0-1 $L relearn_sweep $A --exps pair-1-7 --seeds 0 --theory --ul_from 100 --forget_sizes $SIZES \
  --mu_fs 0 0.75 $UL $RL

A="--loss ce --augment --pt_epochs 100 --out_root data/cifar_unlearning/ceaug"
UL="--ul_forget_clip 5 --ul_eval_every 25"
submit --array=0-1 $L relearn_sweep $A --exps pair-1-7 --seeds 0 --theory --ul_from 100 --forget_sizes $SIZES \
  --mu_fs 0 0.75 $UL $RL
