# Quick check of ce_theory (see ce_theory.sh): seed 0, sizes 10 100 1000, mu_f 0 and 0.75.
R=data/cifar_unlearning/ceaug
A="--loss ce --augment --pt_epochs 100 --out_root $R"
UL="--ul_forget_clip 5 --ul_eval_every 25"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000"
SIZES="10 100 1000"
E="--exps pair-1-7 --seeds 0"

submit --array=0 $L pretrain $A --seeds 0
PT=$LAST
submit --array=0 $L retrain $A $E
RT=$LAST
submit --array=0 $L pretrain_curve $A --seeds 0 --pt_sizes 10 100 1000 --pt_min_steps 3000
submit --array=0-1 --dependency=afterok:$PT:$RT $L unlearn_relearn $A $E --theory --forget_sizes $SIZES --mu_fs 0 0.75 $UL $RL
UR=$LAST
submit --array=0 --dependency=afterok:$PT:$RT $L references $A $E --theory --forget_sizes $SIZES $RL
submit --array=0-1 --dependency=afterok:$UR $L relearn_sweep $A $E --theory --ul_from 10 --forget_sizes $SIZES \
  --mu_fs 0 0.75 $UL $RL
