# Quick check of r8 (see r8.sh): seed 0, sizes 10 100 1000, mu_f 0 and 0.75. Same output root and settings as r8.sh.
R=data/cifar_unlearning/mseaug
A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $R"
UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000"
SIZES="10 100 1000"

[[ -d $R/pretrain_curve && ! -e $R/pretrain_curve_v1 ]] && mv $R/pretrain_curve $R/pretrain_curve_v1
for d in $R/unlearn/exp=pair-1-7/seed=0/mu_f=*/rescale=none/t=10/relearn_sweep; do
  [[ -d $d && ! -e ${d}_v1 ]] && mv "$d" "${d}_v1"
done

submit --array=0 $L reeval $A --exps pair-1-7 --seeds 0
REEVAL=$LAST
submit --array=0 $L pretrain_curve $A --seeds 0 --pt_sizes 0 100 1000 --pt_min_steps 3000
submit --array=0-1 --dependency=afterok:$REEVAL $L relearn_sweep $A --exps pair-1-7 --seeds 0 --theory --ul_from 10 \
  --forget_sizes $SIZES --mu_fs 0 0.75 $UL $RL
