# 8-way retain accuracy (A_r8: argmax over the kept classes' logits only) for the MSE theory arm, pair-1-7.
# 1. reeval: adds A_r8 to every saved model's record (unlearned, relearned, references and their starts).
# 2. pretrain_curve again (the first run saved no models): A_r8 per pair, models now saved.
# 3. relearn_sweep from n_UL = 10 again (it saved no models), after reeval has updated its source records.
# Old results of 2 and 3 are moved to *_v1, not deleted.
R=data/cifar_unlearning/mseaug
A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $R"
UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000"
SIZES="2 4 10 20 50 100 300 1000"

[[ -d $R/pretrain_curve && ! -e $R/pretrain_curve_v1 ]] && mv $R/pretrain_curve $R/pretrain_curve_v1
for d in $R/unlearn/exp=pair-1-7/seed=*/mu_f=*/rescale=none/t=10/relearn_sweep; do
  [[ -d $d && ! -e ${d}_v1 ]] && mv "$d" "${d}_v1"
done

submit --array=0-4 $L reeval $A --exps pair-1-7
REEVAL=$LAST
submit --array=0-4 $L pretrain_curve $A --pt_sizes 0 100 300 1000 3000 10000 --pt_min_steps 3000
submit --array=0-19 --dependency=afterok:$REEVAL $L relearn_sweep $A --exps pair-1-7 --theory --ul_from 10 \
  --forget_sizes $SIZES --mu_fs 0 0.25 0.5 0.75 $UL $RL
