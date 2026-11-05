# S-rescaling sweep at mu_f = 0 (the retain-only unlearning of the toy paper's rescaling figure, Fig. 1d-e), MSE, BatchNorm
# unfrozen, pair-1-7, 5 seeds; same S grid and settings as rs_full (mseaug_rsu), own root so mu_f stays one value per root.
# Stage 1 and Retrain are copied from mseaug. Figures: mseaug_rsu0/figures/ (solid lines coloured by S, scale_sweep.png).
source experiments/cifar_unlearning/rounds/_reuse.sh
SIZES="2 4 10 20 50 100 300 1000"
E="--exps pair-1-7"
MU="--mu_fs 0"
BN="--ul_freeze_bn false --rl_freeze_bn false"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000 $BN"
UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000 $BN"
to=data/cifar_unlearning/mseaug_rsu0
A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $to"
reuse data/cifar_unlearning/mseaug $to 0 1 2 3 4
submit --array=0-4 $L references $A $E --theory --forget_sizes $SIZES $RL
for S in 0.125 0.25 0.5 1 2 4; do
  if [[ $S == 1 ]]; then RS=""; else RS="--rs_mode all --rs_S $S --rs_scale_bn_stats true"; fi
  submit --array=0-4 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL $RS
  submit --array=0-4 --dependency=afterok:$LAST $L relearn_sweep $A $E $MU --theory --ul_from 100 --forget_sizes $SIZES $UL $RL $RS
done
