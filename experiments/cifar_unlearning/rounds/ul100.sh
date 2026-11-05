# Panel (d) from n_UL = 100 instead of 10 for the MSE theory arm (mseaug): relearn the models unlearned with 100 rows
# (saved by unlearn_relearn) with every n_RL. Same settings as the sweep from 10 (r8.sh). The CE arm gets the same
# sweep in ce_theory.sh. plot.py draws panel (d) from the largest n_UL present, so the figures switch to 100 once this
# has run (the sweep from 10 stays in the table).
R=data/cifar_unlearning/mseaug
A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $R"
UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000"
SIZES="2 4 10 20 50 100 300 1000"

submit --array=0-19 $L relearn_sweep $A --exps pair-1-7 --theory --ul_from 100 --forget_sizes $SIZES \
  --mu_fs 0 0.25 0.5 0.75 $UL $RL
