# The MSE theory arm (mseaug) again with cross-entropy, to compare the two losses. Same pipeline and data usage
# (--theory: n rows from all classes, stage 3 the forget-only attack on fresh rows), same augmentation, 100
# pretraining epochs, relearning settings and sizes; only the loss differs, and what that forces:
#   stage 1: CE, the CE default lr 0.1 (MSE diverges above 0.02, so it used 0.01);
#   stage 2: mu_r*CE_retain - mu_f*min(CE_forget, 5) with row-share weights, lr 1e-3 and the 20000-step cap as in the
#            MSE arm (lr 0.01 diverged in 69 of 120 runs at mu_f > 0), stop when forget accuracy <= 1% and retain CE
#            < 1e-2 on the n rows (mu_f = 0 never reaches it and runs the cap);
#   stage 3: KL to onehot on forget rows and to the uniform distribution on retain rows (the CE counterpart of the
#            zero vector), stop at loss < 1e-2.
# BatchNorm frozen in stages 2 and 3. Stage 1, Retrain and the stage-1 curve are copied from ceaug (they do not
# depend on the stage-2 settings). Output root data/cifar_unlearning/ceaug_lr3, pair-1-7, 5 seeds.
source experiments/cifar_unlearning/rounds/_reuse.sh
R=data/cifar_unlearning/ceaug_lr3
reuse data/cifar_unlearning/ceaug $R 0 1 2 3 4
A="--loss ce --augment --pt_epochs 100 --out_root $R"
UL="--ul_forget_clip 5 --ul_lr 1e-3 --ul_max_steps 20000 --ul_eval_every 25"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000"
SIZES="2 4 10 20 50 100 300 1000"
E="--exps pair-1-7"

submit --array=0-19 $L unlearn_relearn $A $E --theory --forget_sizes $SIZES --mu_fs 0 0.25 0.5 0.75 $UL $RL
UR=$LAST
submit --array=0-4 $L references $A $E --theory --forget_sizes $SIZES $RL
submit --array=0-19 --dependency=afterok:$UR $L relearn_sweep $A $E --theory --ul_from 10 --forget_sizes $SIZES \
  --mu_fs 0 0.25 0.5 0.75 $UL $RL
submit --array=0-19 --dependency=afterok:$UR $L relearn_sweep $A $E --theory --ul_from 100 --forget_sizes $SIZES \
  --mu_fs 0 0.25 0.5 0.75 $UL $RL
