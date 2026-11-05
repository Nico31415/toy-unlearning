# The MSE theory arm (mseaug) again with cross-entropy, to compare the two losses. Same pipeline and data usage
# (--theory: n rows from all classes, stage 3 the forget-only attack on fresh rows), same augmentation, 100
# pretraining epochs, relearning settings and sizes; only the loss differs, and what that forces:
#   stage 1: CE, the CE default lr 0.1 (MSE diverges above 0.02, so it used 0.01);
#   stage 2: mu_r*CE_retain - mu_f*min(CE_forget, 5) with row-share weights, stop when forget accuracy <= 1% and
#            retain CE < 1e-2 on the n rows (mu_f = 0 never reaches it and runs the 5000-step cap);
#   stage 3: KL to onehot on forget rows and to the uniform distribution on retain rows (the CE counterpart of the
#            zero vector), stop at loss < 1e-2.
# Output root data/cifar_unlearning/ceaug, pair-1-7, 5 seeds.
R=data/cifar_unlearning/ceaug
A="--loss ce --augment --pt_epochs 100 --out_root $R"
UL="--ul_forget_clip 5 --ul_eval_every 25"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000"
SIZES="2 4 10 20 50 100 300 1000"
E="--exps pair-1-7"

submit --array=0-4 $L pretrain $A
PT=$LAST
submit --array=0-4 $L retrain $A $E
RT=$LAST
submit --array=0-4 $L pretrain_curve $A --pt_sizes 10 30 100 300 1000 3000 10000 --pt_min_steps 3000
submit --array=0-19 --dependency=afterok:$PT:$RT $L unlearn_relearn $A $E --theory --forget_sizes $SIZES \
  --mu_fs 0 0.25 0.5 0.75 $UL $RL
UR=$LAST
submit --array=0-4 --dependency=afterok:$PT:$RT $L references $A $E --theory --forget_sizes $SIZES $RL
submit --array=0-19 --dependency=afterok:$UR $L relearn_sweep $A $E --theory --ul_from 10 --forget_sizes $SIZES \
  --mu_fs 0 0.25 0.5 0.75 $UL $RL
