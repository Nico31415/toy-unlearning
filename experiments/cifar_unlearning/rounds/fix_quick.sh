# Quick checks of two fixes (each in its own output root, stage 1 and Retrain copied from the old root).
# 1. CE arm, frozen BN: unlearning with lr 1e-3 (as the MSE arm) and the 20000-step cap, instead of lr 0.01 and 5000
#    steps, which diverged in 69 of 120 runs at mu_f > 0 (sizes 4 to 100 above all). Seeds 0-4, mu_f 0.5, small
#    sizes, where the divergences were. Root ceaug_lr3.
# 2. MSE arm, BN unfrozen: retain and forget images in one forward pass (separate passes gave every batch the
#    statistics of one kind of image and the stop rule never converged). Seed 0, mu_f 0 and 0.75, sizes 10 100 1000.
#    Root mseaug_bn2.
source experiments/cifar_unlearning/rounds/_reuse.sh
RLF="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000"

to=data/cifar_unlearning/ceaug_lr3
reuse data/cifar_unlearning/ceaug $to 0 1 2 3 4
submit --array=0-4 $L unlearn_relearn --loss ce --augment --pt_epochs 100 --out_root $to --exps pair-1-7 \
  --theory --forget_sizes 2 4 10 20 50 100 --mu_fs 0.5 --ul_forget_clip 5 --ul_lr 1e-3 --ul_max_steps 20000 \
  --ul_eval_every 25 $RLF

to=data/cifar_unlearning/mseaug_bn2
BN="--ul_freeze_bn false --rl_freeze_bn false"
reuse data/cifar_unlearning/mseaug $to 0
submit --array=0-1 $L unlearn_relearn --loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $to \
  --exps pair-1-7 --seeds 0 --theory --forget_sizes 10 100 1000 --mu_fs 0 0.75 --ul_stop_rule loss --ul_lr 1e-3 \
  --ul_eval_every 25 --ul_max_steps 20000 $BN $RLF
