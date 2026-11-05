# Quick check of pt10: stage-1 pretraining on 10 and 30 images, seed 0 only (the curve so far starts at 100).
R=data/cifar_unlearning/mseaug
A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $R"
submit --array=0 $L pretrain_curve $A --seeds 0 --pt_sizes 10 30 --pt_min_steps 3000
