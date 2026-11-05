# Stage-1 pretraining on 10 and 30 images, 5 seeds, so the pretraining curve starts at 10^1 (it started at 100).
# Same settings as r8, which skips nothing here: these sizes are new.
R=data/cifar_unlearning/mseaug
A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $R"
submit --array=0-4 $L pretrain_curve $A --pt_sizes 10 30 --pt_min_steps 3000
