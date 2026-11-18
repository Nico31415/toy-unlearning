# Shared settings of the phase rounds (ph_*.sh), the CIFAR plan's E0-E8. Sourced by those rounds; submits nothing.
# Root mseaug_ph: MSE loss, augmentation, stage 1 as in mseaug (lr 0.01, 100 epochs), BatchNorm unfrozen in stages 2 and 3.
# The forgotten classes: pair-1-7 (as in every earlier round). For one class use EXP=class-1 and a root with --n_probe 1000
# (see ph_e5.sh): class-K has 5000 training images, so 2000 probe images would leave too small a forget pool.
EXP=${EXP:-pair-1-7}
ROOT=${ROOT:-data/cifar_unlearning/mseaug_ph}
SEEDS=${SEEDS:-"0 1 2"}
A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $ROOT"
BN="--ul_freeze_bn false --rl_freeze_bn false"
UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000 $BN"
# Relearning attacks: 10-way loss on forget images with true labels, a fixed budget of 1000 steps for every model
# (curve at 10, 20, 50, 100, 200, 500, 1000), so models are compared at equal attack cost rather than at the step a
# loss threshold happened to be reached (in ph_quick every pretrained model stopped within 50-120 steps).
RL="--rl_lr 1e-3 --rl_eval_every 50 --rl_max_steps 1000 --rl_stop_rule budget --rl_classes all $BN"
# |S_full| values and the pretraining family: every L_full, every L_full - 10 and L_full - 100 (small forget sets),
# and a log grid below, so that A(D minus S) = A(L_full - n_S) is a family member for every cell.
LFULLS=${LFULLS:-"300 1000 3000 5000"}
GRID="0 10 30 100 300 1000 3000 5000"
FAM_LS=$(for L in $GRID $(for f in $LFULLS; do echo $((f - 10)) $((f - 100)); done); do echo $L; done | sort -n | uniq | tr '\n' ' ')
# n_S for one L_full: L_full minus every family member below it (n_S = L_full means S = S_full, A(D minus S) = A(0))
ns_for() { local f=$1 L; for L in $FAM_LS; do (( L < f )) && echo $((f - L)); done | sort -n | tr '\n' ' '; }
