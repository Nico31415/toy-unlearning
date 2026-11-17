# E4: rescaling S in {0.1 .. 5} for F and R at two cells (L_full 5000, n_S 100 and 1000).
#  - mseaug_ph: the BatchNorm ResNet; --ph_rs_mode function multiplies only the head by S (the logits scale exactly by S).
#  - mseaug_ph_plain: a homogeneous network (VGG-11 without BatchNorm or biases), where every weight times S^(1/L)
#    multiplies the logits by S, the toy's rescaling exactly. It needs its own family (submitted here, smaller lr).
source experiments/cifar_unlearning/rounds/_phase.sh
S="0.1 0.3 0.5 1 1.5 2 5"
ns=$(echo $SEEDS | wc -w)
submit --array=0-$((2 * ns - 1)) $L phase $A --exps $EXP --seeds $SEEDS --Lfulls 5000 --methods F R \
  --ph_nS 100 1000 --ph_targets ybar --ph_arms full --ph_S $S $UL
P="--loss mse --augment --arch vgg11_plain --pt_lr 0.01 --pt_epochs 100 --pt_weight_decay 0 --out_root ${ROOT}_plain"
submit --array=0-$((4 * ns - 1)) $L family $P --exps $EXP --seeds $SEEDS --fam_Ls 0 4000 4900 5000
submit --array=0-$((2 * ns - 1)) --dependency=afterok:$LAST $L phase $P --exps $EXP --seeds $SEEDS --Lfulls 5000 \
  --methods F R --ph_nS 100 1000 --ph_targets ybar --ph_arms full --ph_S $S $UL
