# Quick check of WHICH part of the rescaling matters (MSE, mu_f 0.75, BN unfrozen, seeds 0-1, sizes 10 and 100; settings as rs_full).
# `all` (every parameter times S) is already in mseaug_rsu; the logits then scale as S^2 on the weight part but only as S on the
# head bias, and BatchNorm cancels the scale of the conv weights. Two cleaner variants, S in {0.125, 0.5, 2, 4}:
#   head  only the final linear layer times S (--rs_alpha_head S): the logits scale by exactly S, the toy's "function scale"
#   conv  only the conv weights times S (--rs_alpha_body S, with exact BN stats): the function is unchanged, only the
#         effective learning rate of the convs changes (1/S^2) - the control for the BatchNorm confound
# Root mseaug_rsm. plot_scale.py writes mseaug_rsm/figures/scale_table.csv (column `mode`), including the mean logit norm before
# and after the rescaling (logit_norm_before / logit_norm_after) as a sanity check of the scaling.
source experiments/cifar_unlearning/rounds/_reuse.sh
SIZES="10 100"
E="--exps pair-1-7"
MU="--mu_fs 0.75"
BN="--ul_freeze_bn false --rl_freeze_bn false"
RL="--rl_lr 1e-3 --rl_eval_every 10 --rl_max_steps 20000 $BN"
UL="--ul_stop_rule loss --ul_lr 1e-3 --ul_eval_every 25 --ul_max_steps 20000 $BN"
to=data/cifar_unlearning/mseaug_rsm
A="--loss mse --augment --pt_lr 0.01 --pt_epochs 100 --out_root $to"
reuse data/cifar_unlearning/mseaug $to 0 1
for S in 0.125 0.5 2 4; do
  submit --array=0-1 $L unlearn_relearn $A --exps pair-1-7 $MU --theory --forget_sizes $SIZES $UL $RL --rs_mode body_head --rs_alpha_head $S
  submit --array=0-1 $L unlearn_relearn $A --exps pair-1-7 $MU --theory --forget_sizes $SIZES $UL $RL --rs_mode body_head --rs_alpha_body $S --rs_scale_bn_stats true
done
# the unscaled run and `all` with the same two seeds, so all variants share one root
submit --array=0-1 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL
for S in 0.125 0.5 2 4; do
  submit --array=0-1 $L unlearn_relearn $A $E $MU --theory --forget_sizes $SIZES $UL $RL --rs_mode all --rs_S $S --rs_scale_bn_stats true
done
