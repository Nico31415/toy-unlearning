# E10: the keep + forget loss. Every image of D keeps a term on its true label (weight mu_r) and S adds a forget term
# (weight mu_f), so each forget image's target is (1 - omega) onehot + omega g, omega = mu_f / (mu_r + mu_f).
# omega in {0.25, 0.5, 0.75, 1}, targets ybar and oracle, F and FR, kappa 1 and 0.1, at L_full 5000, |S| in {100, 1000}.
# Uses the E0 family (mseaug_ph). Toy prediction: the endpoint moves continuously with omega (the effective teacher on S),
# from A(D) at omega = 0 to pure relabelling at omega = 1.
source experiments/cifar_unlearning/rounds/_phase.sh
ns=$(echo $SEEDS | wc -w)
submit --array=0-$((2 * ns - 1)) $L phase $A --exps $EXP --seeds $SEEDS --Lfulls 5000 --methods F FR --ph_nS 100 1000 \
  --ph_targets ybar oracle --ph_omegas 0.25 0.5 0.75 1 --ph_arms full --ph_S 1 0.1 $UL
