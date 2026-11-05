# Round files

`bash experiments/cifar_unlearning/round.sh <name>` sources `<name>.sh` here. Every round comes in two versions:

- `<name>_quick.sh`: one seed (`--seeds 0`), three sizes (10 100 1000), mu_f 0 and 0.75. A few minutes;
  it runs every code path of the round, so it is the check that everything works.
- `<name>.sh`: the full grid (5 seeds, all sizes, all mu_f). Run only after the quick version looks right.

Both write to the same output root with the same settings, so the full round skips what the quick one
already did. Always `git pull` before `round.sh`, in case round.sh itself changed.
