# Round files

`bash experiments/cifar_unlearning/round.sh <name>` sources `<name>.sh` here. Every round comes in two versions:

- `<name>_quick.sh`: one seed (`--seeds 0`), three sizes (10 100 1000), mu_f 0 and 0.75. A few minutes;
  it runs every code path of the round, so it is the check that everything works.
- `<name>.sh`: the full grid (5 seeds, all sizes, all mu_f). Run only after the quick version looks right.

Both write to the same output root with the same settings, so the full round skips what the quick one
already did. Always `git pull` before `round.sh`, in case round.sh itself changed.


## Preflight

`round.sh` now checks a round before submitting anything (`PREFLIGHT=0` skips it, `SKIP_TESTS=1` skips the tests):
1. every third-party module the pipeline imports, including inside functions, imports in `$PY` (`preflight.py imports`);
2. the test suite passes in `$PY` (pytest is installed into the env if missing);
3. every task of every `submit` parses (`launch.py <stage> 0 --check`), and each `--array` matches the task count;
4. each distinct from-scratch configuration (family, pretrain, retrain) trains for warm-up + 300 steps on real CIFAR
   without a non-finite loss and with the loss falling (`preflight.py train`).
Not covered: stage-2 stability on a family the same round has not trained yet.
