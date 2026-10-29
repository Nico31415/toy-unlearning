# CIFAR-10 unlearning and relearning

Pretrain, Unlearn, Relearn on CIFAR-10 with ResNet-18 (CIFAR stem), scored against a
Retrain control. The design is in the "Implementation plan: CIFAR-10 unlearning and
relearning" doc. Nothing outside this folder is modified; `functions/array_training.py`
is imported for the SLURM arrays.

## Pipeline

| Stage | Command (from the repo root) | Tasks |
| --- | --- | --- |
| download | `python experiments/cifar_unlearning/run.py download` (login node) | 1 |
| pretrain | `launch.sh pretrain` | 5 seeds |
| select_pairs | `launch.sh select_pairs` | 1 |
| retrain | `launch.sh retrain` | experiments x seeds |
| unlearn_relearn | `launch.sh unlearn_relearn` | experiments x seeds x 3 mu_f |
| references | `launch.sh references` | experiments x seeds |

Experiments are the pilot pairs in `<out_root>/pilot_pairs.json` plus `random`.
`launch.py STAGE --count` prints the array size.

`launch.sh` is set up for Cambridge CSD3 (A100 partition `ampere`, 12-hour SL3 limit,
one GPU per task). Before submitting, make its two marked edits: your GPU project
account (`mybalance` shows it, e.g. `XYZ123-SL3-GPU`) and your Python environment
activation. Activate the same environment on the login node too, since the commands
below run `python` there. Compute nodes have no internet, so run `download` on the
login node first. Then:

```bash
mkdir -p slurm
L=experiments/cifar_unlearning/launch.py
S=experiments/cifar_unlearning/launch.sh
n() { echo $(( $(python $L "$1" --count) - 1 )); }

python experiments/cifar_unlearning/run.py download
PT=$(sbatch --parsable --array=0-$(n pretrain) $S pretrain)
SP=$(sbatch --parsable --dependency=afterok:$PT --array=0-0 $S select_pairs)
# after select_pairs has finished (the array sizes below read pilot_pairs.json):
RT=$(sbatch --parsable --array=0-$(n retrain) $S retrain)
sbatch --array=0-$(n unlearn_relearn) $S unlearn_relearn
sbatch --dependency=afterok:$RT --array=0-$(n references) $S references
```

`unlearn_relearn` needs only Pretrain; `references` needs Retrain. Every run skips
work whose `result.json` exists, so failed or timed-out arrays can be resubmitted
unchanged. A run that fails a check (for example Pretrain not reaching train loss
1e-2) writes `failed.json` and exits non-zero.

Extra flags after the stage name go to `run.py`, for example
`sbatch ... $S unlearn_relearn --ul_lr 0.005`. Every hyperparameter is a flag; see
`config.py` for defaults and `python experiments/cifar_unlearning/run.py --help`.

## Outputs

```
data/cifar10/cifar10_uint8.pt                       cached dataset
data/cifar_unlearning/<loss>/
    pretrain/seed=S/{model.pt,result.json}
    pilot_pairs.json                                all 45 pairs ranked by confusion
    retrain/exp=E/seed=S/{model.pt,result.json}
    unlearn/exp=E/seed=S/mu_f=M/rescale=R/f=F/{model_fp16.pt,result.json}
    references/exp=E/seed=S/before.json
    references/exp=E/seed=S/model={pretrained,retrain,scratch}/f=F/result.json
```

Then:

```bash
python experiments/cifar_unlearning/aggregate.py   # CSVs in data/processed/cifar_unlearning/<loss>/
python experiments/cifar_unlearning/plot.py        # PNGs in figures/cifar_unlearning/<loss>/
```

## Unlearning stop rule and safety net

Plain -CE ascent has no minimum, so `unlearn` checks every `--ul_eval_every` steps:

- `--ul_stop_rule both` (default, as agreed): stop when 10-way accuracy on the n_f
  forget images is at most 1% and retain CE is below 1e-2.
- `--ul_stop_rule forget`: stop as soon as forget accuracy is at most 1%; retain
  damage is then a measured outcome.
- If retain CE exceeds `--ul_retain_loss_abort` (default 5, above chance at 2.3), the
  model is rolled back to the previous check and the run ends with status
  `collapsed`. A non-finite loss gives status `diverged`, with the same rollback.

Check `ul_status` in unlearn.csv before reading any result: only `success` runs met
the stop rule.

## Options kept for later

- Rescaling before unlearning: `--rs_mode {all,weights,body_head} --rs_S ... --rs_alpha_body ... --rs_alpha_head ... --rs_recalibrate_bn true`.
  Runs land in their own `rescale=` folder.
- MSE instead of cross-entropy for every stage: `--loss mse` (separate output root).
- Smaller fractions: `--fractions 0.001 0.003 ...`; all 45 pairs: `launch.py ... --all_pairs`.

## Tests

```bash
python -m pytest experiments/cifar_unlearning/tests -q
```

Synthetic data, CPU, about 30 seconds. They cover the splits, BatchNorm freezing,
rescaling, losses, metrics, the stop rule and rollback, every stage end to end
(including ResNet-18), aggregation, plots and the SLURM array mapping.
