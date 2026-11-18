# Unlearning modular addition in a one-layer Transformer

The transformer plan (X0-X6): forget addition mod 60 while retaining mod 59, mod 61 and a partner q whose shared
frequencies with mod 60 are the experimental variable (`p60-q49` none, `p60-q36` low, `p60-q40` medium, `p60-q30` high).
Same design as the CIFAR phase experiments: a pretraining family A(L), stage 2 with methods F / FR / R, targets ybar /
oracle and arms full / readout / linear, distances to both references, relearning attacks; plus Fourier read-outs (key
frequencies of the forget task, their surviving amplitude and phase in the embedding, realised overlap with the partner)
and probes at the "=" position (answer probe, Fourier R^2, frozen A(D) map).

| File | Contents |
| --- | --- |
| `core.py` | tasks and splits, the Transformer (attention written out so the linearised fine-tune works), losses, Fourier tools, probes |
| `run.py` | stages `family`, `phase`, `phase_relearn`; every flag documented in `--help` |
| `aggregate.py`, `plot.py` | CSV tables and figures |
| `launch.py`, `launch.sh`, `round.sh`, `publish.sh` | SLURM arrays and rounds, as for CIFAR (results under `modular/` on the `results` branch) |
| `rounds/mod_*.sh` | the rounds; shared settings in `rounds/_mod.sh` |

Order on blackwell (from the repo root):

    bash experiments/modular_unlearning/round.sh mod_quick   # first: pretraining time, generalisation
    bash experiments/modular_unlearning/round.sh mod_x0      # families (None grid, other conditions, kappa variants)
    bash experiments/modular_unlearning/round.sh mod_x1      # after x0: hidden, not deleted (headline) + relearning
    bash experiments/modular_unlearning/round.sh mod_x2      # phase diagram
    bash experiments/modular_unlearning/round.sh mod_x3      # shared features and the forget target
    bash experiments/modular_unlearning/round.sh mod_x4      # geometry (kappa, linearised removal)
    bash experiments/modular_unlearning/round.sh mod_x5      # rescaling + relearning

Pretraining: full-batch AdamW, lr 1e-3, beta2 0.98, weight decay 1.0, up to 1M steps. It stops when every retain
task reaches 99% held-out accuracy, every training row is classified correctly, and the forget task has either reached
99% or stopped improving for 20 evaluations (`--pt_patience`). The first quick round used Anguita et al.'s setting
(lr 4e-3, no weight decay): after 1M steps no task generalised (held-out accuracy 0.67-0.81 on the retain tasks), so
the models memorised and had no Fourier circuit. Weight decay is used in pretraining only; stage 2 and relearning use
none, so the implicit bias under study is untouched. No LayerNorm (`--ln pre` adds it, as in their codebase).

Tests: `python -m pytest experiments/modular_unlearning/tests -q` (CPU, about a minute).
