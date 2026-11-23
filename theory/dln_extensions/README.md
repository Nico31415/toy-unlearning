# Diagonal-network theory: extensions (23 Nov 2026)

Cross-entropy, early stopping, SGD and explicit regularisers in the diagonal-network model of unlearning.
The maths is in `notes/extensions_maths.pdf` (one template and its variants). `notes/losses.pdf` is the setup document.

| folder | contents |
|---|---|
| `core/` | Bregman projection (`dln_setup.py`), pretraining channel (`se.py`), stage-2 theory with reused samples and rescaling (`se_reuse.py`) |
| `ce/` | `ce_tests.py` (`project`, `flow`, hard-label tests), `se_sign.py` (sign-label channel), `soft.py`, `het.py`, `diag*.py`, `ce_grid.py` (ascent losses), `ksweep*.py` (theory vs simulation against k), `dmft2.py` + `es_fig.py` (two-stage dynamical solver, early stopping) |
| `wd/` | `se_wd.py` (channel for a regulariser kept on), `switchoff.py`, `wd_fig.py` (switched-off weight decay = rescaling), `left2.py` (one-sided decay), `left1.py` (non-convex penalty, negative result), `reg_fig.py`, `more_seeds.py`, `analyse_more.py`, `overlap_check.py` (many-seed checks) |
| `sgd/` | `check.py` (batch-size scaling, stage-2 rescaling identity) |
| `figures/` | the figures used in the notes |

Running: the scripts were written with absolute import paths (`/home/claude/out/setup2`, `/home/claude/out/ce`). Either edit the
`sys.path.insert` lines at the top of each script, or run with `PYTHONPATH=core:ce:wd` from this folder. `wd/left1.py` and
`wd/left2.py` have no `__main__` guard: importing them runs them. Pickled results are not included; every script regenerates its own.
`notes/*.tex` expect the figures next to them (copy from `figures/`).
