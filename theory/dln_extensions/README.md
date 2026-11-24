# Diagonal-network theory: extensions (23 Nov 2026)

Cross-entropy, early stopping, SGD and explicit regularisers in the diagonal-network model of unlearning.
The maths is in `notes/extensions_maths.pdf` (one template and its variants). `notes/losses.pdf` is the setup document.

| folder | contents |
|---|---|
| `core/` | Bregman projection (`dln_setup.py`), pretraining channel (`se.py`), stage-2 theory with reused samples and rescaling (`se_reuse.py`) |
| `ce/` | `ce_tests.py` (`project`, `flow`, hard-label tests), `se_sign.py` (sign-label channel), `soft.py`, `het.py`, `diag*.py`, `ce_grid.py` (ascent losses), `ksweep*.py` (theory vs simulation against k), `dmft2.py` + `es_fig.py` (two-stage dynamical solver, early stopping) |
| `wd/` | `se_wd.py` (channel for a regulariser kept on), `switchoff.py`, `wd_fig.py` (switched-off weight decay = rescaling), `left2.py` (one-sided decay), `left1.py` (non-convex penalty, negative result), `reg_fig.py`, `more_seeds.py`, `analyse_more.py`, `overlap_check.py` (many-seed checks) |
| `crit/` | follow-up checks: `tamper.py` (ascent losses: cosine to the reference and a relearning attack), `se_blocks.py` (regulariser channel with retain/shared/forget blocks), `block_identity.py`, `onesided_gd.py` (identities by gradient descent), `sgd_kappa.py` (how uniform SGD's shrink factor is) |
| `hard/` | hard-label cross-entropy: `se_logistic.py` (end point with a penalty kept on, large d), `dmft_nl.py` (exact large-d training dynamics without a penalty), `dmft_nl2.py` (two stages: ascent, NPO, flipped labels), `nl_check.py`, `lam_sweep.py`, `hard_fig.py` |
| `final/` | `multi.py` (several trainings on the same data: static large-d end points with blocks, anchors, rescaling, references), `multi_test.py`, `sgd_dmft.py` (large-d SGD trajectory), `es_cov.py` (early stopping with other targets/anchor, budget), `whp_tv.py` (task vector and WHP, hard labels), `reconcile.py` (soft vs hard pretraining under ascent), `dmft_nl2.py` (with sample ratios as arguments), logs of the runs. |
| `sgd/` | `check.py` (batch-size scaling, stage-2 rescaling identity) |
| `figures/` | the figures used in the notes |

Running: each script adds `core/`, `ce/`, `wd/` and `sgd/` to the import path itself, so it runs from any directory (tested by importing from outside the repository).
`wd/left1.py` and `wd/left2.py` have no `__main__` guard: importing them runs them. Pickled results are not included; every script regenerates its own.
`notes/*.tex` expect the figures next to them (copy from `figures/`).
