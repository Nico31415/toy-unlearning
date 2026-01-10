import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray

"""
Diagonal PT→FT k-sweep experiment (gamma × lambda_PT × c_PT × teacher beta structure).

This script is a thin ArgparseArray wrapper around `experiments/diagonal/diagonal_network_finetune.py`.
It assumes the corresponding pretrained models already exist at:
  data/diagonal/pretrain/seed={seed}--active_dim=40--c={c:.1e}--lmda={lmda:.10f}--init_method={init_method}/model.pt

Key theory mapping (repo → theory):
  - gamma (FT readout reinit magnitude): `--gamma` (also mirrored into `--scaling` for backwards compatibility)
  - lambda_PT: `--lmda` in the pretrain model path (set by c * lmda_frac)
  - c_PT: `--c` in the pretrain model path

We vary:
  - beta structure via `active_dim_2` (teacher sparsity) and `pretrain_overlap` (overlap with PT features).
  - induced k via (c, lmda_frac) through the pretrained model AND via gamma at FT.
"""

cs = [1e-5, 1e-3, 1e-1]
lmda_fracs = [-1.0, 0.0, 1.0]
gammas = [1e-4, 1e-3, 1e-2, 1e-1]

argparse_array = ArgparseArray(
    # reproducibility
    seed=list(range(3)),

    # teacher / beta axis
    active_dim_1=40,
    active_dim_2=[5, 10, 20, 40],
    aux_overlap_frac=[0.0, 0.5, 1.0],
    pretrain_overlap=(lambda overlap_frac, active_dim_2, **kwargs: int(round(overlap_frac * active_dim_2))),
    one_task=[True],
    linear_readout=[False],
    same_signs=[True],

    # sample-size axis
    n_train1=[1024],
    n_train2=[2**i for i in range(4, 9)],  # 16..256
    inp_dim=[1000],

    # PT-induced knobs (select which pretrained model to load)
    c=cs,
    aux_lmda_frac=lmda_fracs,
    lmda=(lambda c, lmda_frac, **kwargs: f"{c * lmda_frac:.10f}"),
    init_method=['complex'],
    model_scaling=[1e-3],
    model_path=(lambda init_method, seed, c, lmda_frac, **kwargs:
                f"data/diagonal/pretrain/"
                f"seed={seed}--active_dim=40--c={c:.1e}--lmda={c * lmda_frac:.10f}--init_method={init_method}/model.pt"),

    # FT knobs (gamma)
    gamma=gammas,
    scaling=(lambda gamma, **kwargs: gamma),  # keep old scripts happy; gamma is the true knob
    w_scaling=[1.0],

    # training
    load_model=[True],
    threshold=[1e-10],
    epochs=[int(1e5)],
    lr=[0.1],

    # output
    save_path=(lambda init_method, seed, n_train2, active_dim_2, pretrain_overlap, gamma, c, lmda_frac, **kwargs:
               f"data/diagonal/k_sweep/"
               f"init_method={init_method}--seed={seed}--n_train2={n_train2}--active_dim_2={active_dim_2}--"
               f"pretrain_overlap={pretrain_overlap}--gamma={gamma:.1e}--"
               f"lmda={c * lmda_frac:.10f}--c={c:.1e}/"),
    save_weights=[False],
    save_feathers=[True],
)


def main(args):
    import sys as _sys
    resolved_args = argparse_array.get_args(args.array_id)
    print('k-sweep finetune experiment parameters:')
    for key in sorted(resolved_args.keys()):
        print(f"  {key}: {resolved_args[key]}")
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune_k_runner.py', args.array_id, python_cmd=_sys.executable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    main(parser.parse_args())


