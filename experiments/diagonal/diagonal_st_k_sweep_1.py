import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray

"""
Single-task (STL) diagonal experiment sweep with k/r postprocessing.

This runs ONLY `diagonal_network_pretrain.py` (no finetuning), via:
  `experiments/diagonal/diagonal_network_pretrain_k_runner.py`

Axes:
  - beta structure: teacher sparsity via active_dim
  - k (uniform STL scale): via c and lmda (complex init)
  - sample size: n_train
"""

cs = [1e-5, 1e-3, 1e-1]
lmda_fracs = [-1.0, 0.0, 1.0]  # lmda = c * lmda_frac

argparse_array = ArgparseArray(
    seed=list(range(3)),
    inp_dim=[1000],

    # teacher sparsity (beta* axis)
    active_dim=[5, 10, 20, 40],

    # sample size axis
    n_train=[2**i for i in range(4, 11)],  # 16..1024

    # init-induced scale axis (k uniform)
    c=cs,
    aux_lmda_frac=lmda_fracs,
    lmda=(lambda c, lmda_frac, **kwargs: f"{c * lmda_frac:.10f}"),
    init_method=['complex'],
    scaling=[1e-3],

    # training
    threshold=[1e-10],
    epochs=[int(1e6)],
    lr=[0.5],

    # output folder naming
    save_folder=(lambda seed, active_dim, n_train, c, lmda_frac, init_method, **kwargs:
                 f"data/diagonal/stl_k_sweep/"
                 f"seed={seed}--active_dim={active_dim}--n_train={n_train}--"
                 f"c={c:.1e}--lmda={c * lmda_frac:.10f}--init_method={init_method}/"),
)


def main(args):
    import sys as _sys
    resolved_args = argparse_array.get_args(args.array_id)
    print('STL k-sweep pretrain parameters:')
    for key in sorted(resolved_args.keys()):
        print(f"  {key}: {resolved_args[key]}")
    argparse_array.call_script('experiments/diagonal/diagonal_network_pretrain_k_runner.py', args.array_id, python_cmd=_sys.executable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    main(parser.parse_args())




















