import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray

"""
Minimal base experiment for k-sweep signal.

Design:
  - Fix a single pretrained model (seed=0, c=1e-3, lmda=0) and a fixed teacher structure
    (active_dim_2=20, pretrain_overlap=20).
  - Sweep FT sample size n_train2 and FT reinit gamma.

This is the smallest run that should visibly move the induced scale:
    sqrt(k_i) = |beta_PT,i| + gamma^2
and therefore move r_i = 2|beta_FT,i|/sqrt(k_i).
"""

argparse_array = ArgparseArray(
    seed=[0],
    inp_dim=[1000],
    active_dim_1=[40],
    active_dim_2=[20],
    one_task=[True],
    linear_readout=[False],
    same_signs=[True],
    n_train1=[1024],
    n_train2=[16, 32, 64, 128, 256],

    # pick a single PT model (make sure it's trained first)
    c=[1e-3],
    aux_lmda_frac=[0.0],
    lmda=(lambda c, lmda_frac, **kwargs: f"{c * lmda_frac:.10f}"),
    init_method=['complex'],
    model_scaling=[1e-3],
    model_path=(lambda init_method, seed, c, lmda_frac, **kwargs:
                f"data/diagonal/pretrain/"
                f"seed={seed}--active_dim=40--c={c:.1e}--lmda={c * lmda_frac:.10f}--init_method={init_method}/model.pt"),

    # full overlap with PT features
    pretrain_overlap=[20],

    # sweep gamma
    gamma=[1e-4, 1e-3, 1e-2, 1e-1],
    scaling=(lambda gamma, **kwargs: gamma),
    w_scaling=[1.0],

    load_model=[True],
    threshold=[1e-10],
    epochs=[int(1e5)],
    lr=[0.1],

    save_path=(lambda seed, n_train2, gamma, **kwargs:
               f"data/diagonal/k_base/"
               f"seed={seed}--n_train2={n_train2}--active_dim_2=20--pretrain_overlap=20--gamma={gamma:.1e}/"),
    save_feathers=[True],
    save_weights=[False],
)


def main(args):
    import sys as _sys
    resolved_args = argparse_array.get_args(args.array_id)
    print('k-base finetune experiment parameters:')
    for key in sorted(resolved_args.keys()):
        print(f"  {key}: {resolved_args[key]}")
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune_k_runner.py', args.array_id, python_cmd=_sys.executable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    main(parser.parse_args())


