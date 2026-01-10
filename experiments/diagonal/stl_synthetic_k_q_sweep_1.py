import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray

"""
Sweep for STL synthetic-k q-penalty interpolation:
  - x-axis: delta = n_train / inp_dim (via n_train sweep)
  - curve identity: k_scale (log-spaced, 5 curves)

Produces per-run folders under data/diagonal/stl_synthk_q/.
"""

inp_dim = 1000
n_trains = [2**i for i in range(4, 11)]  # 16..1024
k_scales = [1e-8, 1e-5, 1e-2, 1e1, 1e4]  # 5 curves, log-spaced (sqrt_k scale)

argparse_array = ArgparseArray(
    seed=[0, 1, 2],
    inp_dim=[inp_dim],
    active_dim=[40],
    n_train=n_trains,
    n_test=[10000],

    k_pattern=["logspace"],
    k_spread=[1e3],  # internal heterogeneity; fixed for now
    k_scale=k_scales,

    save_folder=(lambda seed, active_dim, n_train, k_pattern, k_scale, k_spread, **kwargs:
                 f"data/diagonal/stl_synthk_q/"
                 f"seed={seed}--active_dim={active_dim}--n_train={n_train}--"
                 f"k_pattern={k_pattern}--k_scale={k_scale:.0e}--k_spread={k_spread:.0e}/"),
)


def main(args):
    import sys as _sys
    argparse_array.call_script("experiments/diagonal/stl_synthetic_k_q_train.py", args.array_id, python_cmd=_sys.executable)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("array_id", type=int)
    main(p.parse_args())




















