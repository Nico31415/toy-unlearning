#!/usr/bin/env python3
"""
Run a larger STL synthetic-k sweep (many seeds, many n_train values) to get smooth curves + error bars.

This script is meant to be run inside the conda env that has torch, e.g.:
  conda run -n mtl_ft python experiments/diagonal/run_stl_synthk_q_big.py --quick

It calls `stl_synthetic_k_q_train.py` as a subprocess to reuse its logging/CSV behavior.
"""

import argparse
import os
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10, help="Number of seeds (0..seeds-1).")
    p.add_argument("--inp_dim", type=int, default=1000)
    p.add_argument("--active_dim", type=int, default=40)
    p.add_argument("--n_test", type=int, default=10000)
    p.add_argument("--n_trains", type=int, nargs="*", default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--k_scales", type=float, nargs="*", default=[1e-8, 1e-5, 1e-2, 1e1, 1e4])
    p.add_argument("--k_spread", type=float, default=1e3)
    p.add_argument("--k_pattern", type=str, default="logspace")
    p.add_argument("--max_outer", type=int, default=4)
    p.add_argument("--lbfgs_max_iter", type=int, default=500)
    p.add_argument("--base_dir", type=str, default="data/diagonal/stl_synthk_q_big")
    p.add_argument("--quick", action="store_true", help="Smaller/faster settings (for a medium check).")
    args = p.parse_args()

    if args.quick:
        args.seeds = min(args.seeds, 5)
        args.n_trains = [128, 256, 512, 1024]
        args.max_outer = 3
        args.lbfgs_max_iter = 250

    os.makedirs(args.base_dir, exist_ok=True)

    for seed in range(args.seeds):
        for n_train in args.n_trains:
            for k_scale in args.k_scales:
                outdir = os.path.join(
                    args.base_dir,
                    f"seed={seed}--n_train={n_train}--k_scale={k_scale:.0e}--k_spread={args.k_spread:.0e}/",
                )
                cmd = [
                    "python",
                    "experiments/diagonal/stl_synthetic_k_q_train.py",
                    "--seed", str(seed),
                    "--inp_dim", str(args.inp_dim),
                    "--active_dim", str(args.active_dim),
                    "--n_train", str(n_train),
                    "--n_test", str(args.n_test),
                    "--k_pattern", str(args.k_pattern),
                    "--k_scale", str(k_scale),
                    "--k_spread", str(args.k_spread),
                    "--max_outer", str(args.max_outer),
                    "--lbfgs_max_iter", str(args.lbfgs_max_iter),
                    "--save_folder", outdir,
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()




















