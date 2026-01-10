#!/usr/bin/env python3
"""
Runner for *single-task* diagonal pretraining + k/r postprocessing.

Does NOT modify training code:
  - runs `experiments/diagonal/diagonal_network_pretrain.py`
  - postprocesses the run directory to compute k/r and write `experiment_results_st_k.csv`
"""

import sys

sys.path.append("")

from experiments.diagonal import diagonal_network_pretrain as pt
from experiments.diagonal.postprocess_diagonal_st_k import postprocess_run


def main():
    parser = pt.get_parser()
    args = parser.parse_args()

    pt.main(args)
    postprocess_run(args.save_folder, write_csv=True)


if __name__ == "__main__":
    main()




















