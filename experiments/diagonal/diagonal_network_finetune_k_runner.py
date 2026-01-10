#!/usr/bin/env python3
"""
Runner that executes the existing diagonal finetuning code, then postprocesses k/r.

This is the "no edits to existing code" pathway: we keep
`experiments/diagonal/diagonal_network_finetune.py` unchanged and implement all
k/r logic in new files.
"""

import os
import sys

sys.path.append("")

from experiments.diagonal import diagonal_network_finetune as fin
from experiments.diagonal.postprocess_diagonal_k import postprocess_run


def main():
    parser = fin.get_parser()
    args = parser.parse_args()

    # Ensure weights are saved so we can reconstruct beta_PT and beta_FT
    if not getattr(args, "save_weights", False):
        print("k-runner requires --save_weights; enabling it.")
        args.save_weights = True
    if not getattr(args, "save_feathers", False):
        # weights_df.feather only saved when save_feathers=True in current finetune script
        print("k-runner requires --save_feathers; enabling it.")
        args.save_feathers = True

    fin.main(args)

    # Postprocess and append to experiment_results_k.csv
    postprocess_run(args.save_path, write_csv=True)


if __name__ == "__main__":
    main()




















