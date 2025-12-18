import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

# Mirror the pretraining script structure: use cs_init and lmdas_init_fraction
cs_init = [10**-5, 10**-3, 10**-1]
scaling_init = 1e-3
lmdas_init_fraction = [-1, 1, 0.0]

argparse_array = ArgparseArray(
    seed=list(range(6)),
    active_dim_1=40,
    active_dim_2=40,
    scaling=1e-3,
    model_scaling=1e-3,
    inp_dim=1000,
    # Match the pretrain path schema for loading the correct pretrained model
    # Compute lmda on the fly from c and lmda_frac to avoid dependency on callable order
    model_path=(lambda init_method, seed, c, lmda_frac, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--c={c:.1e}--lmda={c * lmda_frac:.10f}--init_method={init_method}/model.pt'),
    # model_path=(lambda array_id, seed, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--n_train=1024--scaling=0.001/model.pt'),
    threshold=1e-10,
    epochs=int(1e6),
    load_model=[True],
    one_task=[True],
    linear_readout=[False],
    n_train1=1024,
    n_train2=[2**i for i in range(4, 9)],
    # Use pretrain_overlap directly as a number parameter
    pretrain_overlap=[5, 20, 30, 40],
    same_signs=[True],
    lr=.1,
    # c is a normal (non-aux) iterable -> included in final args
    c=cs_init,

    # fraction spreads the grid but is NOT included in final args
    aux_lmda_frac=lmdas_init_fraction,

    # ---- Callables MUST use stripped names ----
    lmda=(lambda c, lmda_frac, **kwargs: f"{c * lmda_frac:.10f}"),
    init_method=['complex'], 
    # Build save_path with computed lmda to avoid collisions across different lmda settings
    save_path=(lambda init_method, seed, n_train2, pretrain_overlap, load_model, linear_readout, one_task, c, lmda_frac, **kwargs:
               f"data/diagonal/overlap/"
               f"init_method={init_method}--seed={seed}--n_train2={n_train2}--pretrain_overlap={pretrain_overlap}--"
               f"load_model={load_model}--linear_readout={linear_readout}--one_task={one_task}--"
               f"lmda={c * lmda_frac:.10f}--c={c:.1e}/"),
    save_weights=True
)

def main(args):
    import sys as _sys
    # Print resolved experiment parameters before running
    resolved_args = argparse_array.get_args(args.array_id)
    print('Finetune experiment parameters:')
    for key in sorted(resolved_args.keys()):
        print(f"  {key}: {resolved_args[key]}")
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune.py', args.array_id, python_cmd=_sys.executable)

def main_default():
    """Run with default settings equivalent to array_id=0, without CLI parsing."""
    default_args = argparse.Namespace(array_id=0)
    return main(default_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
