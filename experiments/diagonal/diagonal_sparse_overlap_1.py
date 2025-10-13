import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

# Mirror the pretraining script structure: use cs_init and lmdas_init_fraction
cs_init = [10**-5, 10**-3, 10**-1]
scaling_init = 1e-3
lmdas_init_fraction = [-1, 1, 0.9, 0, -0.5, -0.75, -0.85]


argparse_array = ArgparseArray(
    seed=[i for i in range(1)],  # Fixed seed to match pretraining
    active_dim_1=40,
    active_dim_2 = [5, 10, 20, 30, 40],
    scaling=[scaling_init],
    model_scaling=[scaling_init],
    inp_dim=1000,
    # Match the pretrain path schema for loading the correct pretrained model
    # Compute lmda on the fly from c and lmda_frac to avoid dependency on callable order
    model_path=(lambda init_method, seed, c, lmda_frac, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--c={c:.1e}--lmda={c * lmda_frac:.10f}--init_method={init_method}/model.pt'),
    threshold=1e-10,
    epochs=int(1e5),
    one_task=[True],
    n_train1=[1024],
    n_train2 = [2** i for i in range(6, 9)],
    aux_overlap_bool =['no'],
    w_scaling = [1.0],
    pretrain_overlap=(lambda overlap_bool, active_dim_2, **kwargs: 0 if overlap_bool=='no' else active_dim_2),
    same_signs=[True],
    lr=0.1,

    # c is a normal (non-aux) iterable -> included in final args
    c=cs_init,

    # fraction spreads the grid but is NOT included in final args
    aux_lmda_frac=lmdas_init_fraction,

    # ---- Callables MUST use stripped names ----
    lmda=(lambda c, lmda_frac, **kwargs: f"{c * lmda_frac:.10f}"),
    init_method=['complex'],
    load_model=[True],
    linear_readout=[False],

    # Build save_path with computed lmda to avoid collisions across different lmda settings
    save_path=(lambda init_method, seed, n_train2, active_dim_2, load_model, linear_readout, one_task, overlap_bool, c, lmda_frac, model_scaling, w_scaling, same_signs, **kwargs:
               f"data/diagonal/sparse_overlap/"
               f"init_method={init_method}--seed={seed}--n_train2={n_train2}--active_dim_2={active_dim_2}--"
               f"load_model={load_model}--linear_readout={linear_readout}--one_task={one_task}--overlap_bool={overlap_bool}--"
               f"lmda={c * lmda_frac:.10f}--c={c:.1e}--model_scaling={model_scaling}--w_scaling={w_scaling}--same_signs={same_signs}/"),
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
    default_args = argparse.Namespace(array_id=0)
    # args = default_args
    args = parser.parse_args()
    main(args)




