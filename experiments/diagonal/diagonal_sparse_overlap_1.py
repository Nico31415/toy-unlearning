import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

c_init = 10**-5
scaling_init = 1e-3
# lmdas_init = [0, -0.00001]  # [0, -1e-5] to match pretraining
lmdas_init = [0, -0.00001, -0.0000085]
# lmdas_init = [0]


argparse_array = ArgparseArray(
    seed=[i for i in range(1)],  # Fixed seed to match pretraining
    active_dim_1=40,
    # active_dim_2 = [5, 40],
    active_dim_2 = [5, 20, 40],
    # active_dim_2 = [5, 40],
    scaling=1e-3,
    model_scaling=1e-3,
    inp_dim=1000,
    # Map to the correct pretrained model based on init_method and lmda (matches pretrain path schema)
    # model_path=(lambda array_id, seed, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--n_train=1024--scaling=0.001/model.pt'),
    model_path=(lambda init_method, seed, lmda, c, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--c={c}--lmda={lmda}--init_method={init_method}/model.pt'),
    threshold=1e-10,
    epochs=int(1e5),
    one_task=[True],
    n_train1=[1024],
    n_train2 = [2** i for i in range(4, 9)],
    # n_train2 = [64],
    aux_overlap_bool =['yes'],
    w_scaling = [1.0, 0.1],
    pretrain_overlap=(lambda overlap_bool, active_dim_2, **kwargs: 0 if overlap_bool=='no' else active_dim_2),
    same_signs=[True],
    lr=0.1,
    lmda=(lambda lmda, **kwargs: f"{lmda:.10f}"),
    c=(lambda c, **kwargs: c),
    # Mark as auxiliary so it's not forwarded to the finetune script
    aux_lmda=lmdas_init,
    aux_c=[c_init],
    init_method=['complex'], 
    load_model=[True],
    linear_readout=[False],
    # Use name_instance for cleaner save path
    save_path=name_instance('init_method', 'seed', 'n_train2', 'active_dim_2', 'load_model', 'linear_readout', 'one_task', 'overlap_bool', 'lmda', 'c', 'model_scaling', 'w_scaling', 'same_signs',
                            base_folder='data/diagonal/sparse_overlap'),
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




