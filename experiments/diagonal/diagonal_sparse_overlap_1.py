import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

c_init = 10**-5
scaling_init = 1e-3
lmdas_init = [0, -0.00001]  # [0, -1e-5] to match pretraining
# lmdas_init = [0, ]


argparse_array = ArgparseArray(
    seed=[i for i in range(6)],  # Fixed seed to match pretraining
    active_dim_1=40,
    active_dim_2=[5, 40],  # Test both small and large finetuning tasks

    scaling=1e-3,
    model_scaling=1e-3
    inp_dim=1000,
    # Map to the correct pretrained model based on init_method and lmda
    model_path=(lambda init_method, seed, lmda, c, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--c={c}--scaling={scaling_init}--lmda={lmda}--init_method={init_method}/model.pt'),
    threshold=1e-6,  # Less aggressive early stopping to see model_scaling effects
    epochs=int(1e6),
    load_model=(lambda load_model, **kwargs: load_model),
    one_task=[True],
    linear_readout=(lambda linear_readout, **kwargs: linear_readout),
    n_train1=1024,
    #n_train2 = [2** i for i in range(4, 10)],
    n_train2 = [2** i for i in range(4, 10)],
    # n_train2=64,  # Fixed training size for this experiment
    aux_overlap_bool =['yes'],
    #aux_overlap_bool=['yes', 'no'],  # Test both overlap and no overlap
    overlap=(lambda overlap_bool, active_dim_2, **kwargs: 0 if overlap_bool=='no' else active_dim_2),
    lr=1e-3,  # Reduced learning rate to see model_scaling effects
    lmda=(lambda lmda_2, **kwargs: f"{lmda:.10f}"),
    c=(lambda c, **kwargs: c),
    aux_lmda_2=lmdas_init,
    aux_c=[c_init],
    # Add init_method parameter
    # init_method=['simple', 'complex'],
    init_method=['complex'],
    # Add load_model and linear_readout values
    aux_load_model=[True],
    aux_linear_readout=[False],
    # Use name_instance for cleaner save path
    save_path=name_instance('init_method', 'seed', 'n_train2', 'active_dim_2', 'load_model', 'linear_readout', 'one_task', 'overlap_bool', 'lmda', 'c', 'model_scaling',
                            base_folder='data/diagonal/sparse_overlap3'),
    save_weights=True
)

def main(args):
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
