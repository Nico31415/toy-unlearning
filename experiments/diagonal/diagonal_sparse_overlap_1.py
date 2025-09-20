import argparse
import sys

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

c_init = 10**-5
scaling_init = 1e-3
lmdas_init = [0, -0.00001]  # [0, -1e-5] to match pretraining

argparse_array = ArgparseArray(
    seed=[0],  # Fixed seed to match pretraining
    active_dim_1=40,
    active_dim_2=[5, 40],  # Test both small and large finetuning tasks
    scaling=scaling_init,
    model_scaling=scaling_init,
    inp_dim=1000,
    # Map to the correct pretrained model based on init_method and lmda
    model_path=(lambda init_method, seed, lmda, c, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--c={c}--scaling={scaling_init}--lmda={lmda}--init_method={init_method}/model.pt'),
    threshold=1e-10,
    epochs=int(1e6),
    load_model=True,
    one_task=[True],
    linear_readout=[False],
    n_train1=1024,
    n_train2=64,  # Fixed training size for this experiment
    overlap_bool=['yes', 'no'],  # Test both overlap and no overlap
    overlap=(lambda overlap_bool, active_dim_2, **kwargs: 0 if overlap_bool=='no' else active_dim_2),
    lr=1e-1,
    lmda=(lambda lmda, **kwargs: lmda),
    c=(lambda c, **kwargs: c),
    aux_lmda=lmdas_init,
    aux_c=[c_init],
    # Add init_method parameter
    init_method=['simple', 'complex'],
    # Construct save path to include all relevant parameters
    save_path=(lambda init_method, active_dim_2, overlap_bool, lmda, **kwargs: 
               f'data/diagonal/sparse_overlap2/init_method={init_method}--seed=0--n_train2=64--active_dim_2={active_dim_2}--load_model=True--linear_readout=False--one_task=True--overlap_bool={overlap_bool}--lmda={lmda}--c=1e-05/'),
    save_weights=True
)

def main(args):
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
