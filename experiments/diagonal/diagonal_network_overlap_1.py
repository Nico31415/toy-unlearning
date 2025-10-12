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
    seed=list(range(1)),
    active_dim_1=40,
    active_dim_2=40,
    scaling=1e-3,
    model_scaling=1e-3,
    inp_dim=1000,
    model_path=(lambda init_method, seed, lmda, c, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--c={c}--lmda={lmda}--init_method={init_method}/model.pt'),
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
    aux_lmda=lmdas_init,
    aux_c=[c_init],
    init_method=['complex'], 
    save_path=name_instance('seed', 'n_train2', 'pretrain_overlap', 'load_model', 'linear_readout', 'one_task',
                            base_folder='data/diagonal/overlap'),
    save_weights=True
)

def main(args):
    argparse_array.call_script('experiments/diagonal/diagonal_network_finetune.py', args.array_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
