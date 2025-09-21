from copy import deepcopy
import argparse
import math
import sys
import os
from pathlib import Path
sys.path.append('')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
import pandas as pd

import functions.networks as nt

def get_parameters(c, lmda):
    # if np.any(lmda == 0):
    #     raise ValueError("λ must be nonzero.")
    if np.any(c**2 < lmda**2):
        raise ValueError("Require c² ≥ λ² for real outputs.")
    v = np.sqrt((c + lmda) / 2)
    u = np.sqrt((c - lmda) / 2)
    return v, v, u, u  # v⁺, v⁻, u⁺, u⁻

class DiagonalNet(nn.Module):
    def __init__(self, inp_dim, scaling=1., lmda=0., c=0.001, init_method='complex'):
        super().__init__()
        if init_method == 'simple':
            # Simple initialization - all parameters start at the same value
            self.w_pos = nn.Parameter(scaling*torch.ones(inp_dim))
            self.v_pos = nn.Parameter(scaling*torch.ones(inp_dim))
            self.v_neg = nn.Parameter(scaling*torch.ones(inp_dim))
            self.w_neg = nn.Parameter(scaling*torch.ones(inp_dim))
        elif init_method == 'complex':
            # Complex initialization
            w_pos, w_neg, v_pos, v_neg = get_parameters(c, lmda)
            self.w_pos = nn.Parameter(w_pos*torch.ones(inp_dim))
            self.v_pos = nn.Parameter(v_pos*torch.ones(inp_dim))
            self.v_neg = nn.Parameter(v_neg*torch.ones(inp_dim))
            self.w_neg = nn.Parameter(w_neg*torch.ones(inp_dim))
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
    
    def beta(self):
        return self.w_pos*self.v_pos-self.w_neg*self.v_neg

    def forward(self, x):
        return x@self.beta()

def l1_norm(x):
    return torch.sum(torch.abs(x)).item()

def l2_norm(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2)).item()

def train(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    norms = []
    x, y = train_data
    val_x, val_y = val_data
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                val_loss = F.mse_loss(model(val_x), val_y).item()
                new_df = pd.DataFrame({
                    'loss': [val_loss]
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
                norms.append(
                    pd.DataFrame({
                        'norm': ['l1', 'l2'],
                        'value': [l1_norm(model.beta()), l2_norm(model.beta())],
                        'epoch': [i, i]
                    })
                )
                # Print training progress
                print(f"Epoch {i:6d}: Train MSE = {loss:.6f}, Val MSE = {val_loss:.6f}")
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold)
    with torch.no_grad():
        final_val_loss = F.mse_loss(model(val_x), val_y).item()
        new_df = pd.DataFrame({
            'loss': [final_val_loss]
        })
        new_df['epoch'] = i
        test_preds.append(new_df)
        norms.append(
            pd.DataFrame({
                'norm': ['l1', 'l2'],
                'value': [l1_norm(model.beta()), l2_norm(model.beta())],
                'epoch': [i, i]
            })
        )
        # Print final training results
        print(f"\nTraining completed at epoch {i}")
        print(f"Final Train MSE = {loss:.6f}")
        print(f"Final Val MSE = {final_val_loss:.6f}")
    losses = pd.DataFrame({
        'epoch': np.arange(len(losses)),
        'loss': torch.stack(losses).numpy()
    })
    losses['split'] = 'train'
    test_preds = pd.concat(test_preds).reset_index(drop=True)
    test_preds['split'] = 'val'
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), model, pd.concat(norms).reset_index()

def teacher(x, W, V):
    outp = x@W.T
    outp = V*outp
    return outp.sum(dim=-1)

def circular_sample(shape):
    W = Normal(0,1).sample(shape)
    return W/torch.sqrt(torch.mean(W**2, dim=-1, keepdims=True))

def sample_teacher(inp_dim, active_dim):
    W = F.one_hot(torch.randperm(inp_dim)[:active_dim], inp_dim).float()
    V = torch.sign(torch.rand((active_dim,))-0.5).float()/math.sqrt(active_dim)
    return (W, V)

def main(args):
    # Print experiment settings
    print("="*80)
    print("EXPERIMENT SETTINGS")
    print("="*80)
    print(f"Seed: {args.seed}")
    print(f"Input dimension: {args.inp_dim}")
    print(f"Active dimension: {args.active_dim}")
    print(f"Training samples: {args.n_train}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.epochs}")
    print(f"Convergence threshold: {args.threshold}")
    print(f"Scaling: {args.scaling}")
    print(f"Lambda (λ): {args.lmda}")
    print(f"C parameter: {args.c}")
    print(f"Initialization method: {args.init_method}")
    print(f"Learning rate tuning: {not args.no_tuning}")
    print(f"Save folder: {args.save_folder}")
    print("="*80)
    print("Starting training...")
    print("="*80)
    
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    param = sample_teacher(args.inp_dim, args.active_dim)
    x = Normal(0, 1).sample((args.n_train, args.inp_dim))
    x = x/torch.sqrt(torch.mean(x**2, dim=-1, keepdims=True))
    val_x = Normal(0, 1).sample((10000, args.inp_dim))
    val_x = val_x/torch.sqrt(torch.mean(val_x**2, dim=-1, keepdims=True))
    y = teacher(x, *param)
    val_y = teacher(val_x, *param)
    net = DiagonalNet(args.inp_dim, scaling=args.scaling, lmda=args.lmda, c=args.c, init_method=args.init_method)
    df, net, norm_df = train(net, (x, y), (val_x, val_y), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold)
    df.to_feather(os.path.join(args.save_folder, 'df.feather'))
    norm_df.to_feather(os.path.join(args.save_folder, 'norm_df.feather'))
    teacher_df = pd.DataFrame({
        'norm': ['l1', 'l2'],
        'value': [l1_norm(param[1]), l2_norm(param[1])]
    })
    teacher_df.to_feather(os.path.join(args.save_folder, 'teacher_df.feather'))
    torch.save(net.state_dict(), os.path.join(args.save_folder, 'model.pt'))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--n_train', type=int, default=50)
    parser.add_argument('--inp_dim', type=int, default=100)
    parser.add_argument('--active_dim', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--no_tuning', action='store_true')
    parser.add_argument('--lr', type=float, default=1e20)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--scaling', type=float, default=1.)
    parser.add_argument('--lmda', type=float, default=0.)
    parser.add_argument('--c', type=float, default=0.001)
    parser.add_argument('--init_method', type=str, default='complex', choices=['simple', 'complex'])
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
