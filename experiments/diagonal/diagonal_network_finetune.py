# --- TOP OF FILE ---
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # if CUDA present
from copy import deepcopy
import argparse
import sys
import math
import os
from pathlib import Path
sys.path.append('')

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
import pandas as pd
import csv
import json
import os
from datetime import datetime

import functions.networks as nt
import matplotlib.pyplot as plt

import os, random, numpy as np, torch
import fcntl
import time

def make_deterministic(seed: int, use_gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch algos
    torch.use_deterministic_algorithms(True, warn_only=False)

    # Numeric behavior
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Threads
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass



class DiagonalNet(nn.Module):
    def __init__(self, inp_dim, scaling=1., linear_readout=False):
        super().__init__()
        self.w_pos = nn.Parameter(scaling*torch.ones(inp_dim))
        self.v_pos = nn.Parameter(scaling*torch.ones(inp_dim))
        self.v_neg = nn.Parameter(scaling*torch.ones(inp_dim))
        self.w_neg = nn.Parameter(scaling*torch.ones(inp_dim))
        self.linear_readout = linear_readout
    
    def beta(self):
        return self.w_pos*self.v_pos-self.w_neg*self.v_neg

    def parameters(self):
        if self.linear_readout:
            return [self.v_pos, self.v_neg]
        else:
            return [self.w_pos, self.v_pos, self.v_neg, self.w_neg]

    def forward(self, x):
        return x@self.beta()

def l1_norm(x):
    return torch.sum(torch.abs(x)).item()

def l2_norm(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2)).item()

def mt_norm(x):
    piecewise_l2 = torch.sqrt(torch.sum(x**2, dim=1))
    return piecewise_l2.sum().item()

def q(z):
    return 2-torch.sqrt(4+z**2)+z*torch.arcsinh(z/2)

def q_norm(x, gamma):
    return ((torch.abs(x[:,0])+gamma**2)*q(x[:,1]/(torch.abs(x[:,0])+gamma**2))).sum().item()

def plot_training_loss(losses_df, save_path=None, show_plot=True):
    """
    Plot training and validation loss against epochs for the train_one_task function.
    
    Args:
        losses_df: DataFrame containing training and validation losses with columns 'epoch', 'loss', 'split'
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    # Filter for training and validation losses
    train_losses = losses_df[losses_df['split'] == 'train']
    val_losses = losses_df[losses_df['split'] == 'val']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses['epoch'], train_losses['loss'], 'b-', linewidth=2, label='Training Loss')
    
    # Plot validation loss if available
    if not val_losses.empty:
        plt.plot(val_losses['epoch'], val_losses['loss'], 'r-', linewidth=2, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.yscale('log')  # Use log scale for better visualization of loss decay
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return plt.gcf()

class MTDiagonalNet(nn.Module):
    def __init__(self, inp_dim, outp_dim=1, scaling=1., linear_readout=False):
        super().__init__()
        self.w_pos = nn.Parameter(scaling*torch.ones(inp_dim, 1))
        self.v_pos = nn.Parameter(scaling*torch.ones(inp_dim, outp_dim))
        self.v_neg = nn.Parameter(scaling*torch.ones(inp_dim, outp_dim))
        self.w_neg = nn.Parameter(scaling*torch.ones(inp_dim, 1))
        self.linear_readout = linear_readout
    
    def beta(self):
        return self.w_pos*self.v_pos-self.w_neg*self.v_neg

    def parameters(self):
        if self.linear_readout:
            return [self.v_pos, self.v_neg]
        else:
            return [self.w_pos, self.v_pos, self.v_neg, self.w_neg]

    def forward(self, x):
        return x@self.beta()

def train_two_tasks(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5, pretrained_beta=None, optimizer_type='full_batch', adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8):
    or_model = deepcopy(model)
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:  # 'full_batch' — current behaviour, momentum forced to 0
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.)
    losses = []
    test_preds = []
    x, y, task = train_data
    val_x, val_y1, val_y2 = val_data
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(x)
        pred = select_output(pred, task)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                new_df = pd.DataFrame({
                    'loss': [
                        F.mse_loss(model(val_x)[:,0], val_y1).item(),
                        F.mse_loss(model(val_x)[:,1], val_y2).item()
                    ],
                    'split': ['val_1', 'val_2']
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train_two_tasks(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, pretrained_beta=pretrained_beta, optimizer_type=optimizer_type, adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps)
    with torch.no_grad():
        new_df = pd.DataFrame({
            'loss': [
                F.mse_loss(model(val_x)[:,0], val_y1).item(),
                F.mse_loss(model(val_x)[:,1], val_y2).item()
            ],
            'split': ['val_1', 'val_2']
        })
        new_df['epoch'] = i
        test_preds.append(new_df)
    losses = pd.DataFrame({
        'epoch': np.arange(len(losses)),
        'loss': torch.stack(losses).numpy()
    })
    losses['split'] = 'train'
    test_preds = pd.concat(test_preds).reset_index(drop=True)
    with torch.no_grad():
        both_betas = model.beta()
        q_betas = torch.stack([pretrained_beta, both_betas[:,1]], dim=1)
        norm_df = pd.DataFrame({
            'norm': ['l1', 'l2', 'mt', 'q'],
            'value': [l1_norm(both_betas[:,1]), l2_norm(both_betas[:,1]), mt_norm(both_betas), q_norm(q_betas, args.scaling)],
            'kind': 'student'
        })
        df_weights = pd.concat([
            pd.DataFrame({'dim': np.arange(both_betas.shape[0]), 'value': both_betas[:,0], 'task': '1'}),
            pd.DataFrame({'dim': np.arange(both_betas.shape[0]), 'value': both_betas[:,1], 'task': '2'})
        ])
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), norm_df, model, df_weights.reset_index(drop=True)

def train_one_task(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5, beta_1=None, optimizer_type='full_batch', adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8):
    or_model = deepcopy(model)
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:  # 'full_batch' — current behaviour, momentum forced to 0
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.)
    losses = []
    test_preds = []
    x, y = train_data
    val_x, val_y = val_data
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if(i == 60000):
            print('epoch testing')
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                val_loss = F.mse_loss(model(val_x), val_y).item()
                new_df = pd.DataFrame({
                    'loss': [
                        val_loss
                    ],
                    'split': ['val']
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
                print(f"Epoch {i:6d}: Train MSE = {loss:.6f}, Val MSE = {val_loss:.6f}")
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train_one_task(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, beta_1=beta_1, optimizer_type=optimizer_type, adam_beta1=adam_beta1, adam_beta2=adam_beta2, adam_eps=adam_eps)
    with torch.no_grad():
        final_val_loss = F.mse_loss(model(val_x), val_y).item()
        new_df = pd.DataFrame({
            'loss': [
                final_val_loss
            ],
            'split': ['val']
        })
        new_df['epoch'] = i
        test_preds.append(new_df)
    
    print(f"\nTraining completed at epoch {i}")
    print(f"Final Train MSE = {loss:.6f}")
    print(f"Final Val MSE = {final_val_loss:.6f}")
    losses = pd.DataFrame({
        'epoch': np.arange(len(losses)),
        'loss': torch.stack(losses).numpy()
    })
    losses['split'] = 'train'
    test_preds = pd.concat(test_preds).reset_index(drop=True)
    with torch.no_grad():
        beta = model.beta()
        both_betas = torch.stack([beta_1, beta], dim=1)
        norm_df = pd.DataFrame({
            'norm': ['l1', 'l2', 'mt', 'q'],
            'value': [l1_norm(beta), l2_norm(beta), mt_norm(both_betas), q_norm(both_betas, args.scaling)],
            'kind': 'student'
        })
        df_weights = pd.concat([
            pd.DataFrame({'dim': np.arange(both_betas.shape[0]), 'value': both_betas[:,0], 'task': '1'}),
            pd.DataFrame({'dim': np.arange(both_betas.shape[0]), 'value': both_betas[:,1], 'task': '2'})
        ])
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), norm_df, model, df_weights.reset_index(drop=True)

def teacher(x, W, V):
    outp = x@W.T
    outp = V*outp
    return outp.sum(dim=-1)


def circular_sample(shape, generator=None, device=None, dtype=torch.float32):
    W = torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    return W / torch.sqrt((W**2).mean(dim=-1, keepdim=True))

# def circular_sample(shape):
#     W = Normal(0,1).sample(shape)
#     return W/torch.sqrt(torch.mean(W**2, dim=-1, keepdims=True))

def sample_teacher(inp_dim, active_dim):
    W = F.one_hot(torch.randperm(inp_dim)[:active_dim], inp_dim).float()
    V = torch.sign(torch.rand((active_dim,))-0.5).float()/math.sqrt(active_dim)
    return (W, V)

# def sample_two_teachers(inp_dim, active_dim_1, active_dim_2, overlap=0):
#     perm = torch.randperm(inp_dim)
#     W = F.one_hot(perm[:active_dim_1], inp_dim).float()
#     V = torch.sign(torch.rand((active_dim_1,))-0.5).float()
#     W2 = F.one_hot(
#         torch.cat([perm[:overlap], perm[active_dim_1:(active_dim_1+active_dim_2-overlap)]]),
#         inp_dim
#     ).float()
#     return (W, V/math.sqrt(active_dim_1)), (W2, V[:active_dim_2]/math.sqrt(active_dim_2))

def sample_two_teachers(
    inp_dim: int,
    active_dim_1: int,
    active_dim_2: int,
    overlap: int = 0,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
):
    # basic sanity checks (helps catch silent bugs)
    if not (0 <= overlap <= min(active_dim_1, active_dim_2)):
        raise ValueError("overlap must be in [0, min(active_dim_1, active_dim_2)].")
    if active_dim_1 > inp_dim or active_dim_2 > inp_dim:
        raise ValueError("active_dim_* must be <= inp_dim.")

    # permutation of input dimensions (reproducible if generator is fixed)
    perm = torch.randperm(inp_dim, generator=generator, device=device)

    # first teacher
    W = F.one_hot(perm[:active_dim_1], num_classes=inp_dim).to(dtype=dtype)
    # random ±1 signs for the active coordinates
    V = torch.sign(torch.rand((active_dim_1,), generator=generator, device=device, dtype=dtype) - 0.5)

    # second teacher (shares 'overlap' indices with the first)
    idx2 = torch.cat([perm[:overlap],
                      perm[active_dim_1:(active_dim_1 + active_dim_2 - overlap)]])
    W2 = F.one_hot(idx2, num_classes=inp_dim).to(dtype=dtype)

    # scale by sqrt of active dims
    return (W.to(device), V / math.sqrt(active_dim_1)), (W2.to(device), V[:active_dim_2] / math.sqrt(active_dim_2))

def select_output(outp, task):
    task_oh = F.one_hot(task, outp.shape[1])
    return (outp*task_oh).sum(dim=-1)

def sample_finetuning_teacher_with_pretrain_overlap(
    pretrained_beta, 
    active_dim_2, 
    overlap, 
    inp_dim,
    same_signs=True,
    generator=None,
    device=None,
    dtype=torch.float32
):
    """
    Create a finetuning teacher with controlled overlap with pretraining teacher.
    
    Args:
        pretrained_beta: The effective beta from pretrained model (w_pos*v_pos - w_neg*v_neg)
        active_dim_2: Number of active dimensions for finetuning teacher
        overlap: Number of dimensions to overlap with pretraining
        inp_dim: Input dimension
        same_signs: If True, use same signs as pretraining for overlapping dimensions
        generator: Random generator for reproducibility
        device: Device for tensors
        dtype: Data type for tensors
    
    Returns:
        (W, V): Teacher parameters for finetuning
    """
    # Find active dimensions in pretrained model (largest absolute values)
    # We need to find the actual number of active dimensions in pretraining
    # For now, we'll use a threshold-based approach or find top-k dimensions
    abs_beta = torch.abs(pretrained_beta)
    # Find dimensions above threshold or use top-k if threshold gives too few
    above_threshold = (abs_beta > 1e-4).sum().item()
    k = max(above_threshold, active_dim_2)  # Ensure we have enough dimensions
    _, active_indices = torch.topk(abs_beta, k=min(k, len(abs_beta)), largest=True)
    pretrain_active_dims = active_indices.tolist()
    
    # Validate overlap constraint
    if overlap > min(len(pretrain_active_dims), active_dim_2):
        raise ValueError(f"overlap ({overlap}) must be <= min(pretrain_active_dims, active_dim_2)")
    
    # Select overlapping dimensions (random subset of pretraining active dimensions)
    if overlap > 0:
        overlap_indices = torch.randperm(len(pretrain_active_dims), generator=generator, device=device)[:overlap]
        overlap_dims = [pretrain_active_dims[i] for i in overlap_indices]
    else:
        overlap_dims = []
    
    # Select new dimensions (from inactive dimensions)
    remaining_dims = active_dim_2 - overlap
    if remaining_dims > 0:
        inactive_dims = [i for i in range(inp_dim) if i not in pretrain_active_dims]
        if len(inactive_dims) < remaining_dims:
            raise ValueError(f"Not enough inactive dimensions ({len(inactive_dims)}) for remaining_dims ({remaining_dims})")
        
        new_indices = torch.randperm(len(inactive_dims), generator=generator, device=device)[:remaining_dims]
        new_dims = [inactive_dims[i] for i in new_indices]
    else:
        new_dims = []
    
    # Combine all active dimensions for finetuning teacher
    finetune_active_dims = overlap_dims + new_dims
    
    # Create W (one-hot selector for active dimensions)
    W = F.one_hot(torch.tensor(finetune_active_dims, device=device), num_classes=inp_dim).to(dtype=dtype)
    
    # Create V (signs for active dimensions)
    V = torch.zeros(active_dim_2, device=device, dtype=dtype)
    
    # Set signs for overlapping dimensions
    if overlap > 0 and same_signs:
        # Use same signs as pretraining
        for i, dim in enumerate(overlap_dims):
            V[i] = torch.sign(pretrained_beta[dim])
    elif overlap > 0:
        # Use new random signs
        V[:overlap] = torch.sign(torch.rand(overlap, generator=generator, device=device, dtype=dtype) - 0.5)
    
    # Set signs for new dimensions (always random)
    if remaining_dims > 0:
        V[overlap:] = torch.sign(torch.rand(remaining_dims, generator=generator, device=device, dtype=dtype) - 0.5)
    
    # Scale by sqrt of active dimensions
    V = V / math.sqrt(active_dim_2)
    
    return (W, V)

def plot_ground_truth_comparison(pretrain_gt, finetune_gt_task1, finetune_gt_task2, save_path, args):
    """Plot comparison between pretraining and finetuning ground truth betas"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Ground Truth Beta Comparison (Seed={args.seed}, active_dim_1={args.active_dim_1}, active_dim_2={args.active_dim_2}, overlap={args.overlap})', fontsize=14)
    
    # Convert to numpy for plotting
    pretrain_np = pretrain_gt.numpy()
    finetune_task1_np = finetune_gt_task1.numpy()
    finetune_task2_np = finetune_gt_task2.numpy()
    
    # Plot 1: Full parameter vectors - Pretrain vs Finetune Task 1 (active_dim_1)
    axes[0, 0].plot(pretrain_np, 'b-', label='Pretrain GT', alpha=0.7, linewidth=1)
    axes[0, 0].plot(finetune_task1_np, 'r-', label='Finetune Task 1 GT (active_dim_1)', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Pretrain vs Finetune Task 1 (active_dim_1)')
    axes[0, 0].set_xlabel('Parameter Index')
    axes[0, 0].set_ylabel('Parameter Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Full parameter vectors - Pretrain vs Finetune Task 2 (active_dim_2)
    axes[0, 1].plot(pretrain_np, 'b-', label='Pretrain GT', alpha=0.7, linewidth=1)
    axes[0, 1].plot(finetune_task2_np, 'g-', label='Finetune Task 2 GT (active_dim_2)', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Pretrain vs Finetune Task 2 (active_dim_2)')
    axes[0, 1].set_xlabel('Parameter Index')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Finetune Task 1 vs Task 2
    axes[0, 2].plot(finetune_task1_np, 'r-', label='Finetune Task 1 GT (active_dim_1)', alpha=0.7, linewidth=1)
    axes[0, 2].plot(finetune_task2_np, 'g-', label='Finetune Task 2 GT (active_dim_2)', alpha=0.7, linewidth=1)
    axes[0, 2].set_title('Finetune Task 1 vs Task 2')
    axes[0, 2].set_xlabel('Parameter Index')
    axes[0, 2].set_ylabel('Parameter Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot - Pretrain vs Finetune Task 1
    axes[1, 0].scatter(pretrain_np, finetune_task1_np, alpha=0.6, s=10)
    axes[1, 0].plot([pretrain_np.min(), pretrain_np.max()], [pretrain_np.min(), pretrain_np.max()], 'k--', alpha=0.5, label='Perfect Match')
    axes[1, 0].set_title('Pretrain vs Finetune Task 1 (active_dim_1)')
    axes[1, 0].set_xlabel('Pretrain GT')
    axes[1, 0].set_ylabel('Finetune Task 1 GT (active_dim_1)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Scatter plot - Pretrain vs Finetune Task 2
    axes[1, 1].scatter(pretrain_np, finetune_task2_np, alpha=0.6, s=10)
    axes[1, 1].plot([pretrain_np.min(), pretrain_np.max()], [pretrain_np.min(), pretrain_np.max()], 'k--', alpha=0.5, label='Perfect Match')
    axes[1, 1].set_title('Pretrain vs Finetune Task 2 (active_dim_2)')
    axes[1, 1].set_xlabel('Pretrain GT')
    axes[1, 1].set_ylabel('Finetune Task 2 GT (active_dim_2)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Scatter plot - Finetune Task 1 vs Task 2
    axes[1, 2].scatter(finetune_task1_np, finetune_task2_np, alpha=0.6, s=10)
    axes[1, 2].plot([finetune_task1_np.min(), finetune_task1_np.max()], [finetune_task1_np.min(), finetune_task1_np.max()], 'k--', alpha=0.5, label='Perfect Match')
    axes[1, 2].set_title('Finetune Task 1 vs Task 2')
    axes[1, 2].set_xlabel('Finetune Task 1 GT (active_dim_1)')
    axes[1, 2].set_ylabel('Finetune Task 2 GT (active_dim_2)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'ground_truth_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Ground truth comparison plot saved to: {plot_path}")
    
    # Also save as PDF for better quality
    plot_pdf_path = os.path.join(save_path, 'ground_truth_comparison.pdf')
    plt.savefig(plot_pdf_path, bbox_inches='tight')
    print(f"Ground truth comparison plot (PDF) saved to: {plot_pdf_path}")
    
    plt.close()
    
    # Print correlation analysis
    corr_pretrain_finetune = torch.corrcoef(torch.stack([pretrain_gt, finetune_gt_task2]))[0, 1].item()
    
    # Calculate similarity as dot product divided by magnitude of pretraining ground truth
    def similarity_to_pretrain(pretrain_gt, other_gt):
        return torch.dot(pretrain_gt, other_gt) / torch.norm(pretrain_gt)
    
    sim_pretrain_finetune = similarity_to_pretrain(pretrain_gt, finetune_gt_task2).item()
    
    print(f"\nGround Truth Correlations:")
    print(f"  Pretrain vs Finetune: {corr_pretrain_finetune:.6f}")
    
    print(f"\nGround Truth Similarity (dot product / ||pretrain||):")
    print(f"  Pretrain vs Finetune: {sim_pretrain_finetune:.6f}")
    
    # Print overlap analysis
    pretrain_nonzero = set(torch.nonzero(pretrain_gt).flatten().tolist())
    finetune_nonzero = set(torch.nonzero(finetune_gt_task2).flatten().tolist())
    
    overlap_pretrain_finetune = len(pretrain_nonzero.intersection(finetune_nonzero))
    
    print(f"\nGround Truth Overlap Analysis:")
    print(f"  Pretrain vs Finetune: {overlap_pretrain_finetune}/{len(pretrain_nonzero)} ({overlap_pretrain_finetune/len(pretrain_nonzero)*100:.1f}%)")
    print(f"  Finetune is subset of Pretrain: {finetune_nonzero.issubset(pretrain_nonzero)}")

def plot_pretrain_and_finetune_simple(pretrain_gt, finetune_gt, save_path, args):
	"""Plot pretraining and finetuning ground truths on the same axes vs index."""
	try:
		import matplotlib.pyplot as plt
	except Exception as e:
		print(f"matplotlib not available; skipping simple ground truth plot. Reason: {e}")
		return

	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	ax.plot(pretrain_gt.numpy(), label='Pretraining ground truth', linewidth=1.5)
	ax.plot(finetune_gt.numpy(), label='Finetuning ground truth', linewidth=1.5)
	ax.set_xlabel('Index')
	ax.set_ylabel('Value')
	title = f"Pretrain vs Finetune Ground Truth (seed={getattr(args, 'seed', 'NA')}, adim1={getattr(args, 'active_dim_1', 'NA')}, adim2={getattr(args, 'active_dim_2', 'NA')}, overlap={getattr(args, 'overlap', 'NA')})"
	ax.set_title(title)
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.tight_layout()

	# Save figure
	out_png = os.path.join(save_path, 'pretrain_vs_finetune_ground_truth.png')
	plt.savefig(out_png, dpi=300, bbox_inches='tight')
	print(f"Simple ground truth plot saved to: {out_png}")
	out_pdf = os.path.join(save_path, 'pretrain_vs_finetune_ground_truth.pdf')
	plt.savefig(out_pdf, bbox_inches='tight')
	plt.close()

def plot_pretrain_vs_learned(pretrain_gt, learned_beta, save_path, args):
	"""Plot pretraining ground truth and learned (post-finetune) beta on same axes."""
	try:
		import matplotlib.pyplot as plt
	except Exception as e:
		print(f"matplotlib not available; skipping pretrain vs learned plot. Reason: {e}")
		return

	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	ax.plot(pretrain_gt.numpy(), label='Pretraining ground truth', linewidth=1.5)
	ax.plot(learned_beta.numpy(), label='Learned beta after finetune', linewidth=1.5)
	ax.set_xlabel('Index')
	ax.set_ylabel('Value')
	title = f"Pretrain GT vs Learned Beta (seed={getattr(args, 'seed', 'NA')}, adim1={getattr(args, 'active_dim_1', 'NA')}, adim2={getattr(args, 'active_dim_2', 'NA')}, overlap={getattr(args, 'overlap', 'NA')})"
	ax.set_title(title)
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.tight_layout()

	# Save figure
	out_png = os.path.join(save_path, 'pretrain_ground_truth_vs_learned_beta.png')
	plt.savefig(out_png, dpi=300, bbox_inches='tight')
	print(f"Pretrain vs learned beta plot saved to: {out_png}")
	out_pdf = os.path.join(save_path, 'pretrain_ground_truth_vs_learned_beta.pdf')
	plt.savefig(out_pdf, bbox_inches='tight')
	plt.close()

def safe_csv_append(csv_path, new_row_data, max_retries=5, base_delay=0.1):
    """
    Thread-safe CSV append operation using file locking.
    
    Args:
        csv_path: Path to the CSV file
        new_row_data: Dictionary containing the new row data
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
    
    Returns:
        bool: True if successful, False otherwise
    """
    lock_file_path = f"{csv_path}.lock"
    
    for attempt in range(max_retries):
        try:
            # Create lock file and acquire exclusive lock
            with open(lock_file_path, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Read existing data
                if os.path.exists(csv_path):
                    try:
                        existing_df = pd.read_csv(csv_path)
                    except Exception:
                        existing_df = pd.DataFrame()
                else:
                    existing_df = pd.DataFrame()
                
                # Create new row dataframe
                new_df = pd.DataFrame([new_row_data])
                
                # Merge columns and combine data
                all_columns = sorted(set(list(existing_df.columns) + list(new_df.columns)))
                existing_df = existing_df.reindex(columns=all_columns)
                new_df = new_df.reindex(columns=all_columns)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Write back to file
                combined_df.to_csv(csv_path, index=False)
                
                # Lock is automatically released when file is closed
                return True
                
        except (BlockingIOError, OSError) as e:
            # Another process has the lock, wait and retry
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue
            else:
                print(f"Failed to acquire lock after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            print(f"Error in safe_csv_append: {e}")
            return False
    
    return False

def log_experiment_results(args, final_val_loss, final_train_loss, final_epoch):
    """Log experiment results to a CSV file including all args parameters automatically.

    This function:
      - Converts all items in args (simple types or JSON-serializable) into columns
      - Adds timestamp and final metrics
      - If the CSV exists, merges columns with any new parameters and rewrites the file
    """

    # Define the CSV file path
    csv_file = 'experiment_results.csv'
    results_dir = 'experiment_results'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, csv_file)

    # Helper to make values serializable
    def _serialize_value(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        # Try JSON for lists/tuples/dicts and fall back to str
        try:
            return json.dumps(value)
        except Exception:
            return str(value)

    # Collect all args as a dict
    args_dict = {k: _serialize_value(v) for k, v in vars(args).items()}

    # Build the new row with timestamp and metrics
    row_data = {
        **args_dict,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_epoch': final_epoch,
    }

    # Use thread-safe CSV append
    success = safe_csv_append(csv_path, row_data)
    if not success:
        print(f"Warning: Failed to append results to {csv_path}")
        return

    print(f"\nExperiment results logged to: {csv_path}")
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Val Loss: {final_val_loss:.6f}")
    print(f"Final Epoch: {final_epoch}")

def plot_training_and_validation_loss(df, save_path, args):
    """Plot both training and validation loss against epochs"""
    
    # Filter for training and validation data
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'val']
    
    if len(train_data) == 0:
        print("No training data found for plotting")
        return
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot training loss
    ax.plot(train_data['epoch'], train_data['loss'], 'b-', linewidth=2, label='Training Loss')
    
    # Plot validation loss if available
    if len(val_data) > 0:
        ax.plot(val_data['epoch'], val_data['loss'], 'r-', linewidth=2, label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(f'Training and Validation Loss vs Epochs (Seed={args.seed}, active_dim_2={args.active_dim_2}, overlap={args.overlap})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use log scale for y-axis if losses span multiple orders of magnitude
    all_losses = train_data['loss']
    if len(val_data) > 0:
        all_losses = pd.concat([train_data['loss'], val_data['loss']])
    
    if all_losses.max() / all_losses.min() > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (MSE) - Log Scale')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'training_validation_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training and validation loss plot saved to: {plot_path}")
    
    # Also save as PDF for better quality
    plot_pdf_path = os.path.join(save_path, 'training_validation_loss.pdf')
    plt.savefig(plot_pdf_path, bbox_inches='tight')
    print(f"Training and validation loss plot (PDF) saved to: {plot_pdf_path}")
    
    plt.close()
    
    # Print some statistics
    print(f"\nTraining Loss Statistics:")
    print(f"  Initial Loss: {train_data['loss'].iloc[0]:.6f}")
    print(f"  Final Loss: {train_data['loss'].iloc[-1]:.6f}")
    print(f"  Total Epochs: {len(train_data)}")
    print(f"  Loss Reduction: {train_data['loss'].iloc[0] / train_data['loss'].iloc[-1]:.2f}x")
    
    if len(val_data) > 0:
        print(f"\nValidation Loss Statistics:")
        print(f"  Initial Loss: {val_data['loss'].iloc[0]:.6f}")
        print(f"  Final Loss: {val_data['loss'].iloc[-1]:.6f}")
        print(f"  Total Validation Points: {len(val_data)}")
        print(f"  Loss Reduction: {val_data['loss'].iloc[0] / val_data['loss'].iloc[-1]:.2f}x")

def plot_learned_vs_ground_truth(learned_beta, finetune_ground_truth, save_path, args):
    """Plot comparison between learned and ground truth betas for finetuning task"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Learned vs Ground Truth (Seed={args.seed}, active_dim_1={args.active_dim_1}, active_dim_2={args.active_dim_2}, overlap={args.overlap})', fontsize=14)
    
    # Convert to numpy for plotting
    learned_np = learned_beta.numpy()
    finetune_gt_np = finetune_ground_truth.numpy()
    
    # Plot 1: Full parameter vectors - Learned vs Finetuning Ground Truth
    axes[0, 0].plot(learned_np, 'b-', label='Learned', alpha=0.7, linewidth=1)
    axes[0, 0].plot(finetune_gt_np, 'r-', label='Finetuning Ground Truth', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Learned vs Finetuning Ground Truth')
    axes[0, 0].set_xlabel('Parameter Index')
    axes[0, 0].set_ylabel('Parameter Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot - Learned vs Finetuning Ground Truth
    axes[0, 1].scatter(finetune_gt_np, learned_np, alpha=0.6, s=10)
    axes[0, 1].plot([finetune_gt_np.min(), finetune_gt_np.max()], [finetune_gt_np.min(), finetune_gt_np.max()], 'k--', alpha=0.5, label='Perfect Match')
    axes[0, 1].set_title('Learned vs Finetuning Ground Truth')
    axes[0, 1].set_xlabel('Finetuning Ground Truth')
    axes[0, 1].set_ylabel('Learned')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Top elements comparison
    learned_indices = np.argsort(np.abs(learned_np))[-20:][::-1]
    finetune_indices = np.argsort(np.abs(finetune_gt_np))[-20:][::-1]
    
    x_pos = np.arange(20)
    axes[1, 0].bar(x_pos - 0.2, learned_np[learned_indices], 0.4, label='Learned', alpha=0.7)
    axes[1, 0].bar(x_pos + 0.2, finetune_gt_np[finetune_indices], 0.4, label='Finetuning GT', alpha=0.7)
    axes[1, 0].set_title('Top 20 Elements Comparison')
    axes[1, 0].set_xlabel('Rank')
    axes[1, 0].set_ylabel('Parameter Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Absolute values comparison
    axes[1, 1].semilogy(np.sort(np.abs(learned_np))[::-1], 'b-', label='Learned', alpha=0.7)
    axes[1, 1].semilogy(np.sort(np.abs(finetune_gt_np))[::-1], 'r-', label='Finetuning GT', alpha=0.7)
    axes[1, 1].set_title('Sorted Absolute Values (Log Scale)')
    axes[1, 1].set_xlabel('Rank')
    axes[1, 1].set_ylabel('Absolute Parameter Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'learned_vs_ground_truth.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Learned vs ground truth plot saved to: {plot_path}")
    
    # Also save as PDF for better quality
    plot_pdf_path = os.path.join(save_path, 'learned_vs_ground_truth.pdf')
    plt.savefig(plot_pdf_path, bbox_inches='tight')
    print(f"Learned vs ground truth plot (PDF) saved to: {plot_pdf_path}")
    
    plt.close()
    
    # Print correlation analysis
    corr_learned_finetune = torch.corrcoef(torch.stack([learned_beta, finetune_ground_truth]))[0, 1].item()
    
    # Calculate similarity as dot product divided by magnitude of finetuning ground truth
    def similarity_to_finetune(finetune_gt, learned):
        return torch.dot(finetune_gt, learned) / torch.norm(finetune_gt)
    
    sim_learned_finetune = similarity_to_finetune(finetune_ground_truth, learned_beta).item()
    
    print(f"\nLearned vs Finetuning Ground Truth Correlations:")
    print(f"  Learned vs Finetuning: {corr_learned_finetune:.6f}")
    
    print(f"\nLearned vs Finetuning Ground Truth Similarity (dot product / ||finetune||):")
    print(f"  Learned vs Finetuning: {sim_learned_finetune:.6f}")
    
    # Print overlap analysis
    learned_nonzero = set(torch.nonzero(learned_beta).flatten().tolist())
    finetune_nonzero = set(torch.nonzero(finetune_ground_truth).flatten().tolist())
    
    overlap_learned_finetune = len(learned_nonzero.intersection(finetune_nonzero))
    
    print(f"\nLearned vs Finetuning Ground Truth Overlap Analysis:")
    print(f"  Learned vs Finetuning: {overlap_learned_finetune}/{len(finetune_nonzero)} ({overlap_learned_finetune/len(finetune_nonzero)*100:.1f}%)")
    
    # Print statistical analysis
    print(f"\nLearned vs Finetuning Ground Truth Statistics:")
    print(f"  Learned - Non-zero elements: {len(learned_nonzero)}")
    print(f"  Learned - L2 norm: {torch.norm(learned_beta).item():.6f}")
    print(f"  Learned - L1 norm: {torch.norm(learned_beta, p=1).item():.6f}")
    print(f"  Finetuning GT - Non-zero elements: {len(finetune_nonzero)}")
    print(f"  Finetuning GT - L2 norm: {torch.norm(finetune_ground_truth).item():.6f}")
    print(f"  Finetuning GT - L1 norm: {torch.norm(finetune_ground_truth, p=1).item():.6f}")

def main(args):
    make_deterministic(args.seed, use_gpu=False)
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)


    gen1 = torch.Generator(device='cpu').manual_seed(args.seed + 0)
    gen2 = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    gen3 = torch.Generator(device='cpu').manual_seed(args.seed + 2)
    gen4 = torch.Generator(device='cpu').manual_seed(args.seed + 3)

    # Load pretrained model to get effective beta (and PT state, before FT reinit)
    net = DiagonalNet(args.inp_dim, scaling=args.model_scaling, linear_readout=args.linear_readout)
    net.load_state_dict(torch.load(args.model_path))
    # Effective beta from pretraining, captured before any finetune re-scaling
    betapt_groundtruth = net.beta().detach().clone()
    pretrained_beta = betapt_groundtruth

    # (k/r logging is handled by separate postprocessing scripts; keep finetune code unchanged)

    # Handle sign logic
    use_same_signs = getattr(args, 'same_signs', True)

    # Create finetuning teachers
    if hasattr(args, 'pretrain_overlap') and args.pretrain_overlap is not None and args.one_task:
        # Single-task mode with pretraining overlap
        if args.pretrain_overlap > args.active_dim_2:
            raise ValueError(f"pretrain_overlap ({args.pretrain_overlap}) must be <= active_dim_2 ({args.active_dim_2})")
        
        param2 = sample_finetuning_teacher_with_pretrain_overlap(
            pretrained_beta=betapt_groundtruth,
            active_dim_2=args.active_dim_2,
            overlap=args.pretrain_overlap,
            inp_dim=args.inp_dim,
            same_signs=use_same_signs,
            generator=gen4
        )
        # For single-task mode, we only need param2, but we still need param1 for compatibility
        param1 = param2  # This won't be used in single-task mode
    else:
        # Original multi-task logic or single-task without pretrain overlap
        param1, param2 = sample_two_teachers(args.inp_dim, args.active_dim_1, args.active_dim_2, overlap=args.overlap, generator=gen4)
    
    x1 = circular_sample((args.n_train1, args.inp_dim), generator=gen1)
    x2 = circular_sample((args.n_train2, args.inp_dim), generator=gen2)
    val_x = circular_sample((10000, args.inp_dim), generator=gen3)
    y1 = teacher(x1, *param1)
    y2 = teacher(x2, *param2)
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    task = torch.tensor([0]*args.n_train1+[1]*args.n_train2)
    val_y1 = teacher(val_x, *param1)
    val_y2 = teacher(val_x, *param2)
    if not args.load_model:
        if args.one_task:
            net = DiagonalNet(args.inp_dim, scaling=args.scaling, linear_readout=args.linear_readout)
        else:
            net = MTDiagonalNet(args.inp_dim, outp_dim=2, scaling=args.scaling, linear_readout=args.linear_readout)
    if args.one_task:
        nn.init.constant_(net.v_pos, args.scaling)
        nn.init.constant_(net.v_neg, args.scaling)
        net.w_pos = nn.Parameter(args.w_scaling*net.w_pos)
        net.w_neg = nn.Parameter(args.w_scaling*net.w_neg)

    # Choose training routine based on one_task flag (independent of load_model)
    if args.one_task:
        df, norm_df, model, df_weights = train_one_task(net, (x2, y2), (val_x, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold, beta_1=pretrained_beta, optimizer_type=args.optimizer, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2, adam_eps=args.adam_eps)
        
        # Plot training loss
        plot_save_path = os.path.join(args.save_path, 'training_loss_plot.png')
        plot_training_loss(df, save_path=plot_save_path, show_plot=False)
    else:
        df, norm_df, model, df_weights = train_two_tasks(net, (x, y, task), (val_x, val_y1, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold, pretrained_beta=pretrained_beta, optimizer_type=args.optimizer, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2, adam_eps=args.adam_eps)
    true_beta = torch.stack([
        param1[0].T@(param1[1]),
        param2[0].T@(param2[1])
    ], dim=1)
    norm_df = pd.concat([
        norm_df,
        pd.DataFrame({
            'norm': ['l1', 'l2', 'mt', 'q'],
            'value': [l1_norm(true_beta), l2_norm(true_beta), mt_norm(true_beta), q_norm(torch.stack([pretrained_beta, true_beta[:,1]], dim=1), args.scaling)],
            'kind': 'teacher'
        })
    ]).reset_index(drop=True)
    # Save feather files only if requested
    if args.save_feathers:
        df.to_feather(os.path.join(args.save_path, 'df.feather'))
        norm_df.to_feather(os.path.join(args.save_path, 'norm_df.feather'))
        if args.save_weights:
            print('Saving weights')
            print(os.path.join(args.save_path, 'weights_df.feather'))
            df_weights.to_feather(os.path.join(args.save_path, 'weights_df.feather'))
    else:
        print('Skipping feather file saving (--save_feathers=False)')
    
    # Save consolidated experiment results (final train/val MSE) into a single CSV
    # This writes/append to a single file reused across runs: experiment_results.csv at repo root
    try:
        if args.one_task:
            # Single-task fine-tuning uses (x2, y2); validate against (val_x, val_y2)
            final_train_mse = F.mse_loss(model(x2), y2).item()
            final_val_mse = F.mse_loss(model(val_x), val_y2).item()
        else:
            # Multi-task setting: compute average of task-wise validation MSEs as a single scalar
            final_train_mse = F.mse_loss(select_output(model(x), task), y).item()
            val_mse_task1 = F.mse_loss(model(val_x)[:,0], val_y1).item()
            val_mse_task2 = F.mse_loss(model(val_x)[:,1], val_y2).item()
            final_val_mse = 0.5*(val_mse_task1 + val_mse_task2)

        # Print final losses to stdout for quick visibility
        print(f"Final train MSE: {final_train_mse:.6g}")
        print(f"Final val MSE:   {final_val_mse:.6g}")

        # Parse experiment parameters from save_path directory name
        # This approach automatically extracts all parameters from the directory name
        # (which follows the pattern key1=value1--key2=value2--key3=value3)
        # and creates CSV columns for them dynamically. If new parameters are added
        # to the directory naming in the future, they'll automatically get their own columns.
        import re
        
        def parse_experiment_params_from_path(save_path):
            """Parse experiment parameters from directory name in save_path."""
            # Extract the directory name from the full path
            dirname = os.path.basename(save_path.rstrip('/'))
            
            # Parse parameters using regex pattern: key=value--key=value
            # Handle scientific notation and negative numbers properly
            # Use negative lookahead to stop at '--' but allow '-' in values
            param_pattern = r'(\w+)=((?:(?!--).)+?)(?=--|$)'
            matches = re.findall(param_pattern, dirname)
            
            params = {}
            for key, value in matches:
                # Try to convert to appropriate type
                try:
                    # Try boolean first (exact match)
                    if value.lower() in ['true', 'false']:
                        params[key] = value.lower() == 'true'
                    # Try int (digits only, including negative)
                    elif re.match(r'^-?\d+$', value):
                        params[key] = int(value)
                    # Try float (including scientific notation)
                    elif re.match(r'^-?\d*\.?\d+(?:[eE][+-]?\d+)?$', value):
                        params[key] = float(value)
                    # Keep as string
                    else:
                        params[key] = value
                except ValueError:
                    # Keep as string if conversion fails
                    params[key] = value
            
            return params
        
        # Parse parameters from the save_path
        parsed_params = parse_experiment_params_from_path(args.save_path)
        
        # Debug: print parsed parameters
        print(f"\nParsed parameters from directory name:")
        for key, value in sorted(parsed_params.items()):
            print(f"  {key}: {value}")
        
        # Start with the parsed parameters from directory name
        result_row = parsed_params.copy()
        
        # Add the final metrics
        result_row.update({
            'final_train_mse': final_train_mse,
            'final_val_mse': final_val_mse,
            'save_path': args.save_path,
            'model_path': args.model_path,
        })
        
        # Add any additional args that might not be in the directory name
        # (these will only be added if they're not already present from parsing)
        additional_params = {
            'n_train1': args.n_train1,
            'active_dim_1': args.active_dim_1,
            'active_dim_2': args.active_dim_2,
            'inp_dim': args.inp_dim,
            'lr': args.lr,
            'epochs': args.epochs,
            'threshold': args.threshold,
            'c': args.c,
            'lmda': args.lmda,
            'init_method': args.init_method,
            'scaling': args.scaling,
            'model_scaling': args.model_scaling,
            'w_scaling': args.w_scaling,
            'same_signs': args.same_signs,
        }
        
        # Add pretrain_overlap if it exists (it's computed but not in directory name)
        if hasattr(args, 'pretrain_overlap') and args.pretrain_overlap is not None:
            additional_params['pretrain_overlap'] = args.pretrain_overlap
        
        # Extract the actual c value from model_path if it exists
        # This handles cases where aux_c is used for model path but not passed to script
        if hasattr(args, 'model_path') and args.model_path:
            model_path_c_match = re.search(r'c=([\d\.\-e]+?)(?:--|/)', args.model_path)
            if model_path_c_match:
                model_c_value = float(model_path_c_match.group(1))
                additional_params['c'] = model_c_value
                print(f"Extracted c={model_c_value} from model_path")
        
        for key, value in additional_params.items():
            if key not in result_row:
                result_row[key] = value

        csv_path = os.path.abspath('experiment_results.csv')
        success = safe_csv_append(csv_path, result_row)
        if success:
            print(f"Appended results to {csv_path}")
        else:
            print(f"Failed to append results to {csv_path}")
    except Exception as e:
        print(f"Failed to write results CSV: {e}")
    
    # Generate training and validation loss plot
    print(f"\nGenerating training and validation loss plot...")
    plot_training_and_validation_loss(df, args.save_path, args)
    
    # Generate ground truth comparison plots
    print(f"\nGenerating ground truth comparison plots...")
    
    # Get finetuning ground truth betas
    finetune_gt_task1 = true_beta[:, 0]  # First task
    finetune_gt_task2 = true_beta[:, 1]  # Second task
    
    # Generate comparison plots
    plot_ground_truth_comparison(pretrained_beta, finetune_gt_task1, finetune_gt_task2, args.save_path, args)
    # Also generate the simple overlay plot requested
    plot_pretrain_and_finetune_simple(pretrained_beta, finetune_gt_task2, args.save_path, args)
    
    # Generate learned vs ground truth plots
    print(f"\nGenerating learned vs ground truth plots...")
    
    # Get the learned beta from the trained model
    learned_beta = model.beta().detach()
    
    # For single task finetuning, compare learned beta with finetuning ground truth
    plot_learned_vs_ground_truth(learned_beta, finetune_gt_task2, args.save_path, args)

    # Additionally, compare pretraining ground truth against learned beta
    plot_pretrain_vs_learned(pretrained_beta, learned_beta, args.save_path, args)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_scaling', type=float, default=1.)
    parser.add_argument('--n_train1', type=int, default=50)
    parser.add_argument('--n_train2', type=int, default=50)
    parser.add_argument('--active_dim_1', type=int, default=10)
    parser.add_argument('--active_dim_2', type=int, default=10)
    parser.add_argument('--inp_dim', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--no_tuning', action='store_true')
    parser.add_argument('--lr', type=float, default=1e20)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--scaling', type=float, default=1.)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--linear_readout', action='store_true')
    parser.add_argument('--one_task', action='store_true')
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--w_scaling', type=float, default=1.)
    parser.add_argument('--lmda', type=float, default=0.)
    parser.add_argument('--c', type=float, default=0.001)
    parser.add_argument('--init_method', type=str, default='complex', choices=['simple', 'complex'])
    parser.add_argument('--plot_ground_truth_only', action='store_true')
    parser.add_argument('--pretrain_overlap', type=int, default=None, help='Overlap with pretraining teacher (for single-task mode)')
    parser.add_argument('--active_threshold', type=float, default=1e-6, help='Threshold for determining active dimensions in pretrained model')
    parser.add_argument('--same_signs', action='store_true', default=True, help='Use same signs as pretraining for overlapping dimensions')
    parser.add_argument('--save_feathers', action='store_true', default=True, help='Save feather files (df.feather, norm_df.feather, weights_df.feather)')
    parser.add_argument('--optimizer', type=str, default='full_batch', choices=['full_batch', 'sgd', 'adam'])
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
