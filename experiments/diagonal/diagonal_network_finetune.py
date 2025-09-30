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
import os
from datetime import datetime

import functions.networks as nt
import matplotlib.pyplot as plt

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

def train_two_tasks(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5, pretrained_beta=None):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
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
            return train_two_tasks(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, pretrained_beta=pretrained_beta)
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

def train_one_task(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5, beta_1=None):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
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
            return train_one_task(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, beta_1=beta_1)
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

def circular_sample(shape):
    W = Normal(0,1).sample(shape)
    return W/torch.sqrt(torch.mean(W**2, dim=-1, keepdims=True))

def sample_teacher(inp_dim, active_dim):
    W = F.one_hot(torch.randperm(inp_dim)[:active_dim], inp_dim).float()
    V = torch.sign(torch.rand((active_dim,))-0.5).float()/math.sqrt(active_dim)
    return (W, V)

def sample_two_teachers(inp_dim, active_dim_1, active_dim_2, overlap=0):
    perm = torch.randperm(inp_dim)
    W = F.one_hot(perm[:active_dim_1], inp_dim).float()
    V = torch.sign(torch.rand((active_dim_1,))-0.5).float()
    W2 = F.one_hot(
        torch.cat([perm[:overlap], perm[active_dim_1:(active_dim_1+active_dim_2-overlap)]]),
        inp_dim
    ).float()
    return (W, V/math.sqrt(active_dim_1)), (W2, V[:active_dim_2]/math.sqrt(active_dim_2))

def select_output(outp, task):
    task_oh = F.one_hot(task, outp.shape[1])
    return (outp*task_oh).sum(dim=-1)

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

def log_experiment_results(args, final_val_loss, final_train_loss, final_epoch):
    """Log experiment results to a CSV file for easy analysis"""
    
    # Define the CSV file path
    csv_file = 'experiment_results.csv'
    
    # Create the results directory if it doesn't exist
    results_dir = 'experiment_results'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, csv_file)
    
    # Define the fieldnames for the CSV
    fieldnames = [
        'timestamp', 'seed', 'active_dim_1', 'active_dim_2', 'model_scaling', 
        'scaling', 'n_train1', 'n_train2', 'lr', 'threshold', 'epochs',
        'overlap', 'linear_readout', 'one_task', 'load_model', 'init_method',
        'lmda', 'c', 'final_train_loss', 'final_val_loss', 'final_epoch',
        'save_path'
    ]
    
    # Prepare the row data
    row_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': getattr(args, 'seed', 'N/A'),
        'active_dim_1': getattr(args, 'active_dim_1', 'N/A'),
        'active_dim_2': getattr(args, 'active_dim_2', 'N/A'),
        'model_scaling': getattr(args, 'model_scaling', 'N/A'),
        'scaling': getattr(args, 'scaling', 'N/A'),
        'n_train1': getattr(args, 'n_train1', 'N/A'),
        'n_train2': getattr(args, 'n_train2', 'N/A'),
        'lr': getattr(args, 'lr', 'N/A'),
        'threshold': getattr(args, 'threshold', 'N/A'),
        'epochs': getattr(args, 'epochs', 'N/A'),
        'overlap': getattr(args, 'overlap', 'N/A'),
        'linear_readout': getattr(args, 'linear_readout', 'N/A'),
        'one_task': getattr(args, 'one_task', 'N/A'),
        'load_model': getattr(args, 'load_model', 'N/A'),
        'init_method': getattr(args, 'init_method', 'N/A'),
        'lmda': getattr(args, 'lmda', 'N/A'),
        'c': getattr(args, 'c', 'N/A'),
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_epoch': final_epoch,
        'save_path': getattr(args, 'save_path', 'N/A')
    }
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_path)
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow(row_data)
    
    print(f"\nExperiment results logged to: {csv_path}")
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Final Val Loss: {final_val_loss:.6f}")
    print(f"Final Epoch: {final_epoch}")

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
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    
    # Create the finetuning tasks - this will create both pretraining and finetuning teachers
    # with proper overlap relationship
    param1, param2 = sample_two_teachers(args.inp_dim, args.active_dim_1, args.active_dim_2, overlap=args.overlap)
    
    # Extract pretraining ground truth from param1 (first teacher)
    W_pretrain, V_pretrain = param1
    pretrain_ground_truth = torch.zeros(args.inp_dim)
    for i in range(args.active_dim_1):
        active_pos = torch.argmax(W_pretrain[i,:]).item()
        pretrain_ground_truth[active_pos] = V_pretrain[i]
    x1 = circular_sample((args.n_train1, args.inp_dim))
    x2 = circular_sample((args.n_train2, args.inp_dim))
    val_x = circular_sample((10000, args.inp_dim))
    y1 = teacher(x1, *param1)
    y2 = teacher(x2, *param2)
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    task = torch.tensor([0]*args.n_train1+[1]*args.n_train2)
    val_y1 = teacher(val_x, *param1)
    val_y2 = teacher(val_x, *param2)
    net = DiagonalNet(args.inp_dim, scaling=1.0, linear_readout=args.linear_readout)  # Use default scaling
    net.load_state_dict(torch.load(args.model_path))
    
    # Apply model_scaling to the loaded pretrained weights
    with torch.no_grad():
        net.w_pos.data *= args.model_scaling
        net.v_pos.data *= args.model_scaling
        net.w_neg.data *= args.model_scaling
        net.v_neg.data *= args.model_scaling
    
    pretrained_beta = net.beta().detach().clone()
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
        df, norm_df, model, df_weights = train_one_task(net, (x2, y2), (val_x, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold, beta_1=pretrained_beta)
    else:
        df, norm_df, model, df_weights = train_two_tasks(net, (x, y, task), (val_x, val_y1, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold, pretrained_beta=pretrained_beta)
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
    df.to_feather(os.path.join(args.save_path, 'df.feather'))
    norm_df.to_feather(os.path.join(args.save_path, 'norm_df.feather'))
    if args.save_weights:
        print('Saving weights')
        print(os.path.join(args.save_path, 'weights_df.feather'))
        df_weights.to_feather(os.path.join(args.save_path, 'weights_df.feather'))
    
    # Extract final losses for logging
    final_train_loss = df[df['split'] == 'train']['loss'].iloc[-1]
    final_val_loss = df[df['split'] == 'val']['loss'].iloc[-1]
    final_epoch = df[df['split'] == 'train']['epoch'].iloc[-1]
    
    # Log experiment results to CSV
    log_experiment_results(args, final_val_loss, final_train_loss, final_epoch)
    
    # Generate ground truth comparison plots
    print(f"\nGenerating ground truth comparison plots...")
    
    # Get finetuning ground truth betas
    finetune_gt_task1 = true_beta[:, 0]  # First task
    finetune_gt_task2 = true_beta[:, 1]  # Second task
    
    # Verify that pretraining ground truth matches Task 1 (they should be identical)
    pretrain_task1_match = torch.allclose(pretrain_ground_truth, finetune_gt_task1, atol=1e-6)
    print(f"Pretraining ground truth matches Task 1: {pretrain_task1_match}")
    if not pretrain_task1_match:
        print("WARNING: Pretraining and Task 1 ground truths don't match!")
        print(f"Max difference: {torch.max(torch.abs(pretrain_ground_truth - finetune_gt_task1)).item():.8f}")
    
    # Generate comparison plots
    plot_ground_truth_comparison(pretrain_ground_truth, finetune_gt_task1, finetune_gt_task2, args.save_path, args)
    # Also generate the simple overlay plot requested
    plot_pretrain_and_finetune_simple(pretrain_ground_truth, finetune_gt_task2, args.save_path, args)
    
    # Generate learned vs ground truth plots
    print(f"\nGenerating learned vs ground truth plots...")
    
    # Get the learned beta from the trained model
    learned_beta = model.beta().detach()
    
    # For single task finetuning, compare learned beta with finetuning ground truth
    plot_learned_vs_ground_truth(learned_beta, finetune_gt_task2, args.save_path, args)

    # Additionally, compare pretraining ground truth against learned beta
    plot_pretrain_vs_learned(pretrain_ground_truth, learned_beta, args.save_path, args)

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
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
