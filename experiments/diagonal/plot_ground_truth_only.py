#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

sys.path.append('')
from functions.array_training import ArgparseArray, name_instance

def sample_teacher(inp_dim, active_dim):
    """Sample a teacher network with given input dimension and active dimension"""
    torch.manual_seed(0)  # Fixed seed for reproducibility
    W = torch.randn(active_dim, inp_dim)
    V = torch.randn(active_dim)
    return W, V

def sample_two_teachers(inp_dim, active_dim_1, active_dim_2, overlap=True):
    """Sample two teacher networks with potential overlap"""
    torch.manual_seed(0)  # Fixed seed for reproducibility
    
    # First teacher (pretraining)
    W1 = torch.randn(active_dim_1, inp_dim)
    V1 = torch.randn(active_dim_1)
    
    # Second teacher (finetuning)
    if overlap:
        # Use first active_dim_2 dimensions from first teacher
        W2 = W1[:active_dim_2, :]
        V2 = V1[:active_dim_2]
    else:
        # Completely different teacher
        W2 = torch.randn(active_dim_2, inp_dim)
        V2 = torch.randn(active_dim_2)
    
    return (W1, V1), (W2, V2)

def plot_ground_truth_comparison(pretrain_gt, finetune_gt, save_path, seed=0, active_dim_1=40, active_dim_2=5, overlap=True):
    """Plot comparison between pretraining and finetuning ground truth betas"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Ground Truth Comparison (Seed={seed}, active_dim_1={active_dim_1}, active_dim_2={active_dim_2}, overlap={overlap})', fontsize=14)
    
    # Convert to numpy for plotting
    pretrain_np = pretrain_gt.numpy()
    finetune_np = finetune_gt.numpy()
    
    # Plot 1: Full parameter vectors - Pretraining
    axes[0, 0].plot(pretrain_np, 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Pretraining Ground Truth')
    axes[0, 0].set_xlabel('Parameter Index')
    axes[0, 0].set_ylabel('Parameter Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Full parameter vectors - Finetuning
    axes[0, 1].plot(finetune_np, 'r-', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Finetuning Ground Truth')
    axes[0, 1].set_xlabel('Parameter Index')
    axes[0, 1].set_ylabel('Parameter Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Overlay comparison
    axes[0, 2].plot(pretrain_np, 'b-', label='Pretraining', alpha=0.7, linewidth=1)
    axes[0, 2].plot(finetune_np, 'r-', label='Finetuning', alpha=0.7, linewidth=1)
    axes[0, 2].set_title('Overlay Comparison')
    axes[0, 2].set_xlabel('Parameter Index')
    axes[0, 2].set_ylabel('Parameter Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot - Pretraining vs Finetuning
    axes[1, 0].scatter(pretrain_np, finetune_np, alpha=0.6, s=10)
    axes[1, 0].plot([pretrain_np.min(), pretrain_np.max()], [pretrain_np.min(), pretrain_np.max()], 'k--', alpha=0.5, label='Perfect Match')
    axes[1, 0].set_title('Pretraining vs Finetuning')
    axes[1, 0].set_xlabel('Pretraining Ground Truth')
    axes[1, 0].set_ylabel('Finetuning Ground Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Top elements comparison
    pretrain_indices = np.argsort(np.abs(pretrain_np))[-20:][::-1]
    finetune_indices = np.argsort(np.abs(finetune_np))[-20:][::-1]
    
    x_pos = np.arange(20)
    axes[1, 1].bar(x_pos - 0.2, pretrain_np[pretrain_indices], 0.4, label='Pretraining', alpha=0.7)
    axes[1, 1].bar(x_pos + 0.2, finetune_np[finetune_indices], 0.4, label='Finetuning', alpha=0.7)
    axes[1, 1].set_title('Top 20 Elements Comparison')
    axes[1, 1].set_xlabel('Rank')
    axes[1, 1].set_ylabel('Parameter Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Absolute values comparison
    axes[1, 2].semilogy(np.sort(np.abs(pretrain_np))[::-1], 'b-', label='Pretraining', alpha=0.7)
    axes[1, 2].semilogy(np.sort(np.abs(finetune_np))[::-1], 'r-', label='Finetuning', alpha=0.7)
    axes[1, 2].set_title('Sorted Absolute Values (Log Scale)')
    axes[1, 2].set_xlabel('Rank')
    axes[1, 2].set_ylabel('Absolute Parameter Value')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'ground_truth_only.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Ground truth plot saved to: {plot_path}")
    
    # Also save as PDF for better quality
    plot_pdf_path = os.path.join(save_path, 'ground_truth_only.pdf')
    plt.savefig(plot_pdf_path, bbox_inches='tight')
    print(f"Ground truth plot (PDF) saved to: {plot_pdf_path}")
    
    plt.close()
    
    # Print analysis
    corr = torch.corrcoef(torch.stack([pretrain_gt, finetune_gt]))[0, 1].item()
    similarity = torch.dot(pretrain_gt, finetune_gt) / torch.norm(pretrain_gt)
    
    pretrain_nonzero = set(torch.nonzero(pretrain_gt).flatten().tolist())
    finetune_nonzero = set(torch.nonzero(finetune_gt).flatten().tolist())
    overlap_count = len(pretrain_nonzero.intersection(finetune_nonzero))
    
    print(f"\nGround Truth Analysis:")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Similarity (dot product / ||pretrain||): {similarity:.6f}")
    print(f"  Pretraining non-zero elements: {len(pretrain_nonzero)}")
    print(f"  Finetuning non-zero elements: {len(finetune_nonzero)}")
    print(f"  Overlap: {overlap_count}/{len(pretrain_nonzero)} ({overlap_count/len(pretrain_nonzero)*100:.1f}%)")
    print(f"  Finetuning is subset of pretraining: {finetune_nonzero.issubset(pretrain_nonzero)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int, help='Array ID to use for configuration')
    args = parser.parse_args()
    
    # Use the same configuration as diagonal_sparse_overlap_1.py
    c_init = 10**-5
    lmdas_init = [0]
    
    argparse_array = ArgparseArray(
        seed=list(range(6)),
        active_dim_1=40,
        active_dim_2=[5],
        scaling=1e-3,
        model_scaling=1e-3,
        inp_dim=1000,
        model_path=(lambda array_id, seed, lmda, c, **kwargs: f'data/diagonal/pretrain/seed={seed}--active_dim=40--c={c}--lmda={lmda}/model.pt'),
        threshold=1e-10,
        epochs=int(1e6),
        load_model=True,
        one_task=[True],
        linear_readout=[False],
        n_train1=1024,
        n_train2=64,
        aux_overlap_bool=['yes', 'no'],
        overlap=(lambda array_id, overlap_bool, active_dim_2, **kwargs: 0 if overlap_bool=='no' else active_dim_2),
        lr=1e-1,
        lmda=(lambda array_id, lmda, **kwargs: float(f"{lmda:.10f}")),
        c=(lambda array_id, c, **kwargs: c),
        aux_lmda=lmdas_init,
        aux_c=[c_init],
        save_path=name_instance('seed', 'n_train2', 'active_dim_2', 'load_model', 'linear_readout', 'one_task', 'overlap_bool', 'lmda', 'c',
                                base_folder='data/diagonal/sparse_overlap2'),
        save_weights=True
    )
    
    # Get the configuration for this array_id
    config = argparse_array.get_args(args.array_id)
    
    # Create save directory
    save_path = config['save_path']
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Set seed
    torch.manual_seed(config['seed'])
    
    # Generate ground truth for pretraining
    W_pretrain, V_pretrain = sample_teacher(config['inp_dim'], config['active_dim_1'])
    pretrain_ground_truth = torch.zeros(config['inp_dim'])
    for i in range(config['active_dim_1']):
        active_pos = torch.argmax(W_pretrain[i,:]).item()
        pretrain_ground_truth[active_pos] = V_pretrain[i]
    
    # Generate ground truth for finetuning
    overlap_bool = config['overlap'] > 0  # Convert overlap value to boolean
    (W1, V1), (W2, V2) = sample_two_teachers(config['inp_dim'], config['active_dim_1'], config['active_dim_2'], overlap=overlap_bool)
    finetune_ground_truth = torch.zeros(config['inp_dim'])
    for i in range(config['active_dim_2']):
        active_pos = torch.argmax(W2[i,:]).item()
        finetune_ground_truth[active_pos] = V2[i]
    
    # Generate plots
    plot_ground_truth_comparison(pretrain_ground_truth, finetune_ground_truth, save_path, 
                                config['seed'], config['active_dim_1'], config['active_dim_2'], overlap_bool)
    
    print(f"\nConfiguration used:")
    print(f"  Array ID: {args.array_id}")
    print(f"  Seed: {config['seed']}")
    print(f"  Active dim 1 (pretrain): {config['active_dim_1']}")
    print(f"  Active dim 2 (finetune): {config['active_dim_2']}")
    print(f"  Overlap: {overlap_bool}")
    print(f"  Save path: {save_path}")

if __name__ == '__main__':
    import os
    main()
