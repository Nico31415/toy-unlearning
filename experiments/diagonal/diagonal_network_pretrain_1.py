import argparse
import sys
import os
import torch
import numpy as np
import math
import torch.nn.functional as F
import itertools

sys.path.append('')

from functions.array_training import ArgparseArray, name_instance

def get_parameters(c, lmda):
    # if np.any(lmda == 0):
    #     raise ValueError("λ must be nonzero.")
    if np.any(c**2 < lmda**2):
        raise ValueError("Require c² ≥ λ² for real outputs.")
    v = np.sqrt((c + lmda) / 2)
    u = np.sqrt((c - lmda) / 2)
    return v, v, u, u  # v⁺, v⁻, u⁺, u⁻

class DiagonalNet(torch.nn.Module):
    def __init__(self, inp_dim, scaling=1., lmda=0., c=0.001, init_method='complex'):
        super().__init__()
        if init_method == 'simple':
            # Simple initialization - all parameters start at the same value
            self.w_pos = torch.nn.Parameter(scaling*torch.ones(inp_dim))
            self.v_pos = torch.nn.Parameter(scaling*torch.ones(inp_dim))
            self.v_neg = torch.nn.Parameter(scaling*torch.ones(inp_dim))
            self.w_neg = torch.nn.Parameter(scaling*torch.ones(inp_dim))
        elif init_method == 'complex':
            # Complex initialization
            w_pos, w_neg, v_pos, v_neg = get_parameters(c, lmda)
            self.w_pos = torch.nn.Parameter(w_pos*torch.ones(inp_dim))
            self.v_pos = torch.nn.Parameter(v_pos*torch.ones(inp_dim))
            self.v_neg = torch.nn.Parameter(v_neg*torch.ones(inp_dim))
            self.w_neg = torch.nn.Parameter(w_neg*torch.ones(inp_dim))
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
    
    def beta(self):
        return self.w_pos*self.v_pos-self.w_neg*self.v_neg

    def forward(self, x):
        return x@self.beta()

def sample_teacher(inp_dim, active_dim):
    W = F.one_hot(torch.randperm(inp_dim)[:active_dim], inp_dim).float()
    V = torch.sign(torch.rand((active_dim,))-0.5).float()/math.sqrt(active_dim)
    return (W, V)

def plot_beta_comparison(ground_truth, learned, save_path, args_dict):
    """Plot comparison between ground truth and learned beta parameters"""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available; skipping beta comparison plot. Reason: {e}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Beta Parameter Analysis (Seed={args_dict["seed"]}, n_train={args_dict["n_train"]}, active_dim={args_dict["active_dim"]})', fontsize=14)
    
    # Convert to numpy for plotting
    gt_np = ground_truth.numpy()
    learned_np = learned.numpy()
    
    # Plot 1: Full parameter vectors
    axes[0, 0].plot(gt_np, 'b-', label='Ground Truth', alpha=0.7, linewidth=1)
    axes[0, 0].plot(learned_np, 'r-', label='Learned', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Full Parameter Vectors')
    axes[0, 0].set_xlabel('Parameter Index')
    axes[0, 0].set_ylabel('Parameter Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of learned vs ground truth
    axes[0, 1].scatter(gt_np, learned_np, alpha=0.6, s=10)
    axes[0, 1].plot([gt_np.min(), gt_np.max()], [gt_np.min(), gt_np.max()], 'k--', alpha=0.5, label='Perfect Match')
    axes[0, 1].set_title('Learned vs Ground Truth')
    axes[0, 1].set_xlabel('Ground Truth')
    axes[0, 1].set_ylabel('Learned')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Top 40 elements comparison
    gt_indices = np.argsort(np.abs(gt_np))[-40:][::-1]
    learned_indices = np.argsort(np.abs(learned_np))[-40:][::-1]
    
    x_pos = np.arange(40)
    axes[1, 0].bar(x_pos - 0.2, gt_np[gt_indices], 0.4, label='Ground Truth', alpha=0.7)
    axes[1, 0].bar(x_pos + 0.2, learned_np[learned_indices], 0.4, label='Learned', alpha=0.7)
    axes[1, 0].set_title('Top 40 Elements Comparison')
    axes[1, 0].set_xlabel('Rank')
    axes[1, 0].set_ylabel('Parameter Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Absolute values comparison
    axes[1, 1].semilogy(np.sort(np.abs(gt_np))[::-1], 'b-', label='Ground Truth', alpha=0.7)
    axes[1, 1].semilogy(np.sort(np.abs(learned_np))[::-1], 'r-', label='Learned', alpha=0.7)
    axes[1, 1].set_title('Sorted Absolute Values (Log Scale)')
    axes[1, 1].set_xlabel('Rank')
    axes[1, 1].set_ylabel('Absolute Parameter Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'beta_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Beta comparison plot saved to: {plot_path}")
    
    # Also save as PDF for better quality
    plot_pdf_path = os.path.join(save_path, 'beta_comparison.pdf')
    plt.savefig(plot_pdf_path, bbox_inches='tight')
    print(f"Beta comparison plot (PDF) saved to: {plot_pdf_path}")
    
    plt.close()

def plot_loss_curves(df, save_path, args_dict):
    """Plot training and validation loss curves"""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available; skipping loss curves plot. Reason: {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Separate training and validation data
    train_data = df[df['split'] == 'train']
    val_data = df[df['split'] == 'val']
    
    # Plot training loss
    ax.plot(train_data['epoch'], train_data['loss'], 'b-', label='Training Loss', alpha=0.8, linewidth=1.5)
    
    # Plot validation loss
    ax.plot(val_data['epoch'], val_data['loss'], 'r-', label='Validation Loss', alpha=0.8, linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(f'Training and Validation Loss Curves\n(Seed={args_dict["seed"]}, n_train={args_dict["n_train"]}, active_dim={args_dict["active_dim"]}, λ={args_dict["lmda"]:.2e})', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Use log scale for better visualization
    
    # Add final loss values as text
    final_train_loss = train_data['loss'].iloc[-1]
    final_val_loss = val_data['loss'].iloc[-1]
    ax.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.2e}\nFinal Val Loss: {final_val_loss:.2e}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_path, 'loss_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves plot saved to: {plot_path}")
    
    # Also save as PDF for better quality
    plot_pdf_path = os.path.join(save_path, 'loss_curves.pdf')
    plt.savefig(plot_pdf_path, bbox_inches='tight')
    print(f"Loss curves plot (PDF) saved to: {plot_pdf_path}")
    
    plt.close()

c_init = 10**-5
scaling_init = 1e-3
lmdas_init = [0, -0.00001]  # [0, -1e-5] for different lambda values

argparse_array = ArgparseArray(
    seed=[i for i in range(6)],  # Fixed seed for all cases
    inp_dim=[1000],
    active_dim=[40],
    n_train=1024,
    c=[c_init],
    scaling=[scaling_init],
    threshold=1e-10,
    epochs=int(1e6),
    lr=0.01,
    lmda=(lambda lmda_val, **kwargs: f"{lmda_val:.10f}"),  # Function to pass lambda values with proper formatting
    aux_lmda_val=lmdas_init,  # The lambda values to iterate over
    init_method=['complex'],
    # init_method=['simple', 'complex'],  # This creates 4 array IDs: 0=simple+lmda=0, 1=complex+lmda=0, 2=simple+lmda=-c, 3=complex+lmda=-c
    save_folder=(lambda **kwargs: 
                 f"data/diagonal/pretrain/seed={kwargs['seed']}--active_dim={kwargs['active_dim']}--c={kwargs['c']}--lmda={kwargs['lmda_val']}--init_method={kwargs['init_method']}/")
)

def main(args):
    # Validate array_id against available combinations
    try:
        total_array_ids = len(list(itertools.product(*argparse_array.array_args.values())))
    except Exception:
        total_array_ids = None
    if total_array_ids is not None and (args.array_id < 0 or args.array_id >= total_array_ids):
        print(f"Invalid array_id {args.array_id}. Valid range is 0..{total_array_ids-1}.")
        return

    # Get the arguments dictionary for this array_id
    args_dict = argparse_array.get_args(args.array_id)
    # Ensure numeric lambda for downstream math/formatting
    if 'lmda' in args_dict and isinstance(args_dict['lmda'], str):
        try:
            args_dict['lmda'] = float(args_dict['lmda'])
        except Exception:
            # fallback to aux value if available
            if 'aux_lmda_val' in args_dict:
                args_dict['lmda'] = float(args_dict['aux_lmda_val'])
            else:
                raise
    
    # Run training first
    argparse_array.call_script('experiments/diagonal/diagonal_network_pretrain.py', args.array_id)
    
    # After training, load the trained model and get the learned beta
    model_path = os.path.join(args_dict['save_folder'], 'model.pt')
    if os.path.exists(model_path):
        # Load the trained model
        net = DiagonalNet(args_dict['inp_dim'], scaling=args_dict['scaling'], lmda=args_dict['lmda'], c=args_dict['c'], init_method=args_dict['init_method'])
        net.load_state_dict(torch.load(model_path))
        learned_beta = net.beta().detach()
        
        # Get final MSE loss from saved results
        df_path = os.path.join(args_dict['save_folder'], 'df.feather')
        if os.path.exists(df_path):
            try:
                import pandas as pd
                df = pd.read_feather(df_path)
            except Exception as e:
                print(f"pandas not available or failed to read feather; skipping detailed loss analysis. Reason: {e}")
                df = None
            final_train_loss = df[df['split'] == 'train']['loss'].iloc[-1]
            final_val_loss = df[df['split'] == 'val']['loss'].iloc[-1]
            print(f"\n" + "="*60)
            print(f"POST-TRAINING ANALYSIS")
            print(f"="*60)
            print(f"Final Training MSE: {final_train_loss:.6f}")
            print(f"Final Validation MSE: {final_val_loss:.6f}")
            print(f"="*60)
            
            # Plot and save loss curves
            print(f"\nGenerating loss curves plot...")
            if df is not None:
                plot_loss_curves(df, args_dict['save_folder'], args_dict)
        
        # Reconstruct ground truth beta using same random seed
        torch.manual_seed(args_dict['seed'])
        param = sample_teacher(args_dict['inp_dim'], args_dict['active_dim'])
        W, V = param
        ground_truth_beta = torch.zeros(args_dict['inp_dim'])
        for i in range(args_dict['active_dim']):
            active_pos = torch.argmax(W[i,:]).item()
            ground_truth_beta[active_pos] = V[i]
        
        # Compute MSE with ground truth beta
        # First, we need to reconstruct the training and validation data
        from torch.distributions.normal import Normal
        
        # Reconstruct training data
        torch.manual_seed(args_dict['seed'])
        x_train = Normal(0, 1).sample((args_dict['n_train'], args_dict['inp_dim']))
        x_train = x_train/torch.sqrt(torch.mean(x_train**2, dim=-1, keepdims=True))
        y_train = x_train @ ground_truth_beta
        
        # Reconstruct validation data
        torch.manual_seed(args_dict['seed'] + 1)  # Different seed for validation
        x_val = Normal(0, 1).sample((10000, args_dict['inp_dim']))
        x_val = x_val/torch.sqrt(torch.mean(x_val**2, dim=-1, keepdims=True))
        y_val = x_val @ ground_truth_beta
        
        # Compute MSE with ground truth
        gt_train_mse = F.mse_loss(x_train @ ground_truth_beta, y_train).item()
        gt_val_mse = F.mse_loss(x_val @ ground_truth_beta, y_val).item()
        
        # Print ground truth MSE comparison
        print(f"\nGround Truth Beta Performance:")
        print(f"  Ground Truth Training MSE: {gt_train_mse:.6f}")
        print(f"  Ground Truth Validation MSE: {gt_val_mse:.6f}")
        print(f"\nPerformance Gap:")
        print(f"  Training MSE Gap: {final_train_loss - gt_train_mse:.6f}")
        print(f"  Validation MSE Gap: {final_val_loss - gt_val_mse:.6f}")
        
        # Convert to numpy for easier manipulation
        learned_np = learned_beta.numpy()
        gt_np = ground_truth_beta.numpy()
        
        # Get indices of largest elements
        learned_indices = np.argsort(np.abs(learned_np))[-40:][::-1]
        gt_indices = np.argsort(np.abs(gt_np))[-40:][::-1]
        
        # Print largest elements
        print("\nLargest 40 elements of Ground Truth:")
        for i, idx in enumerate(gt_indices):
            print(f"  {i+1:2d}. Index {idx:3d}: {gt_np[idx]:.6f}")
        
        print("\nLargest 40 elements of Learned:")
        for i, idx in enumerate(learned_indices):
            print(f"  {i+1:2d}. Index {idx:3d}: {learned_np[idx]:.6f}")
        
        # Statistical Analysis
        print(f"\nGround Truth Stats:")
        print(f"  Non-zero elements: {torch.count_nonzero(ground_truth_beta).item()}")
        print(f"  L2 norm: {torch.norm(ground_truth_beta).item():.6f}")
        print(f"  L1 norm: {torch.norm(ground_truth_beta, p=1).item():.6f}")
        
        print(f"\nLearned Stats:")
        print(f"  Non-zero elements: {torch.count_nonzero(learned_beta).item()}")
        print(f"  L2 norm: {torch.norm(learned_beta).item():.6f}")
        print(f"  L1 norm: {torch.norm(learned_beta, p=1).item():.6f}")
        
        # Compute correlation
        correlation = torch.corrcoef(torch.stack([ground_truth_beta, learned_beta]))[0, 1].item()
        print(f"\nCorrelation: {correlation:.6f}")
        
        # Print overlap analysis
        gt_nonzero = set(torch.nonzero(ground_truth_beta).flatten().tolist())
        learned_nonzero = set(torch.nonzero(learned_beta).flatten().tolist())
        overlap = len(gt_nonzero.intersection(learned_nonzero))
        print(f"\nOverlap Analysis:")
        print(f"  Ground truth non-zero positions: {len(gt_nonzero)}")
        print(f"  Learned non-zero positions: {len(learned_nonzero)}")
        print(f"  Overlap: {overlap}/{len(gt_nonzero)} ({overlap/len(gt_nonzero)*100:.1f}%)")
        
        # Generate and save comparison plots
        print(f"\nGenerating comparison plots...")
        plot_beta_comparison(ground_truth_beta, learned_beta, args_dict['save_folder'], args_dict)
        
    else:
        print(f"Model file not found at {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('array_id', type=int)
    args = parser.parse_args()
    main(args)
