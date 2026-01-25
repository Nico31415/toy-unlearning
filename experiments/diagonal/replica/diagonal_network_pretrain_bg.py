# --- TOP OF FILE ---
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # if CUDA present
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
import json

import functions.networks as nt
import random


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

def get_parameters(c, lmda):
    """
    Compute initialization parameters for complex diagonal-net init.
    
    The network has parameters (w_pos, v_pos, v_neg, w_neg) with:
        lambda = w_pos^2 - v_pos^2
        c = w_pos * w_neg + v_pos * v_neg
    
    For target (c, lmda), we compute:
        v = sqrt((c + lmda) / 2)  -> w_pos, w_neg
        u = sqrt((c - lmda) / 2)  -> v_pos, v_neg
    
    Args:
        c: scalar or array, must satisfy c² >= lmda²
        lmda: scalar lambda parameter
    
    Returns:
        (v_pos, v_neg, u_pos, u_neg) for assignment to (w_pos, v_pos, v_neg, w_neg)
        This gives: w_pos=v, v_pos=u, v_neg=u, w_neg=v
        
    Raises:
        ValueError: if c² < lmda² (would give complex outputs)
    """
    if np.any(c**2 < lmda**2):
        raise ValueError("Require c² ≥ λ² for real outputs.")
    v = np.sqrt((c + lmda) / 2)
    u = np.sqrt((c - lmda) / 2)
    # For scalar c, v and u are scalars (no aliasing issue)
    # For array c, return copies to avoid aliasing with torch.from_numpy()
    if np.isscalar(v):
        return v, u, u, v
    return v.copy(), u.copy(), u.copy(), v.copy()


def get_parameters_vectorized(c_vec, lmda):
    """
    Vectorized version of get_parameters for per-coordinate c values.
    
    The network has parameters (w_pos, v_pos, v_neg, w_neg) with:
        lambda_i = w_pos_i^2 - v_pos_i^2
        c_i = w_pos_i * w_neg_i + v_pos_i * v_neg_i
    
    Args:
        c_vec: numpy array of shape (inp_dim,) with per-coordinate c values
        lmda: scalar lambda value
    
    Returns:
        (v_pos, v_neg, u_pos, u_neg) for assignment to (w_pos, v_pos, v_neg, w_neg)
        This gives: w_pos=v, v_pos=u, v_neg=u, w_neg=v
    """
    c_vec = np.asarray(c_vec, dtype=np.float64)
    if np.any(c_vec**2 < lmda**2):
        raise ValueError("Require c² ≥ λ² for all coordinates.")
    v = np.sqrt((c_vec + lmda) / 2)
    u = np.sqrt((c_vec - lmda) / 2)
    # Return COPIES to avoid aliasing when used with torch.from_numpy()
    # (which shares memory with the underlying array)
    return v.copy(), u.copy(), u.copy(), v.copy()


def check_init_invariants(net, c_expected, lmda_expected, atol=1e-12):
    """
    Verify that the network initialization correctly encodes c and lambda.
    
    The diagonal network parameterization satisfies:
        lambda_i = w_pos_i^2 - v_pos_i^2
        c_i = w_pos_i * w_neg_i + v_pos_i * v_neg_i
    
    Args:
        net: DiagonalNet instance
        c_expected: expected c value (scalar or array)
        lmda_expected: expected lambda value (scalar)
        atol: absolute tolerance for comparison
    
    Raises:
        AssertionError: if invariants are not satisfied
    """
    with torch.no_grad():
        lam_i = (net.w_pos**2 - net.v_pos**2)
        c_i = (net.w_pos * net.w_neg + net.v_pos * net.v_neg)
        
        lmda_target = torch.full_like(lam_i, float(lmda_expected))
        if not torch.allclose(lam_i, lmda_target, atol=atol):
            max_err = torch.max(torch.abs(lam_i - lmda_target)).item()
            raise AssertionError(
                f"Lambda invariant violated: max error = {max_err:.6e}, "
                f"expected lambda = {lmda_expected}, atol = {atol}"
            )
        
        if np.isscalar(c_expected):
            c_target = torch.full_like(c_i, float(c_expected))
        else:
            c_target = torch.as_tensor(c_expected, dtype=c_i.dtype, device=c_i.device)
        
        if not torch.allclose(c_i, c_target, atol=atol):
            max_err = torch.max(torch.abs(c_i - c_target)).item()
            raise AssertionError(
                f"C invariant violated: max error = {max_err:.6e}, "
                f"expected c = {c_expected if np.isscalar(c_expected) else 'c_vec'}, atol = {atol}"
            )


class DiagonalNet(nn.Module):
    def __init__(self, inp_dim, scaling=1., lmda=0., c=0.001, c_vec=None, init_method='complex'):
        """
        Diagonal network with optional per-coordinate c initialization.
        
        Args:
            inp_dim: Input dimension
            scaling: Scaling factor for simple init
            lmda: Lambda parameter
            c: Scalar c value (used if c_vec is None)
            c_vec: Optional per-coordinate c values, shape (inp_dim,)
            init_method: 'simple' or 'complex'
        """
        super().__init__()
        if init_method == 'simple':
            # Simple initialization - all parameters start at the same value
            self.w_pos = nn.Parameter(scaling*torch.ones(inp_dim))
            self.v_pos = nn.Parameter(scaling*torch.ones(inp_dim))
            self.v_neg = nn.Parameter(scaling*torch.ones(inp_dim))
            self.w_neg = nn.Parameter(scaling*torch.ones(inp_dim))
            print('lmda for simple initialization:', (self.w_pos**2 - self.v_pos**2)[0].item())
            print('c for simple initialization:', (self.w_pos*self.w_neg + self.v_pos*self.v_neg)[0].item())
        elif init_method == 'complex':
            if c_vec is not None:
                # Per-coordinate initialization using vectorized function
                c_vec = np.asarray(c_vec, dtype=np.float64)
                if c_vec.shape[0] != inp_dim:
                    raise ValueError(f"c_vec has shape {c_vec.shape}, expected ({inp_dim},)")
                v_pos, v_neg, u_pos, u_neg = get_parameters_vectorized(c_vec, lmda)
                self.w_pos = nn.Parameter(torch.from_numpy(v_pos))
                self.v_pos = nn.Parameter(torch.from_numpy(v_neg))
                self.v_neg = nn.Parameter(torch.from_numpy(u_pos))
                self.w_neg = nn.Parameter(torch.from_numpy(u_neg))
                print(f'Per-coordinate c init: c_vec range [{c_vec.min():.6f}, {c_vec.max():.6f}]')
            else:
                # Homogeneous initialization (fixed to match vectorized path)
                v_pos, v_neg, u_pos, u_neg = get_parameters(c, lmda)
                self.w_pos = nn.Parameter(v_pos * torch.ones(inp_dim))
                self.v_pos = nn.Parameter(v_neg * torch.ones(inp_dim))
                self.v_neg = nn.Parameter(u_pos * torch.ones(inp_dim))
                self.w_neg = nn.Parameter(u_neg * torch.ones(inp_dim))
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

def sample_beta_star_bg(inp_dim, rho, generator=None):
    """
    Sample Bernoulli-Gaussian teacher beta_star:
    - beta_i^* = 0 w.p. 1-rho
    - beta_i^* ~ N(0, 1/rho) w.p. rho
    So that E[(beta_i^*)^2] = 1
    """
    if generator is None:
        generator = torch.Generator()
    
    # Sample Bernoulli mask
    mask = torch.rand(inp_dim, generator=generator) < rho
    
    # Sample Gaussian values with variance 1/rho
    gaussian_vals = torch.randn(inp_dim, generator=generator) / math.sqrt(rho)
    
    # Apply mask (convert boolean mask to float64 to match default dtype)
    beta_star = mask.double() * gaussian_vals
    
    return beta_star

def train(model, train_data, test_data, beta_star, test_every_n_epochs=200, log_every_n_epochs=None, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-9, stop_pred_mse=None, stop_beta_rate=0.0, stop_grad_norm=0.0, lr_decay=1.0, lr_decay_interval=2000, save_folder=None,
          min_epochs_before_stop=None, require_eval_before_stop=False, disable_legacy_loss_stop=False,
          fixed_point_beta_rate=1e-6, fixed_point_consecutive_evals=2, fixed_point_grad_norm=0.0):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    all_results = []
    norms = []
    x, y = train_data
    test_x, test_y = test_data
    beta_star = beta_star.to(model.beta().device)
    
    last_evaluated_epoch = -1
    beta_hat_prev = None
    prev_eval_epoch = -1
    
    # === Fixed-point stopping state variables ===
    eval_count = 0
    consecutive_fixed_point_evals = 0
    beta_hat_prev_eval = None  # ONLY updated at eval checkpoints (not per-epoch)
    beta_update_rate = float('inf')  # Track latest rate for guard logic
    
    # === Safe defaults (resolve None to safe values) ===
    # This ensures sweeps are safe even without explicit flags
    if min_epochs_before_stop is None:
        min_epochs_before_stop = test_every_n_epochs
    
    # Use stop_pred_mse if provided, otherwise use threshold
    if stop_pred_mse is None:
        stop_pred_mse = threshold
    
    # Track stop reason (will be set when training stops)
    detected_stop_reason = None
    
    print(f"\nStarting training loop (max {epochs} epochs, threshold={threshold:.2e})...")
    sys.stdout.flush()
    
    # Disable tqdm progress bar in non-interactive mode (SLURM batch jobs)
    # This prevents massive log files from tqdm updates
    is_interactive = sys.stdout.isatty()
    
    # Create tqdm progress bar - disabled if not interactive to avoid log bloat
    pbar = tqdm(
        range(epochs), 
        desc="Training", 
        unit="epoch", 
        miniters=1000 if not is_interactive else 1,  # Less frequent updates in batch mode
        mininterval=10.0 if not is_interactive else 0.1,  # 10s interval in batch mode
        disable=not is_interactive,  # Completely disable in batch mode
        file=sys.stdout
    )
    if is_interactive:
        pbar.set_postfix({'loss': 'N/A', 'pct': '0.0%'})
    sys.stdout.flush()

    # Logging throttle: evaluation can be frequent (for early stopping),
    # but printing can be less frequent (to keep logs manageable).
    if log_every_n_epochs is None:
        log_every_n_epochs = int(test_every_n_epochs)
    else:
        log_every_n_epochs = int(log_every_n_epochs)
    if log_every_n_epochs <= 0:
        log_every_n_epochs = 10**18  # effectively disable periodic logs

    # --- Track (approximate) conserved quantities under gradient flow ---
    # We record drift of:
    #   lambda_pos_i = w_pos_i^2 - v_pos_i^2
    #   lambda_neg_i = w_neg_i^2 - v_neg_i^2
    #   c_i          = w_pos_i*w_neg_i + v_pos_i*v_neg_i
    # These are expected to be conserved under continuous-time dynamics, but may drift under
    # discrete SGD with finite learning rate. We log drift at evaluation checkpoints.
    with torch.no_grad():
        lam_pos0 = (model.w_pos ** 2 - model.v_pos ** 2).detach().clone()
        lam_neg0 = (model.w_neg ** 2 - model.v_neg ** 2).detach().clone()
        c0 = (model.w_pos * model.w_neg + model.v_pos * model.v_neg).detach().clone()

    # Print initial invariant values once (sanity check)
    try:
        print(
            "[invariants@init] "
            f"c_emp: mean={c0.mean().item():.6e}, min={c0.min().item():.6e}, max={c0.max().item():.6e}; "
            f"lmda_pos: mean={lam_pos0.mean().item():.6e}, min={lam_pos0.min().item():.6e}, max={lam_pos0.max().item():.6e}; "
            f"lmda_neg: mean={lam_neg0.mean().item():.6e}, min={lam_neg0.min().item():.6e}, max={lam_neg0.max().item():.6e}",
            flush=True,
        )
    except Exception:
        # Avoid breaking training if printing fails for any reason
        pass

    def _invariant_stats() -> dict:
        with torch.no_grad():
            lam_pos = (model.w_pos ** 2 - model.v_pos ** 2)
            lam_neg = (model.w_neg ** 2 - model.v_neg ** 2)
            c_emp = (model.w_pos * model.w_neg + model.v_pos * model.v_neg)

            lam_pos_dev = (lam_pos - lam_pos0).abs()
            lam_neg_dev = (lam_neg - lam_neg0).abs()
            c_dev = (c_emp - c0).abs()

            return {
                # current means (handy sanity checks)
                "lam_pos_mean": float(lam_pos.mean().item()),
                "lam_neg_mean": float(lam_neg.mean().item()),
                "c_emp_mean": float(c_emp.mean().item()),
                # drift from initialization
                "lam_pos_max_dev": float(lam_pos_dev.max().item()),
                "lam_neg_max_dev": float(lam_neg_dev.max().item()),
                "c_emp_max_dev": float(c_dev.max().item()),
                "lam_pos_p95_dev": float(torch.quantile(lam_pos_dev, 0.95).item()),
                "lam_neg_p95_dev": float(torch.quantile(lam_neg_dev, 0.95).item()),
                "c_emp_p95_dev": float(torch.quantile(c_dev, 0.95).item()),
                # symmetry gap between +/- branches
                "lam_pos_neg_max_gap": float((lam_pos - lam_neg).abs().max().item()),
                "lam_pos_neg_p95_gap": float(torch.quantile((lam_pos - lam_neg).abs(), 0.95).item()),
            }
    
    for i in pbar:
        # LR decay schedule
        if lr_decay < 1.0 and i > 0 and i % lr_decay_interval == 0:
            old_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            new_lr = optimizer.param_groups[0]['lr']
            pbar.write(f"Epoch {i:6d}: Learning rate decayed from {old_lr:.6e} to {new_lr:.6e}")
        
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        
        # Compute gradient norm before step
        grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item() if any(p.grad is not None for p in model.parameters()) else 0.0
        
        optimizer.step()
        loss = loss.item()
        
        # Update progress bar with current loss and percentage (only in interactive mode)
        pct_complete = 100.0 * (i + 1) / epochs
        if is_interactive:
            pbar.set_postfix({
                'loss': f'{loss:.6e}',
                'pct': f'{pct_complete:.1f}%',
                'grad_norm': f'{grad_norm:.6e}'
            })
            pbar.refresh()  # Force refresh to ensure updates are visible
        
        # Evaluate at test intervals or at the end
        should_evaluate = (i % test_every_n_epochs == 0) or (i == epochs - 1)
        if should_evaluate and i != last_evaluated_epoch:
            with torch.no_grad():
                train_pred_mse = F.mse_loss(model(x), y).item()
                test_pred_mse = F.mse_loss(model(test_x), test_y).item()
                beta_hat = model.beta()
                param_mse = F.mse_loss(beta_hat, beta_star).item()
                inv_stats = _invariant_stats()
                
                # Compute beta norms
                beta_l2 = l2_norm(beta_hat)
                beta_l1 = l1_norm(beta_hat)
                
                # Compute beta update rate (relative change between consecutive evals)
                # Uses beta_hat_prev_eval (ONLY updated at eval checkpoints)
                if beta_hat_prev_eval is not None:
                    numer = (beta_hat - beta_hat_prev_eval).norm().item()
                    denom = beta_hat_prev_eval.norm().item() + 1e-12
                    beta_update_rate = numer / denom
                    delta_beta = numer  # For logging
                else:
                    delta_beta = float('inf')
                    beta_update_rate = float('inf')
                
                # Update prev for next eval (ONLY inside eval block, not per-epoch)
                beta_hat_prev_eval = beta_hat.clone().detach()
                eval_count += 1
                
                # Also keep old tracking for backward compat
                beta_hat_prev = beta_hat.clone()
                prev_eval_epoch = i
                
                # Train split
                all_results.append({
                    'epoch': i,
                    'split': 'train',
                    'pred_mse': train_pred_mse,
                    'param_mse': param_mse,
                    'grad_norm': grad_norm,
                    'delta_beta': delta_beta,
                    'beta_update_rate': beta_update_rate,
                    'beta_l2': beta_l2,
                    'beta_l1': beta_l1,
                    **inv_stats,
                })
                
                # Test split
                all_results.append({
                    'epoch': i,
                    'split': 'test',
                    'pred_mse': test_pred_mse,
                    'param_mse': param_mse,
                    'grad_norm': grad_norm,
                    'delta_beta': delta_beta,
                    'beta_update_rate': beta_update_rate,
                    'beta_l2': beta_l2,
                    'beta_l1': beta_l1,
                    **inv_stats,
                })
                
                norms.append(
                    pd.DataFrame({
                        'norm': ['l1', 'l2'],
                        'value': [l1_norm(beta_hat), l2_norm(beta_hat)],
                        'epoch': [i, i]
                    })
                )
                # Print training progress (throttled).
                # In non-interactive environments (e.g. SLURM, many notebooks), tqdm is disabled;
                # use print(..., flush=True) so logs still show up.
                should_log = (i % log_every_n_epochs == 0) or (i == epochs - 1)
                if should_log:
                    msg = (
                        f"Epoch {i:6d} ({pct_complete:.1f}%): "
                        f"Train pred MSE = {train_pred_mse:.6e}, "
                        f"Test pred MSE = {test_pred_mse:.6e}, "
                        f"Param MSE = {param_mse:.6e}, "
                        f"Grad norm = {grad_norm:.6e}, "
                        f"Beta update rate = {beta_update_rate:.6e}, "
                        f"c_emp_max_dev = {inv_stats['c_emp_max_dev']:.3e}, "
                        f"lam_pos_max_dev = {inv_stats['lam_pos_max_dev']:.3e}, "
                        f"lam_pos_neg_max_gap = {inv_stats['lam_pos_neg_max_gap']:.3e}"
                    )
                    if is_interactive:
                        pbar.write(msg)
                    else:
                        print(msg, flush=True)
                last_evaluated_epoch = i
                
                # === Fixed-point detection ===
                # Check if model is at a fixed point (beta not changing significantly)
                fixed_point_met = (beta_update_rate < fixed_point_beta_rate)
                if fixed_point_grad_norm > 0.0:
                    fixed_point_met = fixed_point_met and (grad_norm < fixed_point_grad_norm)
                
                # Update consecutive counter (increment if met, RESET to 0 otherwise)
                if fixed_point_met:
                    consecutive_fixed_point_evals += 1
                else:
                    consecutive_fixed_point_evals = 0
                
                # === Fixed-point stopping (requires >= 2 evals to define a rate) ===
                if (eval_count >= 2 and 
                    i >= min_epochs_before_stop and 
                    consecutive_fixed_point_evals >= fixed_point_consecutive_evals):
                    detected_stop_reason = "fixed_point"
                    break
                
                # === Multi-criterion stopping (existing logic, kept for backward compat) ===
                # These criteria are only evaluated at eval checkpoints
                pred_mse_met = (train_pred_mse < stop_pred_mse)
                beta_rate_met = False
                grad_norm_met = False
                
                if stop_beta_rate > 0:
                    beta_rate_met = (beta_update_rate < stop_beta_rate) or (beta_update_rate == 0.0)
                elif beta_update_rate == 0.0:
                    if train_pred_mse < 1e-5:
                        beta_rate_met = True
                
                if stop_grad_norm > 0:
                    grad_norm_met = (grad_norm < stop_grad_norm)
                
                # Determine stop reason based on criteria combination
                should_stop = False
                
                if stop_beta_rate > 0 and stop_grad_norm > 0:
                    if pred_mse_met and beta_rate_met and grad_norm_met:
                        should_stop = True
                        detected_stop_reason = "train_pred_mse_and_beta_rate_and_grad_norm"
                elif stop_beta_rate > 0:
                    if pred_mse_met and beta_rate_met:
                        should_stop = True
                        detected_stop_reason = "train_pred_mse_and_beta_rate"
                elif stop_grad_norm > 0:
                    if pred_mse_met and grad_norm_met:
                        should_stop = True
                        detected_stop_reason = "train_pred_mse_and_grad_norm"
                elif beta_rate_met and train_pred_mse < 1e-5:
                    should_stop = True
                    detected_stop_reason = "train_pred_mse"
                elif pred_mse_met:
                    should_stop = True
                    detected_stop_reason = "train_pred_mse"
                
                if should_stop:
                    break
        
        # === Legacy loss threshold stop (GUARDED - safe by default) ===
        # This prevents premature stops where loss is tiny but beta is still moving
        if not disable_legacy_loss_stop:
            if loss < threshold:
                # Guard conditions (ALL must be true to allow legacy stop):
                # 1. Minimum epochs reached
                # 2. At least one eval has occurred (so beta_update_rate is valid)
                # 3. Near fixed-point (beta not still moving fast)
                can_stop_epochs = (i >= min_epochs_before_stop)
                can_stop_eval = (eval_count >= 1)
                near_fixed_point = (consecutive_fixed_point_evals >= 1) or \
                                   (beta_update_rate < fixed_point_beta_rate)
                
                if can_stop_epochs and can_stop_eval and near_fixed_point:
                    detected_stop_reason = "loss_threshold_legacy_guarded"
                    break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            pbar.write('='*80)
            pbar.write('WARNING: Learning rate tuning triggered!')
            pbar.write(f'  Loss = {loss:.6e}, decreasing learning rate to {lr:.6e}')
            pbar.write('='*80)
            # Update optimizer LR before recursive call
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # Recursive call returns (df, model, norm_df, stop_reason, final_epoch)
            return train(or_model, train_data, test_data, beta_star, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, stop_pred_mse=stop_pred_mse, stop_beta_rate=stop_beta_rate, stop_grad_norm=stop_grad_norm, lr_decay=lr_decay, lr_decay_interval=lr_decay_interval, save_folder=save_folder,
                         min_epochs_before_stop=min_epochs_before_stop, require_eval_before_stop=require_eval_before_stop,
                         disable_legacy_loss_stop=disable_legacy_loss_stop, fixed_point_beta_rate=fixed_point_beta_rate,
                         fixed_point_consecutive_evals=fixed_point_consecutive_evals, fixed_point_grad_norm=fixed_point_grad_norm)
    
    # Ensure final epoch is evaluated
    if last_evaluated_epoch != i:
        with torch.no_grad():
            train_pred_mse = F.mse_loss(model(x), y).item()
            test_pred_mse = F.mse_loss(model(test_x), test_y).item()
            beta_hat = model.beta()
            param_mse = F.mse_loss(beta_hat, beta_star).item()
            inv_stats = _invariant_stats()
            
            # Compute beta norms
            beta_l2 = l2_norm(beta_hat)
            beta_l1 = l1_norm(beta_hat)
        
        # Compute gradient norm (need to do forward/backward again, OUTSIDE no_grad)
        optimizer.zero_grad()
        loss_temp = F.mse_loss(model(x), y)
        loss_temp.backward()
        grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item() if any(p.grad is not None for p in model.parameters()) else 0.0
        
        # Compute beta update rate (using relative norm formula, consistent with eval block)
        if beta_hat_prev_eval is not None:
            numer = (beta_hat - beta_hat_prev_eval).norm().item()
            denom = beta_hat_prev_eval.norm().item() + 1e-12
            beta_update_rate = numer / denom
            delta_beta = numer
        else:
            delta_beta = float('inf')
            beta_update_rate = float('inf')
        
        # Update eval tracking for final epoch
        beta_hat_prev_eval = beta_hat.clone().detach()
        eval_count += 1
        
        all_results.append({
            'epoch': i,
            'split': 'train',
            'pred_mse': train_pred_mse,
            'param_mse': param_mse,
            'grad_norm': grad_norm,
            'delta_beta': delta_beta,
            'beta_update_rate': beta_update_rate,
            'beta_l2': beta_l2,
            'beta_l1': beta_l1,
            **inv_stats,
        })
        
        all_results.append({
            'epoch': i,
            'split': 'test',
            'pred_mse': test_pred_mse,
            'param_mse': param_mse,
            'grad_norm': grad_norm,
            'delta_beta': delta_beta,
            'beta_update_rate': beta_update_rate,
            'beta_l2': beta_l2,
            'beta_l1': beta_l1,
            **inv_stats,
        })
        
        norms.append(
                pd.DataFrame({
                    'norm': ['l1', 'l2'],
                    'value': [l1_norm(beta_hat), l2_norm(beta_hat)],
                    'epoch': [i, i]
                })
            )
    
    # Print final training results
    final_train = [r for r in all_results if r['split'] == 'train' and r['epoch'] == i][0]
    final_test = [r for r in all_results if r['split'] == 'test' and r['epoch'] == i][0]
    
    # Determine stop reason using explicit labels
    if i == epochs - 1:
        stop_reason = "max_epochs"
    elif detected_stop_reason is not None:
        stop_reason = detected_stop_reason
    else:
        # Fallback (should not happen with proper tracking)
        stop_reason = "unknown"
    
    print(f"\nTraining completed at epoch {i}")
    print(f"Stop reason: {stop_reason}")
    print(f"Final Train pred MSE = {final_train['pred_mse']:.6e}")
    print(f"Final Test pred MSE = {final_test['pred_mse']:.6e}")
    print(f"Final Param MSE = {final_test['param_mse']:.6e}")
    if 'grad_norm' in final_train:
        print(f"Final Grad norm = {final_train['grad_norm']:.6e}")
    if 'beta_update_rate' in final_train:
        print(f"Final Beta update rate = {final_train['beta_update_rate']:.6e}")
    
    # Print success flag (use stop_pred_mse threshold)
    success_flag = final_train['pred_mse'] < stop_pred_mse
    print(f"Success flag: {success_flag} (train_pred_mse < {stop_pred_mse:.2e})")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    norm_df = pd.concat(norms).reset_index(drop=True) if norms else pd.DataFrame()
    
    # Save results_meta.json if save_folder is provided
    if save_folder is not None:
        meta = {
            'stop_reason': stop_reason,
            'final_epoch': int(i),
            'final_train_pred_mse': float(final_train['pred_mse']),
            'final_test_pred_mse': float(final_test['pred_mse']),
            'final_param_mse': float(final_test['param_mse'])
        }
        if 'grad_norm' in final_train:
            meta['final_grad_norm'] = float(final_train['grad_norm'])
        if 'beta_update_rate' in final_train:
            # Handle inf values for JSON serialization
            bur = final_train['beta_update_rate']
            meta['final_beta_update_rate'] = float(bur) if np.isfinite(bur) else None
        
        # Add new diagnostic fields for fixed-point stopping
        meta['eval_count'] = eval_count
        meta['final_consecutive_fixed_point_evals'] = consecutive_fixed_point_evals
        meta['min_epochs_before_stop'] = min_epochs_before_stop
        meta['fixed_point_beta_rate'] = fixed_point_beta_rate
        meta['fixed_point_consecutive_evals'] = fixed_point_consecutive_evals
        meta['fixed_point_grad_norm'] = fixed_point_grad_norm
        meta['legacy_loss_stop_disabled'] = disable_legacy_loss_stop
        
        meta_path = os.path.join(save_folder, 'results_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Results metadata saved to {meta_path}")
    
    return df, model, norm_df, stop_reason, i

def main(args):
    # Print immediately to confirm script is running
    print("\n" + "="*80)
    print("SCRIPT STARTED - Running diagonal network BG experiment...")
    print("="*80 + "\n")
    sys.stdout.flush()
    
    make_deterministic(args.seed, use_gpu=False)
    # Set default dtype to float64 for higher numerical precision
    torch.set_default_dtype(torch.float64)
    # Print experiment settings
    print("="*80)
    print("EXPERIMENT SETTINGS")
    print("="*80)
    print(f"Seed: {args.seed}")
    print(f"Input dimension: {args.inp_dim}")
    print(f"Sparsity (rho): {args.rho}")
    print(f"Training samples: {args.n_train}")
    print(f"Test samples: {args.n_test}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.epochs}")
    print(f"Test every N epochs: {args.test_every_n_epochs}")
    print(f"Convergence threshold: {args.threshold}")
    print(f"Stop pred MSE: {args.stop_pred_mse if args.stop_pred_mse is not None else args.threshold}")
    print(f"Stop beta rate: {args.stop_beta_rate if args.stop_beta_rate > 0 else 'disabled'}")
    print(f"Stop grad norm: {args.stop_grad_norm if args.stop_grad_norm > 0 else 'disabled'}")
    print(f"LR decay: {args.lr_decay if args.lr_decay < 1.0 else 'disabled'}")
    print(f"LR decay interval: {args.lr_decay_interval}")
    print(f"Scaling: {args.scaling}")
    print(f"Lambda (λ): {args.lmda}")
    print(f"C parameter: {args.c}")
    print(f"Initialization method: {args.init_method}")
    print(f"Learning rate tuning: {not args.no_tuning}")
    print(f"Save folder: {args.save_folder}")
    # Print c_mode info
    c_mode = getattr(args, 'c_mode', 'homogeneous')
    print(f"C mode: {c_mode}")
    if c_mode == 'mixture':
        print(f"  c_A={args.c_A}, c_B={args.c_B}, pi_A={args.pi_A}")
    elif c_mode == 'support':
        print(f"  c_nz={args.c_nz}, c_z={args.c_z}")
    # Print fixed-point stopping settings
    print(f"Fixed-point stopping settings:")
    print(f"  min_epochs_before_stop: {args.min_epochs_before_stop if args.min_epochs_before_stop is not None else f'auto ({args.test_every_n_epochs})'}")
    print(f"  require_eval_before_stop: {args.require_eval_before_stop}")
    print(f"  disable_legacy_loss_stop: {args.disable_legacy_loss_stop}")
    print(f"  fixed_point_beta_rate: {args.fixed_point_beta_rate}")
    print(f"  fixed_point_consecutive_evals: {args.fixed_point_consecutive_evals}")
    print(f"  fixed_point_grad_norm: {args.fixed_point_grad_norm if args.fixed_point_grad_norm > 0 else 'disabled'}")
    print("="*80)
    print("Starting training...")
    print("="*80)
    sys.stdout.flush()  # Ensure output is flushed

    # Use separate generators for train X, test X, teacher beta_star, and c mask
    gen_train_x = torch.Generator(device='cpu').manual_seed(args.seed + 0)
    gen_test_x = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    gen_beta_star = torch.Generator(device='cpu').manual_seed(args.seed + 2)
    gen_c_mask = torch.Generator(device='cpu').manual_seed(args.seed + 3)  # For mixture mode
    
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    
    # Sample Bernoulli-Gaussian teacher
    beta_star = sample_beta_star_bg(args.inp_dim, args.rho, generator=gen_beta_star)
    
    # Build per-coordinate c vector if needed (must be after beta_star for support mode)
    c_vec = build_c_vec(args, beta_star, gen_c_mask)
    
    # # Sample design matrices with RS scaling
    # # Train: X_{ij} ~ N(0, 1/n_train)
    # x = torch.randn(args.n_train, args.inp_dim, generator=gen_train_x) / math.sqrt(args.n_train)
    
    # # Test: X_{ij} ~ N(0, 1/n_test)
    # test_x = torch.randn(args.n_test, args.inp_dim, generator=gen_test_x) / math.sqrt(args.n_test)
    #NOTE CHANGING FOR REPLICA METHOD MATCH
    # Sample design matrices with replica/theory scaling:
    # X_{ij} ~ N(0, 1/inp_dim)
    x = torch.randn(args.n_train, args.inp_dim, generator=gen_train_x) / math.sqrt(args.inp_dim)
    test_x = torch.randn(args.n_test, args.inp_dim, generator=gen_test_x) / math.sqrt(args.inp_dim)
        
    # Noiseless outputs: y = X @ beta_star
    y = x @ beta_star
    test_y = test_x @ beta_star
    
    # Initialize network (with optional per-coordinate c)
    net = DiagonalNet(args.inp_dim, scaling=args.scaling, lmda=args.lmda, c=args.c, c_vec=c_vec, init_method=args.init_method)
    
    # Verify initialization invariants for homogeneous complex init
    if c_vec is None and args.init_method == "complex":
        check_init_invariants(net, args.c, args.lmda)
        print(f"Initialization invariants verified: c={args.c}, lmda={args.lmda}")
    
    # Train
    df, net, norm_df, stop_reason, final_epoch = train(
        net, 
        (x, y), 
        (test_x, test_y), 
        beta_star,
        test_every_n_epochs=args.test_every_n_epochs,
        lr=args.lr, 
        epochs=args.epochs, 
        lr_tuning=(not args.no_tuning), 
        threshold=args.threshold,
        stop_pred_mse=args.stop_pred_mse,
        stop_beta_rate=args.stop_beta_rate,
        stop_grad_norm=args.stop_grad_norm,
        lr_decay=args.lr_decay,
        lr_decay_interval=args.lr_decay_interval,
        save_folder=args.save_folder,
        # Fixed-point stopping args
        min_epochs_before_stop=args.min_epochs_before_stop,
        require_eval_before_stop=args.require_eval_before_stop,
        disable_legacy_loss_stop=args.disable_legacy_loss_stop,
        fixed_point_beta_rate=args.fixed_point_beta_rate,
        fixed_point_consecutive_evals=args.fixed_point_consecutive_evals,
        fixed_point_grad_norm=args.fixed_point_grad_norm,
    )
    
    # Save results
    df.to_feather(os.path.join(args.save_folder, 'df.feather'))
    norm_df.to_feather(os.path.join(args.save_folder, 'norm_df.feather'))
    torch.save(beta_star, os.path.join(args.save_folder, 'beta_star.pt'))
    torch.save(net.state_dict(), os.path.join(args.save_folder, 'model.pt'))
    
    # Save c_vec if using heterogeneous c
    if c_vec is not None:
        np.save(os.path.join(args.save_folder, 'c_vec.npy'), c_vec)
        print(f"Saved c_vec to {os.path.join(args.save_folder, 'c_vec.npy')}")
    
    # Save experiment config including c_mode info
    config = {
        'seed': args.seed,
        'inp_dim': args.inp_dim,
        'n_train': args.n_train,
        'n_test': args.n_test,
        'rho': args.rho,
        'c': args.c,
        'lmda': args.lmda,
        'c_mode': getattr(args, 'c_mode', 'homogeneous'),
    }
    if config['c_mode'] == 'mixture':
        config.update({'c_A': args.c_A, 'c_B': args.c_B, 'pi_A': args.pi_A})
    elif config['c_mode'] == 'support':
        config.update({'c_nz': args.c_nz, 'c_z': args.c_z})
    
    config_path = os.path.join(args.save_folder, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to {args.save_folder}")

def build_c_vec(args, beta_star, gen_c_mask):
    """
    Build per-coordinate c vector based on c_mode.
    
    Args:
        args: Parsed arguments with c_mode, c_A, c_B, pi_A, c_nz, c_z
        beta_star: Teacher vector (used for support mode)
        gen_c_mask: Generator for reproducible mask sampling in mixture mode
    
    Returns:
        c_vec: numpy array of shape (inp_dim,) or None for homogeneous mode
    """
    c_mode = getattr(args, 'c_mode', 'homogeneous')
    
    if c_mode == 'homogeneous':
        return None  # Use scalar c
    
    inp_dim = args.inp_dim
    
    if c_mode == 'mixture':
        # Validate required args
        if args.c_A is None or args.c_B is None or args.pi_A is None:
            raise ValueError("c_mode='mixture' requires --c_A, --c_B, and --pi_A")
        if not (0.0 < args.pi_A < 1.0):
            raise ValueError("--pi_A must be in (0, 1)")
        if args.c_A <= 0 or args.c_B <= 0:
            raise ValueError("--c_A and --c_B must be > 0")
        
        # Sample coordinate mask: g_i ~ Bernoulli(pi_A)
        # Use separate generator for reproducibility
        mask = torch.rand(inp_dim, generator=gen_c_mask) < args.pi_A
        
        # Set c_i = c_A if mask_i=1, else c_B
        c_vec = np.where(mask.numpy(), args.c_A, args.c_B)
        
        frac_A = mask.float().mean().item()
        print(f"[c_mode=mixture] pi_A={args.pi_A}, empirical frac_A={frac_A:.4f}")
        print(f"[c_mode=mixture] c_A={args.c_A}, c_B={args.c_B}")
        
        return c_vec
    
    elif c_mode == 'support':
        # Validate required args
        if args.c_nz is None or args.c_z is None:
            raise ValueError("c_mode='support' requires --c_nz and --c_z")
        if args.c_nz <= 0 or args.c_z <= 0:
            raise ValueError("--c_nz and --c_z must be > 0")
        
        # Use teacher support mask: m_i = 1[beta_star_i != 0]
        if beta_star is None:
            raise ValueError("beta_star required for c_mode='support'")
        
        support_mask = (beta_star != 0).numpy()
        
        # Set c_i = c_nz where support, else c_z
        c_vec = np.where(support_mask, args.c_nz, args.c_z)
        
        frac_nz = support_mask.mean()
        print(f"[c_mode=support] support fraction={frac_nz:.4f} (rho={args.rho})")
        print(f"[c_mode=support] c_nz={args.c_nz}, c_z={args.c_z}")
        
        return c_vec
    
    else:
        raise ValueError(f"Unknown c_mode: {c_mode}")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--n_train', type=int, default=50)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--inp_dim', type=int, default=100)
    parser.add_argument('--rho', type=float, required=True, help='Sparsity parameter for Bernoulli-Gaussian teacher')
    parser.add_argument('--threshold', type=float, default=1e-9)
    parser.add_argument('--stop_pred_mse', type=float, default=1e-10, help='Stop when train_pred_mse < this value')
    parser.add_argument('--stop_beta_rate', type=float, default=0.0, help='Stop when beta_update_rate < this value (disabled if 0.0)')
    parser.add_argument('--stop_grad_norm', type=float, default=0.0, help='Stop when grad_norm < this value (disabled if 0.0)')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='LR decay factor (no decay if 1.0)')
    parser.add_argument('--lr_decay_interval', type=int, default=2000, help='Epochs between LR decay steps')
    parser.add_argument('--no_tuning', action='store_true')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=200000)
    parser.add_argument('--test_every_n_epochs', type=int, default=200, help='How often to evaluate on test set')
    parser.add_argument('--scaling', type=float, default=1.)
    parser.add_argument('--w_scaling', type=float, default=1.)
    parser.add_argument('--lmda', type=float, default=0.)
    parser.add_argument('--c', type=float, default=0.001)
    parser.add_argument('--init_method', type=str, default='complex', choices=['simple', 'complex'])
    
    # Heterogeneous c initialization (Steps 1 & 2)
    parser.add_argument('--c_mode', type=str, default='homogeneous',
                        choices=['homogeneous', 'mixture', 'support'],
                        help='Mode for per-coordinate c: homogeneous (default), mixture, or support')
    # Mixture mode arguments (Step 1)
    parser.add_argument('--c_A', type=float, default=None,
                        help='Group-A c value for mixture mode')
    parser.add_argument('--c_B', type=float, default=None,
                        help='Group-B c value for mixture mode')
    parser.add_argument('--pi_A', type=float, default=None,
                        help='Probability a coordinate belongs to group A in mixture mode')
    # Support mode arguments (Step 2)
    parser.add_argument('--c_nz', type=float, default=None,
                        help='c value for nonzero teacher coordinates in support mode')
    parser.add_argument('--c_z', type=float, default=None,
                        help='c value for zero teacher coordinates in support mode')
    
    # Fixed-point stopping arguments (prevent premature legacy stops)
    parser.add_argument('--min_epochs_before_stop', type=int, default=None,
                        help='Min epoch before any early stop (default: auto=test_every_n_epochs)')
    parser.add_argument('--require_eval_before_stop', action='store_true',
                        help='Require at least one eval before stopping')
    parser.add_argument('--disable_legacy_loss_stop', action='store_true',
                        help='Completely disable legacy loss<threshold stop')
    parser.add_argument('--fixed_point_beta_rate', type=float, default=1e-6,
                        help='Beta update rate threshold for fixed-point detection')
    parser.add_argument('--fixed_point_consecutive_evals', type=int, default=2,
                        help='Consecutive evals at fixed-point before stopping')
    parser.add_argument('--fixed_point_grad_norm', type=float, default=0.0,
                        help='Grad norm threshold for fixed-point (0.0=disabled)')
    
    return parser

if __name__ == '__main__':
    # Print immediately when script starts
    print("Starting diagonal network pretrain BG script...", flush=True)
    parser = get_parser()
    args = parser.parse_args()
    main(args)

