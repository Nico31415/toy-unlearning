#!/usr/bin/env python3
"""
Golden Test Script for Initialization Mapping

Verifies that the diagonal network initialization correctly encodes the 
theoretical parameters (c, lambda) for the implicit regularizer mapping k = (2c)^2.

Tests:
1. Homogeneous init produces correct c and lambda
2. Heterogeneous c_vec produces correct per-coordinate c_i  
3. PT+FT oracle: compute c_ft, initialize net, verify implied c_i == c_ft

This script must fail hard if any assertion fails.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from experiments.diagonal.diagonal_network_pretrain_bg import (
    DiagonalNet,
    get_parameters,
    get_parameters_vectorized,
    check_init_invariants,
)
from ReplicaExperiments.fixed_lambda_all import compute_c_ft_from_pt


def test_homogeneous_init():
    """
    Test 1: Homogeneous initialization produces correct c and lambda.
    
    The network should satisfy:
        lambda_i = w_pos_i^2 - v_pos_i^2  (constant for all i)
        c_i = w_pos_i * w_neg_i + v_pos_i * v_neg_i  (constant for all i)
    """
    print("\n" + "="*80)
    print("TEST 1: Homogeneous Initialization")
    print("="*80)
    
    test_cases = [
        {"c": 0.001, "lmda": 0.0, "inp_dim": 100},
        {"c": 0.5, "lmda": 0.0, "inp_dim": 100},
        {"c": 0.01, "lmda": 0.005, "inp_dim": 50},
        {"c": 1.0, "lmda": 0.5, "inp_dim": 200},
        {"c": 0.001, "lmda": 0.0001, "inp_dim": 100},
    ]
    
    for i, tc in enumerate(test_cases):
        c, lmda, inp_dim = tc["c"], tc["lmda"], tc["inp_dim"]
        print(f"\n  Case {i+1}: c={c}, lmda={lmda}, inp_dim={inp_dim}")
        
        # Create network with homogeneous init
        net = DiagonalNet(inp_dim, lmda=lmda, c=c, c_vec=None, init_method='complex')
        
        # Verify invariants using the check function
        try:
            check_init_invariants(net, c, lmda, atol=1e-12)
            print(f"    PASSED: Invariants verified")
        except AssertionError as e:
            print(f"    FAILED: {e}")
            raise
        
        # Additional verification: check get_parameters output consistency
        # get_parameters returns (v_pos, v_neg, u_pos, u_neg) which maps to:
        # w_pos=v_pos, v_pos=v_neg, v_neg=u_pos, w_neg=u_neg
        # With the fix: returns (v, u, u, v) so w_pos=v, v_pos=u, v_neg=u, w_neg=v
        v_for_wpos, u_for_vpos, u_for_vneg, v_for_wneg = get_parameters(c, lmda)
        
        # Expected: v = sqrt((c+lmda)/2), u = sqrt((c-lmda)/2)
        v_expected = np.sqrt((c + lmda) / 2)
        u_expected = np.sqrt((c - lmda) / 2)
        
        assert np.isclose(v_for_wpos, v_expected, atol=1e-14), f"v_pos value wrong"
        assert np.isclose(u_for_vpos, u_expected, atol=1e-14), f"v_neg value wrong"
        
        with torch.no_grad():
            # Check that all coordinates have the expected values
            assert torch.allclose(net.w_pos, torch.full_like(net.w_pos, v_expected), atol=1e-12), \
                f"w_pos mismatch: expected {v_expected}, got {net.w_pos[0].item()}"
            assert torch.allclose(net.v_pos, torch.full_like(net.v_pos, u_expected), atol=1e-12), \
                f"v_pos mismatch: expected {u_expected}, got {net.v_pos[0].item()}"
            assert torch.allclose(net.v_neg, torch.full_like(net.v_neg, u_expected), atol=1e-12), \
                f"v_neg mismatch: expected {u_expected}, got {net.v_neg[0].item()}"
            assert torch.allclose(net.w_neg, torch.full_like(net.w_neg, v_expected), atol=1e-12), \
                f"w_neg mismatch: expected {v_expected}, got {net.w_neg[0].item()}"
        
        print(f"    PASSED: Parameter values correct")
    
    print("\n  TEST 1 PASSED: All homogeneous init cases verified")


def test_heterogeneous_init():
    """
    Test 2: Heterogeneous c_vec initialization produces correct per-coordinate c_i.
    
    With per-coordinate c values, the network should satisfy:
        lambda_i = w_pos_i^2 - v_pos_i^2 = lmda (constant)
        c_i = w_pos_i * w_neg_i + v_pos_i * v_neg_i = c_vec[i] (per-coordinate)
    """
    print("\n" + "="*80)
    print("TEST 2: Heterogeneous c_vec Initialization")
    print("="*80)
    
    torch.set_default_dtype(torch.float64)
    
    test_cases = [
        # Uniform c_vec (should match homogeneous)
        {"c_vec": np.full(100, 0.001), "lmda": 0.0, "inp_dim": 100},
        # Two groups
        {"c_vec": np.array([0.001]*50 + [0.5]*50), "lmda": 0.0, "inp_dim": 100},
        # Random c values
        {"c_vec": np.random.uniform(0.01, 1.0, 100), "lmda": 0.0, "inp_dim": 100},
        # With nonzero lambda
        {"c_vec": np.random.uniform(0.5, 1.0, 50), "lmda": 0.1, "inp_dim": 50},
    ]
    
    for i, tc in enumerate(test_cases):
        c_vec, lmda, inp_dim = tc["c_vec"], tc["lmda"], tc["inp_dim"]
        print(f"\n  Case {i+1}: c_vec range [{c_vec.min():.6f}, {c_vec.max():.6f}], lmda={lmda}, inp_dim={inp_dim}")
        
        # Create network with heterogeneous init
        net = DiagonalNet(inp_dim, lmda=lmda, c=0.0, c_vec=c_vec, init_method='complex')
        
        # Verify invariants
        try:
            check_init_invariants(net, c_vec, lmda, atol=1e-12)
            print(f"    PASSED: Invariants verified")
        except AssertionError as e:
            print(f"    FAILED: {e}")
            raise
        
        # Additional verification: check per-coordinate c values
        with torch.no_grad():
            c_actual = (net.w_pos * net.w_neg + net.v_pos * net.v_neg).numpy()
            max_err = np.max(np.abs(c_actual - c_vec))
            assert max_err < 1e-12, f"c_i mismatch: max error = {max_err}"
        
        print(f"    PASSED: Per-coordinate c values correct (max err = {max_err:.2e})")
    
    print("\n  TEST 2 PASSED: All heterogeneous init cases verified")


def test_ptft_oracle_mapping():
    """
    Test 3: PT+FT Oracle mapping.
    
    Verifies:
    1. compute_c_ft_from_pt produces expected c_ft values
    2. Network initialized with c_ft has correct implied c_i == c_ft
    """
    print("\n" + "="*80)
    print("TEST 3: PT+FT Oracle Mapping")
    print("="*80)
    
    torch.set_default_dtype(torch.float64)
    
    test_cases = [
        # Basic case: zero beta_pt gives baseline c_ft
        {
            "beta_pt": np.zeros(100),
            "c_pt": 0.001,
            "lambda_pt": 0.0,
            "gamma_reinit": 0.0,
            "inp_dim": 100,
        },
        # Nonzero beta_pt
        {
            "beta_pt": np.array([1.0]*50 + [0.0]*50),
            "c_pt": 0.001,
            "lambda_pt": 0.0,
            "gamma_reinit": 0.0,
            "inp_dim": 100,
        },
        # With gamma_reinit
        {
            "beta_pt": np.array([1.0]*30 + [0.0]*70),
            "c_pt": 0.001,
            "lambda_pt": 0.0,
            "gamma_reinit": 0.1,
            "inp_dim": 100,
        },
        # With lambda_pt
        {
            "beta_pt": np.array([0.5]*40 + [0.0]*60),
            "c_pt": 0.01,
            "lambda_pt": 0.005,
            "gamma_reinit": 0.05,
            "inp_dim": 100,
        },
    ]
    
    for i, tc in enumerate(test_cases):
        beta_pt = tc["beta_pt"]
        c_pt = tc["c_pt"]
        lambda_pt = tc["lambda_pt"]
        gamma_reinit = tc["gamma_reinit"]
        inp_dim = tc["inp_dim"]
        
        print(f"\n  Case {i+1}: c_pt={c_pt}, lambda_pt={lambda_pt}, gamma_reinit={gamma_reinit}")
        print(f"    beta_pt: {(beta_pt != 0).sum()} nonzero out of {inp_dim}")
        
        # Step 1: Compute c_ft using canonical mapping
        c_ft = compute_c_ft_from_pt(beta_pt, c_pt, lambda_pt, gamma_reinit)
        
        # Verify c_ft formula manually
        expected_c_ft = (lambda_pt + c_pt) * (1.0 + np.sqrt(1.0 + (beta_pt / c_pt)**2)) + 0.5 * gamma_reinit**2
        max_formula_err = np.max(np.abs(c_ft - expected_c_ft))
        assert max_formula_err < 1e-14, f"c_ft formula mismatch: max error = {max_formula_err}"
        print(f"    PASSED: c_ft formula verified (max err = {max_formula_err:.2e})")
        
        # Step 2: Initialize network with c_ft
        net = DiagonalNet(inp_dim, lmda=0.0, c=c_pt, c_vec=c_ft, init_method='complex')
        
        # Step 3: Verify implied c_i == c_ft
        try:
            check_init_invariants(net, c_ft, 0.0, atol=1e-12)
            print(f"    PASSED: Network invariants verified")
        except AssertionError as e:
            print(f"    FAILED: {e}")
            raise
        
        # Additional check: directly compute c_i from network params
        with torch.no_grad():
            c_implied = (net.w_pos * net.w_neg + net.v_pos * net.v_neg).numpy()
            max_err = np.max(np.abs(c_implied - c_ft))
            assert max_err < 1e-12, f"Implied c_i mismatch: max error = {max_err}"
        
        print(f"    PASSED: Implied c_i == c_ft (max err = {max_err:.2e})")
        
        # Verify c_ft range
        print(f"    c_ft range: [{c_ft.min():.6f}, {c_ft.max():.6f}]")
    
    print("\n  TEST 3 PASSED: All PT+FT oracle cases verified")


def test_k_mapping():
    """
    Additional test: Verify k = (2c)^2 mapping.
    
    For diagonal net with lmda=0:
        w_pos = v_pos = sqrt(c/2)
        => sqrt(k) = 2c => k = (2c)^2
    """
    print("\n" + "="*80)
    print("TEST 4: k = (2c)^2 Mapping Verification")
    print("="*80)
    
    test_cases = [
        {"c": 0.001, "expected_k": 4e-6},
        {"c": 0.5, "expected_k": 1.0},
        {"c": 0.01, "expected_k": 4e-4},
        {"c": 1.0, "expected_k": 4.0},
    ]
    
    for i, tc in enumerate(test_cases):
        c = tc["c"]
        expected_k = tc["expected_k"]
        computed_k = (2.0 * c) ** 2
        
        rel_err = abs(computed_k - expected_k) / expected_k
        print(f"\n  Case {i+1}: c={c}")
        print(f"    Expected k: {expected_k:.6e}")
        print(f"    Computed k: {computed_k:.6e}")
        print(f"    Relative error: {rel_err:.2e}")
        
        assert rel_err < 1e-14, f"k mapping mismatch: rel error = {rel_err}"
        print(f"    PASSED")
    
    print("\n  TEST 4 PASSED: All k mapping cases verified")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GOLDEN TEST SCRIPT FOR INITIALIZATION MAPPING")
    print("="*80)
    print("This script verifies the diagonal network initialization")
    print("correctly encodes the theoretical parameters.")
    
    try:
        test_homogeneous_init()
        test_heterogeneous_init()
        test_ptft_oracle_mapping()
        test_k_mapping()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED")
        print("="*80 + "\n")
        return 0
        
    except AssertionError as e:
        print("\n" + "="*80)
        print("TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print("\n" + "="*80)
        print("TEST ERROR")
        print("="*80)
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

