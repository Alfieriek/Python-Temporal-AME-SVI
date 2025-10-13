"""
Integration tests for temporal AME project.

This script tests the full pipeline from data generation to inference
to visualization, ensuring all components work together correctly.

Author: Sean Plummer
Date: October 2025
"""

import sys
import os
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.temporal_ame import TemporalAMEModel
from models.static_ame import StaticAMEModel
from inference.naive_mf import TemporalAMENaiveMFVI
from inference.structured_mf import TemporalAMEStructuredMFVI
from utils.alignment import (
    align_temporal_states,
    compute_alignment_error,
    compute_correlation_after_alignment
)
from utils.diagnostics import (
    compute_reconstruction_error,
    compute_temporal_contributions,
    print_diagnostic_summary,
    compare_methods
)
from utils.metrics import (
    mean_squared_error,
    pearson_correlation,
    temporal_consistency_score
)
from visualization.temporal import (
    plot_state_trajectories,
    plot_trajectory_comparison,
    plot_temporal_contributions
)
from visualization.comparison import (
    plot_convergence_comparison,
    plot_method_comparison,
    plot_parameter_recovery_grid
)


def test_static_ame_model():
    """Test static AME model."""
    print("\n" + "="*70)
    print("TEST 1: Static AME Model")
    print("="*70)
    
    try:
        # Create model
        model = StaticAMEModel(
            n_nodes=10,
            latent_dim=2,
            rho_additive=0.5,
            rho_multiplicative=0.3,
            rho_dyadic=0.5,
            seed=42
        )
        print("✓ Model initialization successful")
        
        # Generate data
        Y, A, M = model.generate_data(return_latents=True)
        print(f"✓ Data generation successful")
        print(f"  Y shape: {Y.shape} (expected: (10, 10, 2))")
        print(f"  A shape: {A.shape} (expected: (10, 2))")
        print(f"  M shape: {M.shape} (expected: (10, 4))")
        
        # Verify shapes
        assert Y.shape == (10, 10, 2), f"Y shape mismatch: {Y.shape}"
        assert A.shape == (10, 2), f"A shape mismatch: {A.shape}"
        assert M.shape == (10, 4), f"M shape mismatch: {M.shape}"
        print("✓ All shapes correct")
        
        # Compute mean
        mu = model.compute_mean(A, M)
        print(f"✓ Mean computation successful, shape: {mu.shape}")
        
        print("\n✅ Static AME Model: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Static AME Model: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_ame_model():
    """Test temporal AME model."""
    print("\n" + "="*70)
    print("TEST 2: Temporal AME Model")
    print("="*70)
    
    try:
        # Create model
        model = TemporalAMEModel(
            n_nodes=10,
            n_time=5,
            latent_dim=2,
            ar_coefficient=0.8,
            rho_dyadic=0.5,
            process_noise_scale=0.1,
            seed=42
        )
        print("✓ Model initialization successful")
        print(f"  State dimension d: {model.d} (expected: 6 = 2 + 2*2)")
        
        # Generate data
        Y, X = model.generate_data(return_latents=True)
        print(f"✓ Data generation successful")
        print(f"  Y shape: {Y.shape} (expected: (10, 10, 5, 2))")
        print(f"  X shape: {X.shape} (expected: (10, 5, 6))")
        
        # Verify shapes
        assert Y.shape == (10, 10, 5, 2), f"Y shape mismatch: {Y.shape}"
        assert X.shape == (10, 5, 6), f"X shape mismatch: {X.shape}"
        print("✓ All shapes correct")
        
        # Test state extraction
        A_t, M_t = model.get_states_at_time(0)
        print(f"✓ State extraction successful")
        print(f"  A_0 shape: {A_t.shape} (expected: (10, 2))")
        print(f"  M_0 shape: {M_t.shape} (expected: (10, 4))")
        
        assert A_t.shape == (10, 2), f"A_t shape mismatch: {A_t.shape}"
        assert M_t.shape == (10, 4), f"M_t shape mismatch: {M_t.shape}"
        
        print("\n✅ Temporal AME Model: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Temporal AME Model: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_naive_mf_inference():
    """Test naive mean-field inference."""
    print("\n" + "="*70)
    print("TEST 3: Naive Mean-Field Inference")
    print("="*70)
    
    try:
        # Create model and generate data
        model = TemporalAMEModel(
            n_nodes=10,
            n_time=5,
            latent_dim=2,
            ar_coefficient=0.8,
            seed=42
        )
        Y, X_true = model.generate_data(return_latents=True)
        print("✓ Data generation successful")
        
        # Initialize inference
        vi = TemporalAMENaiveMFVI(
            model,
            learning_rate=1.0,
            init_scale=0.1,
            seed=42
        )
        print("✓ Inference initialization successful")
        print(f"  X_mean shape: {vi.X_mean.shape} (expected: (10, 5, 6))")
        print(f"  X_cov shape: {vi.X_cov.shape} (expected: (10, 5, 6, 6))")
        
        # Run inference
        print("\nRunning inference (10 iterations)...")
        history = vi.fit(max_iter=10, verbose=True)
        print(f"✓ Inference completed")
        print(f"  Final ELBO: {history['elbo'][-1]:.2f}")
        print(f"  Final MSE: {history['reconstruction_error'][-1]:.6f}")
        
        # Verify convergence
        assert len(history['elbo']) == 10, "Wrong number of iterations"
        assert history['elbo'][-1] > history['elbo'][0], "ELBO should increase"
        print("✓ ELBO increased as expected")
        
        print("\n✅ Naive MF Inference: PASSED\n")
        return True, model, X_true, vi
        
    except Exception as e:
        print(f"\n❌ Naive MF Inference: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def test_structured_mf_inference():
    """Test structured mean-field inference."""
    print("\n" + "="*70)
    print("TEST 4: Structured Mean-Field Inference")
    print("="*70)
    
    try:
        # Create model and generate data
        model = TemporalAMEModel(
            n_nodes=10,
            n_time=5,
            latent_dim=2,
            ar_coefficient=0.8,
            seed=42
        )
        Y, X_true = model.generate_data(return_latents=True)
        print("✓ Data generation successful")
        
        # Test both factorizations
        for factorization in ['good', 'bad']:
            print(f"\nTesting '{factorization}' factorization...")
            
            vi = TemporalAMEStructuredMFVI(
                model,
                factorization=factorization,
                learning_rate=1.0,
                seed=42
            )
            print(f"  ✓ Inference initialization successful")
            
            # Run inference
            history = vi.fit(max_iter=10, verbose=False)
            print(f"  ✓ Inference completed")
            print(f"    Final ELBO: {history['elbo'][-1]:.2f}")
            print(f"    Final MSE: {history['reconstruction_error'][-1]:.6f}")
            
            # Verify convergence
            assert len(history['elbo']) == 10, "Wrong number of iterations"
            print(f"  ✓ {factorization.capitalize()} factorization works")
        
        print("\n✅ Structured MF Inference: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Structured MF Inference: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alignment_utils():
    """Test alignment utilities."""
    print("\n" + "="*70)
    print("TEST 5: Alignment Utilities")
    print("="*70)
    
    try:
        # Create random test data
        X_true = torch.randn(10, 5, 6)
        X_est = torch.randn(10, 5, 6)
        
        # Test temporal alignment
        X_aligned = align_temporal_states(X_est, X_true, latent_dim=2)
        print("✓ Temporal state alignment successful")
        print(f"  Aligned shape: {X_aligned.shape}")
        assert X_aligned.shape == X_true.shape
        
        # Test alignment error
        error, X_aligned = compute_alignment_error(
            X_est, X_true, latent_dim=2, align=True
        )
        print(f"✓ Alignment error computation successful: {error:.6f}")
        
        # Test correlation after alignment
        corr = compute_correlation_after_alignment(X_est, X_true, latent_dim=2)
        print(f"✓ Correlation after alignment: {corr:.3f}")
        
        print("\n✅ Alignment Utilities: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Alignment Utilities: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diagnostic_utils():
    """Test diagnostic utilities."""
    print("\n" + "="*70)
    print("TEST 6: Diagnostic Utilities")
    print("="*70)
    
    try:
        # Create test data
        X = torch.randn(10, 5, 6)
        Y_true = torch.randn(10, 10, 5, 2)
        Y_pred = torch.randn(10, 10, 5, 2)
        
        # Test reconstruction error
        recon_error = compute_reconstruction_error(Y_true, Y_pred)
        print(f"✓ Reconstruction error: {recon_error:.6f}")
        
        # Test temporal contributions
        add_contrib, mult_contrib = compute_temporal_contributions(X, latent_dim=2)
        print(f"✓ Temporal contributions computed")
        print(f"  Additive shape: {add_contrib.shape} (expected: (5,))")
        print(f"  Multiplicative shape: {mult_contrib.shape} (expected: (5,))")
        
        assert add_contrib.shape == (5,)
        assert mult_contrib.shape == (5,)
        
        print("\n✅ Diagnostic Utilities: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Diagnostic Utilities: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metric_utils():
    """Test metric utilities."""
    print("\n" + "="*70)
    print("TEST 7: Metric Utilities")
    print("="*70)
    
    try:
        # Create test data
        y_true = torch.randn(100)
        y_pred = torch.randn(100)
        X = torch.randn(10, 5, 6)
        
        # Test MSE
        mse = mean_squared_error(y_true, y_pred)
        print(f"✓ MSE: {mse:.6f}")
        
        # Test correlation
        corr = pearson_correlation(y_true, y_pred)
        print(f"✓ Pearson correlation: {corr:.3f}")
        
        # Test temporal consistency
        consistency = temporal_consistency_score(X, order=1)
        print(f"✓ Temporal consistency score: {consistency:.6f}")
        
        print("\n✅ Metric Utilities: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Metric Utilities: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizations():
    """Test visualization functions."""
    print("\n" + "="*70)
    print("TEST 8: Visualization Functions")
    print("="*70)
    
    try:
        # Create test data
        X = torch.randn(10, 5, 6)
        X_true = torch.randn(10, 5, 6)
        X_est = torch.randn(10, 5, 6)
        add_contrib = torch.randn(5).abs()
        mult_contrib = torch.randn(5).abs()
        
        # Test state trajectory plot
        fig = plot_state_trajectories(X, node_indices=[0, 1])
        plt.close(fig)
        print("✓ State trajectory plot created")
        
        # Test trajectory comparison
        fig = plot_trajectory_comparison(X_true, X_est, node_index=0)
        plt.close(fig)
        print("✓ Trajectory comparison plot created")
        
        # Test temporal contributions plot
        fig = plot_temporal_contributions(add_contrib, mult_contrib)
        plt.close(fig)
        print("✓ Temporal contributions plot created")
        
        # Test method comparison (requires results dict)
        history1 = {'elbo': list(range(10)), 'reconstruction_error': [0.1] * 10}
        history2 = {'elbo': list(range(10)), 'reconstruction_error': [0.15] * 10}
        results = {
            'Method 1': {'history': history1, 'X_est': X_est},
            'Method 2': {'history': history2, 'X_est': X_est}
        }
        
        fig = plot_convergence_comparison(results, metric='elbo')
        plt.close(fig)
        print("✓ Convergence comparison plot created")
        
        fig = plot_method_comparison(results, metrics=['reconstruction_error'])
        plt.close(fig)
        print("✓ Method comparison plot created")
        
        fig = plot_parameter_recovery_grid(X_true, results)
        plt.close(fig)
        print("✓ Parameter recovery grid created")
        
        print("\n✅ Visualization Functions: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Visualization Functions: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("\n" + "="*70)
    print("TEST 9: End-to-End Pipeline")
    print("="*70)
    
    try:
        # 1. Generate data
        print("\nStep 1: Generating data...")
        model = TemporalAMEModel(
            n_nodes=15,
            n_time=8,
            latent_dim=2,
            ar_coefficient=0.8,
            rho_dyadic=0.5,
            process_noise_scale=0.1,
            seed=42
        )
        Y, X_true = model.generate_data(return_latents=True)
        print(f"  ✓ Data generated: Y{Y.shape}, X{X_true.shape}")
        
        # 2. Run multiple inference methods
        print("\nStep 2: Running inference methods...")
        results = {}
        
        # Naive MF
        print("  Running Naive MF...")
        vi_naive = TemporalAMENaiveMFVI(model, learning_rate=1.0, seed=42)
        start_time = time.time()
        history_naive = vi_naive.fit(max_iter=20, verbose=False)
        runtime_naive = time.time() - start_time
        results['Naive MF'] = {
            'vi': vi_naive,
            'history': history_naive,
            'X_est': vi_naive.X_mean.detach(),
            'runtime': runtime_naive,
            'iterations': len(history_naive['elbo'])
        }
        print(f"    ✓ Completed in {runtime_naive:.2f}s")
        
        # Structured MF (Good)
        print("  Running Structured MF (Good)...")
        vi_smf_good = TemporalAMEStructuredMFVI(
            model, factorization='good', learning_rate=1.0, seed=42
        )
        start_time = time.time()
        history_smf_good = vi_smf_good.fit(max_iter=20, verbose=False)
        runtime_smf_good = time.time() - start_time
        results['Good SMF'] = {
            'vi': vi_smf_good,
            'history': history_smf_good,
            'X_est': vi_smf_good.X_mean.detach(),
            'runtime': runtime_smf_good,
            'iterations': len(history_smf_good['elbo'])
        }
        print(f"    ✓ Completed in {runtime_smf_good:.2f}s")
        
        # Structured MF (Bad)
        print("  Running Structured MF (Bad)...")
        vi_smf_bad = TemporalAMEStructuredMFVI(
            model, factorization='bad', learning_rate=1.0, seed=42
        )
        start_time = time.time()
        history_smf_bad = vi_smf_bad.fit(max_iter=20, verbose=False)
        runtime_smf_bad = time.time() - start_time
        results['Bad SMF'] = {
            'vi': vi_smf_bad,
            'history': history_smf_bad,
            'X_est': vi_smf_bad.X_mean.detach(),
            'runtime': runtime_smf_bad,
            'iterations': len(history_smf_bad['elbo'])
        }
        print(f"    ✓ Completed in {runtime_smf_bad:.2f}s")
        
        # 3. Compare methods
        print("\nStep 3: Comparing methods...")
        compare_methods(results, metric='reconstruction_error', X_true=X_true)
        
        # 4. Compute alignment errors
        print("\nStep 4: Computing alignment errors...")
        for method_name, result in results.items():
            error, _ = compute_alignment_error(
                result['X_est'], X_true, latent_dim=2, align=True
            )
            print(f"  {method_name:15s}: Alignment MSE = {error:.6f}")
        
        # 5. Generate visualizations
        print("\nStep 5: Generating visualizations...")
        
        # Create output directory
        output_dir = Path('test_outputs')
        output_dir.mkdir(exist_ok=True)
        
        # Convergence comparison
        fig = plot_convergence_comparison(
            results,
            metric='elbo',
            save_path=output_dir / 'convergence.png'
        )
        plt.close(fig)
        print(f"  ✓ Saved: {output_dir / 'convergence.png'}")
        
        # Method comparison
        fig = plot_method_comparison(
            results,
            metrics=['reconstruction_error'],
            save_path=output_dir / 'method_comparison.png'
        )
        plt.close(fig)
        print(f"  ✓ Saved: {output_dir / 'method_comparison.png'}")
        
        # Parameter recovery
        fig = plot_parameter_recovery_grid(
            X_true,
            results,
            save_path=output_dir / 'parameter_recovery.png'
        )
        plt.close(fig)
        print(f"  ✓ Saved: {output_dir / 'parameter_recovery.png'}")
        
        # State trajectories
        fig = plot_state_trajectories(
            X_true,
            node_indices=[0, 1, 2],
            save_path=output_dir / 'trajectories.png'
        )
        plt.close(fig)
        print(f"  ✓ Saved: {output_dir / 'trajectories.png'}")
        
        print("\n✅ End-to-End Pipeline: PASSED\n")
        print(f"All visualizations saved to: {output_dir}/")
        return True
        
    except Exception as e:
        print(f"\n❌ End-to-End Pipeline: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("TEMPORAL AME INTEGRATION TESTS")
    print("="*70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    results = {}
    
    # Run tests
    results['Static Model'] = test_static_ame_model()
    results['Temporal Model'] = test_temporal_ame_model()
    naive_result, model, X_true, vi = test_naive_mf_inference()
    results['Naive MF'] = naive_result
    results['Structured MF'] = test_structured_mf_inference()
    results['Alignment Utils'] = test_alignment_utils()
    results['Diagnostic Utils'] = test_diagnostic_utils()
    results['Metric Utils'] = test_metric_utils()
    results['Visualizations'] = test_visualizations()
    results['End-to-End'] = test_end_to_end_pipeline()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Your refactored code is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Please review the errors above.")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)