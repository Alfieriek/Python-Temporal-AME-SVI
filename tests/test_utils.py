"""
Tests for utility functions.

Tests diagnostics, alignment, and metrics modules.

Author: Sean Plummer
Date: October 2025
"""

import pytest
import torch
import numpy as np

from src.utils import (
    # Diagnostics
    compute_reconstruction_error,
    compute_additive_contribution,
    compute_multiplicative_contribution,
    compute_temporal_contributions,
    compute_contribution_ratio,
    compute_state_prediction_error,
    print_diagnostic_summary,
    compare_methods,
    # Alignment
    procrustes_alignment,
    align_signs,
    align_latent_positions,
    align_temporal_states,
    compute_alignment_error,
    compute_correlation_after_alignment,
    # Metrics
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared,
    pearson_correlation,
    temporal_consistency_score,
    link_prediction_metrics,
    calibration_error,
    compute_coverage,
    relative_error
)


class TestDiagnostics:
    """Tests for diagnostic functions."""
    
    def test_compute_reconstruction_error_static(self):
        """Test reconstruction error for static data."""
        Y_true = torch.randn(10, 10, 2)
        Y_pred = Y_true.clone()
        
        # Perfect prediction
        error = compute_reconstruction_error(Y_true, Y_pred)
        assert error < 1e-6
        
        # Noisy prediction
        Y_pred_noisy = Y_true + torch.randn_like(Y_true) * 0.5
        error_noisy = compute_reconstruction_error(Y_true, Y_pred_noisy)
        assert error_noisy > 0.1
    
    def test_compute_reconstruction_error_temporal(self):
        """Test reconstruction error for temporal data."""
        Y_true = torch.randn(10, 10, 5, 2)
        Y_pred = Y_true.clone()
        
        error = compute_reconstruction_error(Y_true, Y_pred)
        assert error < 1e-6
    
    def test_compute_additive_contribution(self):
        """Test additive contribution computation."""
        A = torch.randn(10, 2)
        contrib = compute_additive_contribution(A)
        
        assert isinstance(contrib, float)
        assert contrib >= 0
    
    def test_compute_multiplicative_contribution(self):
        """Test multiplicative contribution computation."""
        M = torch.randn(10, 4)  # r=2
        contrib = compute_multiplicative_contribution(M)
        
        assert isinstance(contrib, float)
        assert contrib >= 0
    
    def test_compute_temporal_contributions(self):
        """Test temporal contributions computation."""
        X = torch.randn(10, 5, 6)  # n=10, T=5, d=6 (r=2)
        
        add_contrib, mult_contrib = compute_temporal_contributions(X, latent_dim=2)
        
        assert add_contrib.shape == (5,)
        assert mult_contrib.shape == (5,)
        assert torch.all(add_contrib >= 0)
        assert torch.all(mult_contrib >= 0)
    
    def test_compute_contribution_ratio(self):
        """Test contribution ratio computation."""
        A = torch.randn(10, 2)
        M = torch.randn(10, 4)
        
        ratio = compute_contribution_ratio(A, M)
        
        assert isinstance(ratio, float)
        assert ratio >= 0
    
    def test_compute_state_prediction_error(self, sample_trajectories):
        """Test state prediction error."""
        X_true, X_est = sample_trajectories
        
        error = compute_state_prediction_error(X_true, X_est)
        
        assert isinstance(error, float)
        assert error >= 0
    
    def test_print_diagnostic_summary(self, mock_history, capsys):
        """Test diagnostic summary printing."""
        X_true = torch.randn(10, 5, 6)
        X_est = X_true + torch.randn_like(X_true) * 0.1
        
        print_diagnostic_summary(
            "Test Method",
            mock_history,
            X_true=X_true,
            X_est=X_est,
            latent_dim=2,
            final_only=True
        )
        
        captured = capsys.readouterr()
        assert "Test Method" in captured.out
        assert "MSE" in captured.out
    
    def test_compare_methods(self, capsys):
        """Test method comparison."""
        results = {
            'Method1': {
                'history': {'reconstruction_error': [0.5, 0.4, 0.3]},
                'X_est': torch.randn(10, 5, 6)
            },
            'Method2': {
                'history': {'reconstruction_error': [0.6, 0.5, 0.35]},
                'X_est': torch.randn(10, 5, 6)
            }
        }
        
        compare_methods(results, metric='reconstruction_error')
        
        captured = capsys.readouterr()
        assert "Method1" in captured.out
        assert "Method2" in captured.out


class TestAlignment:
    """Tests for alignment functions."""
    
    def test_procrustes_alignment(self):
        """Test Procrustes alignment."""
        X_true = torch.randn(20, 3)
        
        # Create rotated version
        angle = np.pi / 4
        R_rot = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        X_est = torch.matmul(X_true, R_rot.t())
        
        # Align
        X_aligned, R = procrustes_alignment(X_est, X_true)
        
        # Should recover X_true
        assert torch.allclose(X_aligned, X_true, atol=1e-5)
    
    def test_align_signs(self):
        """Test sign alignment."""
        X_true = torch.randn(20, 3)
        X_est = -X_true  # Flip all signs
        
        X_aligned = align_signs(X_est, X_true, dim=1)
        
        # Should recover X_true
        assert torch.allclose(X_aligned, X_true, atol=1e-6)
    
    def test_align_latent_positions(self):
        """Test latent position alignment."""
        M_true = torch.randn(20, 4)  # r=2
        
        # Create misaligned version with rotation
        M_est = M_true + torch.randn_like(M_true) * 0.1
        
        M_aligned = align_latent_positions(M_est, M_true, latent_dim=2)
        
        # Should be closer to true after alignment
        error_before = ((M_est - M_true) ** 2).mean()
        error_after = ((M_aligned - M_true) ** 2).mean()
        
        assert error_after <= error_before
    
    def test_align_temporal_states(self, sample_trajectories):
        """Test temporal state alignment."""
        X_true, X_est = sample_trajectories
        
        X_aligned = align_temporal_states(X_est, X_true, latent_dim=2)
        
        # Should reduce error
        error_before = ((X_est - X_true) ** 2).mean()
        error_after = ((X_aligned - X_true) ** 2).mean()
        
        assert error_after <= error_before
    
    def test_compute_alignment_error(self, sample_trajectories):
        """Test alignment error computation."""
        X_true, X_est = sample_trajectories
        
        # With alignment
        error_aligned, X_aligned = compute_alignment_error(
            X_est, X_true, latent_dim=2, align=True
        )
        
        # Without alignment
        error_unaligned, _ = compute_alignment_error(
            X_est, X_true, latent_dim=2, align=False
        )
        
        # Aligned should be better or equal
        assert error_aligned <= error_unaligned + 1e-5
    
    def test_compute_correlation_after_alignment(self, sample_trajectories):
        """Test correlation computation after alignment."""
        X_true, X_est = sample_trajectories
        
        corr = compute_correlation_after_alignment(X_est, X_true, latent_dim=2)
        
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0


class TestMetrics:
    """Tests for metric functions."""
    
    def test_mean_squared_error(self):
        """Test MSE computation."""
        y_true = torch.randn(100)
        y_pred = y_true.clone()
        
        mse = mean_squared_error(y_true, y_pred)
        assert mse < 1e-6
        
        y_pred_noisy = y_true + torch.randn_like(y_true) * 0.5
        mse_noisy = mean_squared_error(y_true, y_pred_noisy)
        assert mse_noisy > 0.1
    
    def test_root_mean_squared_error(self):
        """Test RMSE computation."""
        y_true = torch.randn(100)
        y_pred = y_true + torch.randn_like(y_true) * 0.3
        
        rmse = root_mean_squared_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        assert np.isclose(rmse, np.sqrt(mse))
    
    def test_mean_absolute_error(self):
        """Test MAE computation."""
        y_true = torch.ones(100)
        y_pred = torch.ones(100) * 1.5
        
        mae = mean_absolute_error(y_true, y_pred)
        assert np.isclose(mae, 0.5)
    
    def test_r_squared(self):
        """Test R² computation."""
        y_true = torch.randn(100)
        
        # Perfect prediction
        r2_perfect = r_squared(y_true, y_true)
        assert np.isclose(r2_perfect, 1.0)
        
        # Random prediction
        y_pred_random = torch.randn(100)
        r2_random = r_squared(y_true, y_pred_random)
        assert r2_random < 1.0
    
    def test_pearson_correlation(self):
        """Test Pearson correlation."""
        y_true = torch.randn(100)
        
        # Perfect correlation
        corr_perfect = pearson_correlation(y_true, y_true)
        assert np.isclose(corr_perfect, 1.0)
        
        # Anti-correlation
        corr_neg = pearson_correlation(y_true, -y_true)
        assert np.isclose(corr_neg, -1.0)
    
    def test_temporal_consistency_score(self):
        """Test temporal consistency computation."""
        # Smooth trajectory
        X_smooth = torch.linspace(0, 10, 100).unsqueeze(0).unsqueeze(-1)
        X_smooth = X_smooth.expand(5, 100, 3)
        
        consistency_smooth = temporal_consistency_score(X_smooth, order=1)
        
        # Noisy trajectory
        X_noisy = X_smooth + torch.randn_like(X_smooth) * 5.0
        consistency_noisy = temporal_consistency_score(X_noisy, order=1)
        
        # Noisy should have higher (worse) consistency score
        assert consistency_noisy > consistency_smooth
    
    def test_link_prediction_metrics(self):
        """Test link prediction metrics."""
        Y_true = torch.randn(20, 20)
        Y_pred = Y_true + torch.randn_like(Y_true) * 0.3
        
        metrics = link_prediction_metrics(Y_true, Y_pred, threshold=0.0)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # All metrics should be in [0, 1]
        for value in metrics.values():
            assert 0.0 <= value <= 1.0
    
    def test_calibration_error(self):
        """Test calibration error computation."""
        predictions = torch.randn(100)
        targets = predictions + torch.randn(100) * 0.5
        uncertainties = torch.ones(100) * 0.5
        
        ece = calibration_error(predictions, uncertainties, targets)
        
        assert isinstance(ece, float)
        assert ece >= 0
    
    def test_compute_coverage(self):
        """Test coverage computation."""
        preds = torch.zeros(100)
        targets = torch.randn(100) * 0.5
        lower = preds - 1.0
        upper = preds + 1.0
        
        coverage = compute_coverage(preds, lower, upper, targets)
        
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0
    
    def test_relative_error(self):
        """Test relative error computation."""
        y_true = torch.ones(100)
        y_pred = torch.ones(100) * 1.1
        
        rel_err = relative_error(y_true, y_pred)
        
        assert isinstance(rel_err, float)
        assert rel_err >= 0
    
    def test_masked_metrics(self):
        """Test that masking works for metrics."""
        y_true = torch.randn(10, 10)
        y_pred = torch.randn(10, 10)
        mask = torch.eye(10) == 0  # Exclude diagonal
        
        mse = mean_squared_error(y_true, y_pred, mask)
        mae = mean_absolute_error(y_true, y_pred, mask)
        
        assert isinstance(mse, float)
        assert isinstance(mae, float)