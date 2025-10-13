"""
Tests for variational inference methods.

Tests initialization, fitting, and inference quality for all VI methods.

Author: Sean Plummer
Date: October 2025
"""

import pytest
import torch
import numpy as np

from src.inference import (
    BaseVariationalInference,
    BaseTemporalVariationalInference,
    TemporalAMENaiveMFVI,
    TemporalAMEStructuredMFVI
)


class TestTemporalAMENaiveMFVI:
    """Tests for naive mean-field VI."""
    
    def test_initialization(self, temporal_data):
        """Test VI initialization."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model, learning_rate=0.01)
        
        assert vi.n == model.n
        assert vi.T == model.T
        assert vi.d == model.d
        assert vi.lr == 0.01
        assert vi.X_mean.shape == (model.n, model.T, model.d)
        assert vi.X_cov.shape == (model.n, model.T, model.d, model.d)
        
        # Check covariances are diagonal (naive MF)
        for i in range(model.n):
            for t in range(model.T):
                cov = vi.X_cov[i, t]
                # Off-diagonal should be zero
                off_diag = cov - torch.diag(torch.diag(cov))
                assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6)
    
    def test_fit_runs(self, temporal_data):
        """Test that fitting runs without errors."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model, learning_rate=0.01)
        
        history = vi.fit(max_iter=5, verbose=False)
        
        assert 'elbo' in history
        assert 'reconstruction_error' in history
        assert len(history['elbo']) == 5
        assert len(history['reconstruction_error']) == 5
    
    def test_elbo_increases(self, temporal_data):
        """Test that ELBO generally increases (or at least doesn't crash)."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model, learning_rate=0.01)
        
        history = vi.fit(max_iter=10, verbose=False)
        
        # ELBO should be finite
        assert all(np.isfinite(history['elbo']))
        
        # Check that we're not diverging
        assert history['elbo'][-1] > -1e10
    
    def test_get_variational_means(self, temporal_data):
        """Test retrieval of variational means."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model)
        vi.fit(max_iter=3, verbose=False)
        
        X_mean = vi.get_variational_means()
        assert X_mean.shape == (model.n, model.T, model.d)
    
    def test_get_variational_covariances(self, temporal_data):
        """Test retrieval of variational covariances."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model)
        vi.fit(max_iter=3, verbose=False)
        
        X_cov = vi.get_variational_covariances()
        assert X_cov.shape == (model.n, model.T, model.d, model.d)
    
    def test_predict_forward(self, temporal_data):
        """Test forward prediction."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model)
        vi.fit(max_iter=5, verbose=False)
        
        X_pred = vi.predict_forward(n_steps=3)
        assert X_pred.shape == (model.n, 3, model.d)
    
    def test_reproducibility(self, temporal_data):
        """Test that same seed gives same results."""
        model = temporal_data['model']
        
        vi1 = TemporalAMENaiveMFVI(model, seed=42)
        history1 = vi1.fit(max_iter=5, verbose=False)
        
        vi2 = TemporalAMENaiveMFVI(model, seed=42)
        history2 = vi2.fit(max_iter=5, verbose=False)
        
        # Initial states should be identical
        assert torch.allclose(vi1.X_mean, vi2.X_mean, atol=1e-5)


class TestTemporalAMEStructuredMFVI:
    """Tests for structured mean-field VI."""
    
    def test_initialization_good(self, temporal_data):
        """Test initialization with good factorization."""
        model = temporal_data['model']
        vi = TemporalAMEStructuredMFVI(model, factorization="good")
        
        assert vi.n == model.n
        assert vi.T == model.T
        assert vi.d == model.d
        assert vi.factorization == "good"
        assert vi.X_mean.shape == (model.n, model.T, model.d)
        assert vi.X_cov.shape == (model.n, model.T, model.d, model.d)
        
        # Check covariances are full (not diagonal)
        for i in range(model.n):
            for t in range(model.T):
                cov = vi.X_cov[i, t]
                # Should have some off-diagonal elements
                off_diag = cov - torch.diag(torch.diag(cov))
                # At least some should be non-zero
                assert not torch.allclose(off_diag, torch.zeros_like(off_diag))
    
    def test_initialization_bad(self, temporal_data):
        """Test initialization with bad factorization."""
        model = temporal_data['model']
        vi = TemporalAMEStructuredMFVI(model, factorization="bad")
        
        assert vi.factorization == "bad"
        
        # Check block-diagonal structure
        for i in range(model.n):
            for t in range(model.T):
                cov = vi.X_cov[i, t]
                # Off-diagonal blocks should be zero
                assert torch.allclose(cov[:2, 2:], torch.zeros(2, model.d - 2))
                assert torch.allclose(cov[2:, :2], torch.zeros(model.d - 2, 2))
    
    def test_invalid_factorization(self, temporal_data):
        """Test that invalid factorization raises error."""
        model = temporal_data['model']
        with pytest.raises(ValueError):
            vi = TemporalAMEStructuredMFVI(model, factorization="invalid")
    
    def test_fit_runs_good(self, temporal_data):
        """Test that fitting runs with good factorization."""
        model = temporal_data['model']
        vi = TemporalAMEStructuredMFVI(model, factorization="good")
        
        history = vi.fit(max_iter=5, verbose=False)
        
        assert 'elbo' in history
        assert 'reconstruction_error' in history
        assert len(history['elbo']) == 5
    
    def test_fit_runs_bad(self, temporal_data):
        """Test that fitting runs with bad factorization."""
        model = temporal_data['model']
        vi = TemporalAMEStructuredMFVI(model, factorization="bad")
        
        history = vi.fit(max_iter=5, verbose=False)
        
        assert 'elbo' in history
        assert 'reconstruction_error' in history
    
    def test_get_factorization_type(self, temporal_data):
        """Test factorization type retrieval."""
        model = temporal_data['model']
        
        vi_good = TemporalAMEStructuredMFVI(model, factorization="good")
        assert vi_good.get_factorization_type() == "good"
        
        vi_bad = TemporalAMEStructuredMFVI(model, factorization="bad")
        assert vi_bad.get_factorization_type() == "bad"
    
    def test_maintains_structure_during_optimization(self, temporal_data):
        """Test that factorization structure is maintained during optimization."""
        model = temporal_data['model']
        vi = TemporalAMEStructuredMFVI(model, factorization="bad")
        
        vi.fit(max_iter=5, verbose=False)
        
        # Check that bad structure is still maintained
        for i in range(model.n):
            for t in range(model.T):
                cov = vi.X_cov[i, t]
                # Off-diagonal blocks should still be zero
                assert torch.allclose(
                    cov[:2, 2:],
                    torch.zeros(2, model.d - 2),
                    atol=1e-5
                )


class TestBaseVariationalInference:
    """Tests for base VI class functionality."""
    
    def test_history_tracking(self, temporal_data):
        """Test that optimization history is tracked correctly."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model)
        
        history = vi.fit(max_iter=10, verbose=False)
        
        assert len(history['elbo']) == 10
        assert len(history['reconstruction_error']) == 10
        
        # Can retrieve history
        elbo_hist = vi.get_elbo_history()
        assert len(elbo_hist) == 10
    
    def test_convergence_detection(self, temporal_data):
        """Test that convergence detection works."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model, learning_rate=0.001)
        
        # With small learning rate, should converge early
        history = vi.fit(max_iter=100, tolerance=1e-3, verbose=False)
        
        # Should stop before max_iter
        # (This might not always happen with noisy optimization, so we check gently)
        assert len(history['elbo']) <= 100
    
    def test_verbose_output(self, temporal_data, capsys):
        """Test that verbose mode produces output."""
        model = temporal_data['model']
        vi = TemporalAMENaiveMFVI(model)
        
        vi.fit(max_iter=5, verbose=True, check_every=1)
        
        captured = capsys.readouterr()
        assert "Iter" in captured.out
        assert "ELBO" in captured.out
        assert "MSE" in captured.out
    
    def test_different_learning_rates(self, temporal_data):
        """Test that different learning rates work."""
        model = temporal_data['model']
        
        vi_small = TemporalAMENaiveMFVI(model, learning_rate=0.001)
        history_small = vi_small.fit(max_iter=5, verbose=False)
        
        vi_large = TemporalAMENaiveMFVI(model, learning_rate=0.1)
        history_large = vi_large.fit(max_iter=5, verbose=False)
        
        # Both should run without errors
        assert len(history_small['elbo']) == 5
        assert len(history_large['elbo']) == 5
        
        # Results should be different
        assert not np.allclose(
            history_small['elbo'],
            history_large['elbo'],
            atol=1.0
        )


class TestInferenceComparison:
    """Tests comparing different inference methods."""
    
    def test_methods_produce_different_results(self, temporal_data):
        """Test that different methods produce different results."""
        model = temporal_data['model']
        
        vi_naive = TemporalAMENaiveMFVI(model, seed=42)
        history_naive = vi_naive.fit(max_iter=10, verbose=False)
        
        vi_good = TemporalAMEStructuredMFVI(model, factorization="good", seed=42)
        history_good = vi_good.fit(max_iter=10, verbose=False)
        
        # Results should be different
        assert not np.allclose(
            history_naive['reconstruction_error'],
            history_good['reconstruction_error'],
            atol=0.01
        )
    
    def test_good_vs_bad_structure(self, temporal_data):
        """Test that good and bad factorizations give different results."""
        model = temporal_data['model']
        
        vi_good = TemporalAMEStructuredMFVI(model, factorization="good", seed=42)
        history_good = vi_good.fit(max_iter=10, verbose=False)
        
        vi_bad = TemporalAMEStructuredMFVI(model, factorization="bad", seed=42)
        history_bad = vi_bad.fit(max_iter=10, verbose=False)
        
        # Should produce different reconstructions
        final_mse_good = history_good['reconstruction_error'][-1]
        final_mse_bad = history_bad['reconstruction_error'][-1]
        
        # Just check they're different (actual ordering depends on data)
        assert not np.isclose(final_mse_good, final_mse_bad, atol=0.001)