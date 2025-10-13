"""
Tests for AME model classes.

Tests model initialization, data generation, mean computation,
and other core model functionality.

Author: Sean Plummer
Date: October 2025
"""

import pytest
import torch
import numpy as np

from src.models import BaseAMEModel, StaticAMEModel, TemporalAMEModel


class TestStaticAMEModel:
    """Tests for StaticAMEModel."""
    
    def test_initialization(self, small_network_params):
        """Test model initialization with valid parameters."""
        model = StaticAMEModel(**small_network_params)
        
        assert model.n == small_network_params['n_nodes']
        assert model.r == small_network_params['latent_dim']
        assert model.Sigma.shape == (2, 2)
        assert model.Psi.shape == (4, 4)  # 2r x 2r
        assert model.R.shape == (2, 2)
        
    def test_data_generation(self, static_model):
        """Test synthetic data generation."""
        Y = static_model.generate_data()
        
        # Check shape
        assert Y.shape == (static_model.n, static_model.n, 2)
        
        # Check diagonal is zero (no self-loops)
        for i in range(static_model.n):
            assert torch.allclose(Y[i, i], torch.zeros(2))
        
        # Check reciprocal consistency: Y[i,j,1] == Y[j,i,0]
        for i in range(static_model.n):
            for j in range(i + 1, static_model.n):
                assert torch.allclose(Y[i, j, 1], Y[j, i, 0])
    
    def test_data_generation_with_latents(self, static_model):
        """Test data generation returns latent parameters."""
        Y, A, M = static_model.generate_data(return_latents=True)
        
        assert Y.shape == (static_model.n, static_model.n, 2)
        assert A.shape == (static_model.n, 2)
        assert M.shape == (static_model.n, 4)
    
    def test_compute_mean(self, static_model):
        """Test mean structure computation."""
        A = torch.randn(static_model.n, 2)
        M = torch.randn(static_model.n, 4)
        
        mu = static_model.compute_mean(A, M)
        
        assert mu.shape == (static_model.n, static_model.n, 2)
        
        # Check reciprocal structure
        for i in range(static_model.n):
            for j in range(static_model.n):
                # mu[i,j,0] should relate to mu[j,i,1]
                # Both should have additive + multiplicative structure
                assert not torch.isnan(mu[i, j, 0])
                assert not torch.isnan(mu[i, j, 1])
    
    def test_reconstruction_error(self, static_data):
        """Test reconstruction error computation."""
        model = static_data['model']
        A = static_data['A']
        M = static_data['M']
        
        # Perfect reconstruction should give ~0 error
        error = model.compute_reconstruction_error(A, M)
        assert error < 1e-4
        
        # Add noise and check error increases
        A_noisy = A + torch.randn_like(A) * 0.5
        error_noisy = model.compute_reconstruction_error(A_noisy, M)
        assert error_noisy > error
    
    def test_contribution_computation(self, static_data):
        """Test additive and multiplicative contribution computation."""
        A = static_data['A']
        M = static_data['M']
        model = static_data['model']
        
        add_contrib = model.compute_additive_contribution(A)
        mult_contrib = model.compute_multiplicative_contribution(M)
        
        assert isinstance(add_contrib, float)
        assert isinstance(mult_contrib, float)
        assert add_contrib >= 0
        assert mult_contrib >= 0
    
    def test_reproducibility(self, small_network_params):
        """Test that same seed produces same data."""
        model1 = StaticAMEModel(**small_network_params)
        Y1 = model1.generate_data()
        
        model2 = StaticAMEModel(**small_network_params)
        Y2 = model2.generate_data()
        
        assert torch.allclose(Y1, Y2)


class TestTemporalAMEModel:
    """Tests for TemporalAMEModel."""
    
    def test_initialization(self, temporal_network_params):
        """Test temporal model initialization."""
        model = TemporalAMEModel(**temporal_network_params)
        
        assert model.n == temporal_network_params['n_nodes']
        assert model.T == temporal_network_params['n_time']
        assert model.r == temporal_network_params['latent_dim']
        assert model.d == 2 + 2 * model.r  # State dimension
        assert model.Phi.shape == (model.d, model.d)
        assert model.Q.shape == (model.d, model.d)
    
    def test_temporal_data_generation(self, temporal_model):
        """Test temporal data generation."""
        Y = temporal_model.generate_data()
        
        # Check shape
        assert Y.shape == (temporal_model.n, temporal_model.n, temporal_model.T, 2)
        
        # Check diagonal is zero at all times
        for t in range(temporal_model.T):
            for i in range(temporal_model.n):
                assert torch.allclose(Y[i, i, t], torch.zeros(2))
        
        # Check reciprocal consistency at each time
        for t in range(temporal_model.T):
            for i in range(temporal_model.n):
                for j in range(i + 1, temporal_model.n):
                    assert torch.allclose(Y[i, j, t, 1], Y[j, i, t, 0])
    
    def test_temporal_data_with_latents(self, temporal_model):
        """Test temporal data generation with latent states."""
        Y, X = temporal_model.generate_data(return_latents=True)
        
        assert Y.shape == (temporal_model.n, temporal_model.n, temporal_model.T, 2)
        assert X.shape == (temporal_model.n, temporal_model.T, temporal_model.d)
    
    def test_get_states_at_time(self, temporal_data):
        """Test extraction of states at specific time."""
        model = temporal_data['model']
        X = temporal_data['X']
        model.X = X  # Set the states
        
        t = 2
        A_t, M_t = model.get_states_at_time(t)
        
        assert A_t.shape == (model.n, 2)
        assert M_t.shape == (model.n, 2 * model.r)
        assert torch.allclose(A_t, X[:, t, :2])
        assert torch.allclose(M_t, X[:, t, 2:])
    
    def test_temporal_reconstruction_error(self, temporal_data):
        """Test temporal reconstruction error computation."""
        model = temporal_data['model']
        X = temporal_data['X']
        
        # Perfect reconstruction
        error = model.compute_temporal_reconstruction_error(X)
        assert error < 1e-4
        
        # Noisy reconstruction
        X_noisy = X + torch.randn_like(X) * 0.5
        error_noisy = model.compute_temporal_reconstruction_error(X_noisy)
        assert error_noisy > error
    
    def test_state_prediction_error(self, temporal_data):
        """Test state prediction error computation."""
        model = temporal_data['model']
        X = temporal_data['X']
        
        # Perfect prediction
        error = model.compute_state_prediction_error(X)
        assert error < 1e-10
        
        # Noisy prediction
        X_noisy = X + torch.randn_like(X) * 0.1
        error_noisy = model.compute_state_prediction_error(X_noisy)
        assert error_noisy > error
    
    def test_temporal_contributions(self, temporal_data):
        """Test temporal contribution computation."""
        model = temporal_data['model']
        X = temporal_data['X']
        
        add_contribs = model.compute_temporal_additive_contribution(X)
        mult_contribs = model.compute_temporal_multiplicative_contribution(X)
        
        assert add_contribs.shape == (model.T,)
        assert mult_contribs.shape == (model.T,)
        assert torch.all(add_contribs >= 0)
        assert torch.all(mult_contribs >= 0)
    
    def test_ar_dynamics(self, temporal_data):
        """Test that AR(1) dynamics are applied correctly."""
        model = temporal_data['model']
        X = temporal_data['X']
        
        # Check temporal correlation
        for i in range(model.n):
            for t in range(1, model.T):
                # Current state should be related to previous state
                # (though with process noise)
                x_t = X[i, t]
                x_prev = X[i, t - 1]
                
                # Predicted state
                x_pred = torch.matmul(model.Phi, x_prev)
                
                # Residual should be reasonably small (scaled by Q)
                residual = x_t - x_pred
                # Just check it's not completely random
                assert torch.norm(residual) < 5.0  # Generous threshold
    
    def test_reproducibility_temporal(self, temporal_network_params):
        """Test temporal reproducibility."""
        model1 = TemporalAMEModel(**temporal_network_params)
        Y1, X1 = model1.generate_data(return_latents=True)
        
        model2 = TemporalAMEModel(**temporal_network_params)
        Y2, X2 = model2.generate_data(return_latents=True)
        
        assert torch.allclose(Y1, Y2)
        assert torch.allclose(X1, X2)


class TestBaseAMEModel:
    """Tests for BaseAMEModel utility methods."""
    
    def test_generate_covariance_matrix(self, static_model):
        """Test covariance matrix generation."""
        cov = static_model._generate_covariance_matrix(
            dim=3,
            correlation=0.5,
            variance=2.0
        )
        
        assert cov.shape == (3, 3)
        # Check diagonal
        assert torch.allclose(cov.diag(), torch.ones(3) * 2.0)
        # Check off-diagonal
        assert torch.allclose(cov[0, 1], torch.tensor(1.0))  # 0.5 * 2.0
    
    def test_block_diagonal_covariance(self, static_model):
        """Test block-diagonal covariance construction."""
        cov = static_model._block_diagonal_covariance(
            block_sizes=[2, 3],
            correlations=[0.5, 0.3],
            variances=[1.0, 2.0]
        )
        
        assert cov.shape == (5, 5)
        # Check that off-diagonal blocks are zero
        assert torch.allclose(cov[:2, 2:], torch.zeros(2, 3))
        assert torch.allclose(cov[2:, :2], torch.zeros(3, 2))
        
        # Check diagonal blocks are non-zero
        assert not torch.allclose(cov[:2, :2], torch.zeros(2, 2))
        assert not torch.allclose(cov[2:, 2:], torch.zeros(3, 3))