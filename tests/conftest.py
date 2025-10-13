"""
Shared fixtures for all tests.

This module provides common test fixtures including test data, models,
and configurations that are reused across multiple test files.

Author: Sean Plummer
Date: October 2025
"""

import pytest
import torch
import numpy as np

from src.models import StaticAMEModel, TemporalAMEModel


@pytest.fixture
def seed():
    """Standard random seed for reproducibility."""
    return 42


@pytest.fixture
def small_network_params():
    """Parameters for small test networks."""
    return {
        'n_nodes': 10,
        'latent_dim': 2,
        'seed': 42
    }


@pytest.fixture
def temporal_network_params():
    """Parameters for temporal test networks."""
    return {
        'n_nodes': 10,
        'n_time': 5,
        'latent_dim': 2,
        'ar_coefficient': 0.8,
        'seed': 42
    }


@pytest.fixture
def static_model(small_network_params):
    """Create a static AME model for testing."""
    return StaticAMEModel(**small_network_params)


@pytest.fixture
def temporal_model(temporal_network_params):
    """Create a temporal AME model for testing."""
    return TemporalAMEModel(**temporal_network_params)


@pytest.fixture
def static_data(static_model):
    """Generate static network data."""
    Y, A, M = static_model.generate_data(return_latents=True)
    return {
        'Y': Y,
        'A': A,
        'M': M,
        'model': static_model
    }


@pytest.fixture
def temporal_data(temporal_model):
    """Generate temporal network data."""
    Y, X = temporal_model.generate_data(return_latents=True)
    return {
        'Y': Y,
        'X': X,
        'model': temporal_model
    }


@pytest.fixture
def mock_history():
    """Mock optimization history for testing diagnostics."""
    return {
        'elbo': [-1000.0, -900.0, -850.0, -820.0, -810.0],
        'reconstruction_error': [0.5, 0.4, 0.35, 0.32, 0.31]
    }


@pytest.fixture(autouse=True)
def set_random_seeds(seed):
    """Automatically set random seeds before each test."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@pytest.fixture
def sample_trajectories():
    """Sample state trajectories for testing."""
    n, T, d = 10, 5, 6
    X_true = torch.randn(n, T, d)
    X_est = X_true + torch.randn(n, T, d) * 0.1
    return X_true, X_est