"""
Tests for visualization functions.

Tests that plotting functions run without errors and return proper figures.

Author: Sean Plummer
Date: October 2025
"""

import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from src.visualization import (
    # Static plots
    plot_convergence,
    plot_network,
    plot_latent_space,
    plot_contribution_breakdown,
    plot_parameter_comparison,
    plot_residuals,
    # Temporal plots
    plot_state_trajectories,
    plot_temporal_contributions,
    plot_trajectory_comparison,
    plot_network_evolution,
    plot_latent_trajectory_2d,
    plot_prediction_horizon,
    # Comparison plots
    plot_method_comparison,
    plot_convergence_comparison,
    plot_reconstruction_comparison,
    plot_parameter_recovery_grid,
    plot_computational_efficiency,
    plot_three_way_comparison
)


class TestStaticPlots:
    """Tests for static plotting functions."""
    
    def test_plot_convergence(self, mock_history):
        """Test convergence plotting."""
        fig = plot_convergence(mock_history, title="Test Convergence")
        
        assert fig is not None
        assert len(fig.axes) == 2  # Two subplots
        plt.close(fig)
    
    def test_plot_network(self, static_data):
        """Test network visualization."""
        Y = static_data['Y']
        
        fig = plot_network(Y, title="Test Network")
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_network_temporal(self, temporal_data):
        """Test network visualization for temporal data."""
        Y = temporal_data['Y']
        
        fig = plot_network(Y, time_index=2, title="Test Temporal Network")
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_latent_space(self, static_data):
        """Test latent space plotting."""
        M = static_data['M']
        
        fig = plot_latent_space(M, title="Test Latent Space")
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_latent_space_with_labels(self, static_data):
        """Test latent space plotting with node labels."""
        M = static_data['M']
        labels = np.random.randint(0, 3, size=10)
        
        fig = plot_latent_space(M, labels=labels)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_contribution_breakdown(self):
        """Test contribution breakdown plotting."""
        fig = plot_contribution_breakdown(
            additive_contrib=0.3,
            multiplicative_contrib=0.7,
            title="Test Contributions"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_parameter_comparison(self, sample_trajectories):
        """Test parameter comparison plotting."""
        X_true, X_est = sample_trajectories
        
        fig = plot_parameter_comparison(
            X_true[:, 0, :],  # First time step
            X_est[:, 0, :],
            title="Test Comparison"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_residuals(self, static_data):
        """Test residual plotting."""
        Y = static_data['Y']
        Y_pred = Y + torch.randn_like(Y) * 0.1
        
        fig = plot_residuals(Y, Y_pred, title="Test Residuals")
        
        assert fig is not None
        assert len(fig.axes) == 2  # Histogram and Q-Q plot
        plt.close(fig)


class TestTemporalPlots:
    """Tests for temporal plotting functions."""
    
    def test_plot_state_trajectories(self, temporal_data):
        """Test state trajectory plotting."""
        X = temporal_data['X']
        
        fig = plot_state_trajectories(
            X,
            node_indices=[0, 1, 2],
            title="Test Trajectories"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_temporal_contributions(self):
        """Test temporal contributions plotting."""
        add_contrib = torch.randn(10).abs()
        mult_contrib = torch.randn(10).abs()
        
        fig = plot_temporal_contributions(
            add_contrib,
            mult_contrib,
            title="Test Temporal Contributions"
        )
        
        assert fig is not None
        assert len(fig.axes) == 2  # Two subplots
        plt.close(fig)
    
    def test_plot_trajectory_comparison(self, sample_trajectories):
        """Test trajectory comparison plotting."""
        X_true, X_est = sample_trajectories
        
        fig = plot_trajectory_comparison(
            X_true,
            X_est,
            node_index=0,
            title="Test Trajectory Comparison"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_network_evolution(self, temporal_data):
        """Test network evolution plotting."""
        Y = temporal_data['Y']
        
        fig = plot_network_evolution(
            Y,
            time_indices=[0, 2, 4],
            title="Test Evolution"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_latent_trajectory_2d(self, temporal_data):
        """Test 2D latent trajectory plotting."""
        X = temporal_data['X']
        M = X[:, :, 2:]  # Multiplicative effects
        
        fig = plot_latent_trajectory_2d(
            M,
            node_indices=[0, 1, 2],
            plot_type='U',
            title="Test 2D Trajectories"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_prediction_horizon(self):
        """Test prediction horizon plotting."""
        mse_by_horizon = {1: 0.1, 2: 0.15, 3: 0.22, 4: 0.31, 5: 0.42}
        
        fig = plot_prediction_horizon(
            mse_by_horizon,
            title="Test Horizon"
        )
        
        assert fig is not None
        plt.close(fig)


class TestComparisonPlots:
    """Tests for comparison plotting functions."""
    
    @pytest.fixture
    def comparison_results(self):
        """Create mock results for comparison."""
        return {
            'Method1': {
                'history': {
                    'elbo': [-1000, -900, -850],
                    'reconstruction_error': [0.5, 0.4, 0.35]
                },
                'X_est': torch.randn(10, 5, 6),
                'runtime': 10.5,
                'iterations': 100
            },
            'Method2': {
                'history': {
                    'elbo': [-1100, -950, -870],
                    'reconstruction_error': [0.6, 0.45, 0.38]
                },
                'X_est': torch.randn(10, 5, 6),
                'runtime': 15.2,
                'iterations': 120
            }
        }
    
    def test_plot_method_comparison(self, comparison_results):
        """Test method comparison plotting."""
        fig = plot_method_comparison(
            comparison_results,
            metrics=['reconstruction_error'],
            title="Test Method Comparison"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_convergence_comparison(self, comparison_results):
        """Test convergence comparison plotting."""
        fig = plot_convergence_comparison(
            comparison_results,
            metric='elbo',
            title="Test Convergence Comparison"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_reconstruction_comparison(self, comparison_results):
        """Test reconstruction comparison plotting."""
        fig = plot_reconstruction_comparison(
            comparison_results,
            title="Test Reconstruction Comparison"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_parameter_recovery_grid(self, comparison_results):
        """Test parameter recovery grid plotting."""
        X_true = torch.randn(10, 5, 6)
        
        fig = plot_parameter_recovery_grid(
            X_true,
            comparison_results,
            title="Test Recovery Grid"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_computational_efficiency(self, comparison_results):
        """Test computational efficiency plotting."""
        fig = plot_computational_efficiency(
            comparison_results,
            title="Test Efficiency"
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_three_way_comparison(self, comparison_results):
        """Test three-way comparison plotting."""
        X_true = torch.randn(10, 5, 6)
        
        # Add third method
        results = comparison_results.copy()
        results['Method3'] = {
            'history': {
                'elbo': [-1050, -920, -860],
                'reconstruction_error': [0.55, 0.42, 0.36]
            },
            'X_est': torch.randn(10, 5, 6)
        }
        
        fig = plot_three_way_comparison(results, X_true=X_true)
        
        assert fig is not None
        plt.close(fig)


class TestPlotSaving:
    """Tests for saving plots to files."""
    
    def test_save_convergence_plot(self, mock_history, tmp_path):
        """Test saving convergence plot."""
        save_path = tmp_path / "convergence.png"
        
        fig = plot_convergence(mock_history, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_save_network_plot(self, static_data, tmp_path):
        """Test saving network plot."""
        save_path = tmp_path / "network.png"
        Y = static_data['Y']
        
        fig = plot_network(Y, save_path=str(save_path))
        
        assert save_path.exists()
        plt.close(fig)


class TestPlotEdgeCases:
    """Tests for edge cases in plotting."""
    
    def test_plot_with_empty_history(self):
        """Test plotting with empty history."""
        history = {'elbo': [], 'reconstruction_error': []}
        
        # Should not crash
        fig = plot_convergence(history)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_single_node(self):
        """Test plotting trajectories for single node."""
        X = torch.randn(1, 10, 6)
        
        fig = plot_state_trajectories(X, node_indices=[0])
        assert fig is not None
        plt.close(fig)
    
    def test_plot_single_time_step(self):
        """Test plotting with single time step."""
        X = torch.randn(10, 1, 6)
        
        # Should handle gracefully
        fig = plot_state_trajectories(X)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_latent_space_wrong_dim(self):
        """Test latent space plot with wrong dimension."""
        M = torch.randn(10, 6)  # r=3, not 2
        
        # Should raise error for non-2D latent space
        with pytest.raises(ValueError):
            plot_latent_space(M)