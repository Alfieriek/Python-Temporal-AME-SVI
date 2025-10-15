"""
Visualization utilities for AME models.

This package provides comprehensive plotting functions for visualizing
AME model results, including static plots, temporal visualizations,
and method comparisons.

Modules
-------
static_plots
    Static visualizations (convergence, networks, latent spaces).
temporal_plots
    Temporal visualizations (trajectories, time-varying contributions).
comparison_plots
    Comparison visualizations (multiple methods side-by-side).

Quick Import
------------
>>> from src.visualization import (
...     plot_convergence,
...     plot_state_trajectories,
...     plot_method_comparison
... )

Examples
--------
>>> # Plot convergence
>>> from src.visualization import plot_convergence
>>> fig = plot_convergence(history, title="Model Convergence")
>>>
>>> # Plot temporal trajectories
>>> from src.visualization import plot_state_trajectories
>>> fig = plot_state_trajectories(X, node_indices=[0, 1, 2])
>>>
>>> # Compare methods
>>> from src.visualization import plot_method_comparison
>>> fig = plot_method_comparison(results, metrics=['reconstruction_error'])

Author: Sean Plummer
Date: October 2025
"""

# Static plots
from .static import (
    plot_convergence,
    plot_network,
    plot_latent_space,
    plot_contribution_breakdown,
    plot_parameter_comparison,
    plot_residuals
)

# Temporal plots
from .temporal import (
    plot_state_trajectories,
    plot_temporal_contributions,
    plot_trajectory_comparison,
    plot_network_evolution,
    plot_latent_trajectory_2d,
    plot_prediction_horizon
)

# Comparison plots
from .comparison import (
    plot_method_comparison,
    plot_convergence_comparison,
    plot_reconstruction_comparison,
    plot_parameter_recovery_grid,
    plot_computational_efficiency,
    plot_three_way_comparison
)

__all__ = [
    # Static plots
    'plot_convergence',
    'plot_network',
    'plot_latent_space',
    'plot_contribution_breakdown',
    'plot_parameter_comparison',
    'plot_residuals',
    # Temporal plots
    'plot_state_trajectories',
    'plot_temporal_contributions',
    'plot_trajectory_comparison',
    'plot_network_evolution',
    'plot_latent_trajectory_2d',
    'plot_prediction_horizon',
    # Comparison plots
    'plot_method_comparison',
    'plot_convergence_comparison',
    'plot_reconstruction_comparison',
    'plot_parameter_recovery_grid',
    'plot_computational_efficiency',
    'plot_three_way_comparison'
]