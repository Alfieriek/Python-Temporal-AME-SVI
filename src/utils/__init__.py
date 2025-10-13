"""
Utility functions for AME models.

This package provides utilities for diagnostics, alignment, and metrics
computation for AME network models.

Modules
-------
diagnostics
    Functions for computing model diagnostics and summaries.
alignment
    Functions for aligning estimated parameters with true parameters.
metrics
    Functions for computing various performance metrics.

Quick Import
------------
>>> from src.utils import (
...     compute_reconstruction_error,
...     procrustes_alignment,
...     mean_squared_error
... )

Author: Sean Plummer
Date: October 2025
"""

# Diagnostics
from .diagnostics import (
    compute_reconstruction_error,
    compute_additive_contribution,
    compute_multiplicative_contribution,
    compute_temporal_contributions,
    compute_contribution_ratio,
    compute_state_prediction_error,
    print_diagnostic_summary,
    compare_methods,
    track_convergence,
    compute_elbo_gap
)

# Alignment
from .alignment import (
    procrustes_alignment,
    align_signs,
    align_latent_positions,
    align_temporal_states,
    compute_alignment_error,
    compute_correlation_after_alignment
)

# Metrics
from .metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared,
    pearson_correlation,
    temporal_consistency_score,
    link_prediction_metrics,
    calibration_error,
    compute_coverage,
    temporal_prediction_metrics,
    relative_error
)

__all__ = [
    # Diagnostics
    'compute_reconstruction_error',
    'compute_additive_contribution',
    'compute_multiplicative_contribution',
    'compute_temporal_contributions',
    'compute_contribution_ratio',
    'compute_state_prediction_error',
    'print_diagnostic_summary',
    'compare_methods',
    'track_convergence',
    'compute_elbo_gap',
    # Alignment
    'procrustes_alignment',
    'align_signs',
    'align_latent_positions',
    'align_temporal_states',
    'compute_alignment_error',
    'compute_correlation_after_alignment',
    # Metrics
    'mean_squared_error',
    'root_mean_squared_error',
    'mean_absolute_error',
    'r_squared',
    'pearson_correlation',
    'temporal_consistency_score',
    'link_prediction_metrics',
    'calibration_error',
    'compute_coverage',
    'temporal_prediction_metrics',
    'relative_error'
]