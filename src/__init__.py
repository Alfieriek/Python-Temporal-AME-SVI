"""
Temporal AME: Variational Inference for Temporal Network Models

This package provides implementations of Additive and Multiplicative Effects
(AME) models for temporal network data, along with various variational
inference algorithms for parameter estimation.

Modules
-------
models
    AME model implementations (static and temporal).
inference
    Variational inference algorithms.
utils
    Utility functions for diagnostics, alignment, and metrics.

Quick Start
-----------
>>> from src.models import TemporalAMEModel
>>> from src.inference import TemporalAMEStructuredMFVI
>>> from src.utils import print_diagnostic_summary
>>>
>>> # Generate synthetic data
>>> model = TemporalAMEModel(n_nodes=15, n_time=10, latent_dim=2)
>>> Y, X_true = model.generate_data(return_latents=True)
>>>
>>> # Fit model with structured MF
>>> vi = TemporalAMEStructuredMFVI(model, factorization="good")
>>> history = vi.fit(max_iter=100, verbose=True)
>>>
>>> # Evaluate results
>>> X_est = vi.get_variational_means()
>>> print_diagnostic_summary(
...     "Structured MF",
...     history,
...     X_true=X_true,
...     X_est=X_est,
...     latent_dim=2
... )

Author: Sean Plummer
Date: October 2025
References: Hoff (2021), Statistical Science
"""

__version__ = '0.1.0'

# Make key classes available at package level
from .models import StaticAMEModel, TemporalAMEModel
from .inference import (
    TemporalAMENaiveMFVI,
    TemporalAMEStructuredMFVI
)

__all__ = [
    'StaticAMEModel',
    'TemporalAMEModel',
    'TemporalAMENaiveMFVI',
    'TemporalAMEStructuredMFVI'
]