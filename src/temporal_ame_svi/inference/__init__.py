"""
Variational inference methods for AME models.

This package provides variational inference algorithms for fitting AME models
to network data, including both naive and structured mean-field approximations.

Classes
-------
BaseVariationalInference
    Abstract base class for VI methods.
BaseTemporalVariationalInference
    Base class for temporal VI methods.
TemporalAMENaiveMFVI
    Naive (fully factorized) mean-field VI.
TemporalAMEStructuredMFVI
    Structured mean-field VI with configurable factorization.

Quick Import
------------
>>> from src.inference import (
...     TemporalAMENaiveMFVI,
...     TemporalAMEStructuredMFVI
... )

Examples
--------
>>> from src.models import TemporalAMEModel
>>> from src.inference import TemporalAMENaiveMFVI
>>>
>>> # Generate data
>>> model = TemporalAMEModel(n_nodes=15, n_time=10)
>>> Y = model.generate_data()
>>>
>>> # Fit with naive MF
>>> vi = TemporalAMENaiveMFVI(model, learning_rate=0.01)
>>> history = vi.fit(max_iter=100, verbose=True)
>>>
>>> # Get estimates
>>> X_est = vi.X_mean

Author: Sean Plummer
Date: October 2025
"""

from .base import (
    BaseVariationalInference,
    BaseTemporalVariationalInference
)
from .naive_mf import TemporalAMENaiveMFVI
from .structured_mf import TemporalAMEStructuredMFVI

__all__ = [
    'BaseVariationalInference',
    'BaseTemporalVariationalInference',
    'TemporalAMENaiveMFVI',
    'TemporalAMEStructuredMFVI'
]