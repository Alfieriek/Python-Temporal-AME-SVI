"""
AME network models.

This package provides implementations of Additive and Multiplicative Effects
(AME) models for network data, including both static and temporal variants.

Classes
-------
BaseAMEModel
    Abstract base class for AME models.
StaticAMEModel
    Static AME model for single-snapshot networks.
TemporalAMEModel
    Temporal AME model with AR(1) dynamics.

Quick Import
------------
>>> from src.models import StaticAMEModel, TemporalAMEModel

Examples
--------
>>> # Static model
>>> from src.models import StaticAMEModel
>>> model = StaticAMEModel(n_nodes=20, latent_dim=2)
>>> Y = model.generate_data()
>>>
>>> # Temporal model
>>> from src.models import TemporalAMEModel
>>> model = TemporalAMEModel(n_nodes=15, n_time=10, latent_dim=2)
>>> Y = model.generate_data()

Author: Sean Plummer
Date: October 2025
"""

from .base import BaseAMEModel
from .static_ame import StaticAMEModel
from .temporal_ame import TemporalAMEModel

__all__ = [
    'BaseAMEModel',
    'StaticAMEModel',
    'TemporalAMEModel'
]