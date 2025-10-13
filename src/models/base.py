"""
Base classes for AME network models.

This module provides abstract base classes for Additive and Multiplicative
Effects (AME) network models. All specific AME model implementations should
inherit from these base classes.

Classes
-------
BaseAMEModel
    Abstract base class for AME models.

Author: Sean Plummer
Date: October 2025
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import numpy as np


class BaseAMEModel(ABC):
    """
    Abstract base class for AME network models.
    
    The AME model represents directed network data with node-specific
    additive effects (sender/receiver effects) and multiplicative effects
    (latent space positions).
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    latent_dim : int, default=2
        Dimension of the latent space.
    seed : int, default=42
        Random seed for reproducibility.
        
    Attributes
    ----------
    n : int
        Number of nodes.
    r : int
        Latent space dimension.
    R : torch.Tensor
        Dyadic error covariance matrix (2x2).
    R_inv : torch.Tensor
        Inverse of dyadic error covariance.
    Q : torch.Tensor
        Swap matrix for index reversal [[0,1],[1,0]].
        
    Notes
    -----
    The basic AME model structure is:
        Y_ij = [y_ij, y_ji]' ~ N(mu_ij, R)
        mu_ij = [a_i + b_j + U_i' V_j, a_j + b_i + U_j' V_i]'
    where:
        - a_i, b_i are additive sender/receiver effects
        - U_i, V_i are multiplicative latent positions
    """
    
    def __init__(
        self,
        n_nodes: int,
        latent_dim: int = 2,
        sigma: float = 1.0,        # ADD THIS
        rho: float = 0.0,          # ADD THIS  
        seed: int = 42
    ):
        """Initialize base AME model."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.n = n_nodes
        self.r = latent_dim
        
        # Parameterized dyadic covariance
        self.sigma = sigma
        self.rho = rho
        self.R = torch.tensor([
            [sigma**2, rho * sigma**2],
            [rho * sigma**2, sigma**2]
        ], dtype=torch.float32)
        self.R_inv = torch.linalg.inv(self.R)
        
        # Swap matrix Q for index reversal
        self.Q = torch.tensor([[0., 1.], [1., 0.]])
        
    @abstractmethod
    def generate_data(self, **kwargs) -> torch.Tensor:
        """
        Generate synthetic network data from the model.
        
        Returns
        -------
        Y : torch.Tensor
            Generated network data.
            
        Notes
        -----
        This method must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def compute_mean(self, **kwargs) -> torch.Tensor:
        """
        Compute the expected network structure.
        
        Returns
        -------
        mu : torch.Tensor
            Expected network structure.
            
        Notes
        -----
        This method must be implemented by subclasses.
        """
        pass
    
    def _generate_covariance_matrix(
        self,
        dim: int,
        correlation: float = 0.5,
        variance: float = 1.0
    ) -> torch.Tensor:
        """
        Generate a covariance matrix with specified correlation structure.
        
        Parameters
        ----------
        dim : int
            Dimension of the covariance matrix.
        correlation : float, default=0.5
            Off-diagonal correlation coefficient.
        variance : float, default=1.0
            Diagonal variance.
            
        Returns
        -------
        cov : torch.Tensor
            Covariance matrix of shape (dim, dim).
            
        Notes
        -----
        Creates a matrix with `variance` on the diagonal and
        `correlation * variance` on off-diagonals.
        """
        cov = torch.ones(dim, dim) * correlation * variance
        cov.diagonal().copy_(torch.ones(dim) * variance)
        return cov
    
    def _block_diagonal_covariance(
        self,
        block_sizes: list,
        correlations: list,
        variances: Optional[list] = None
    ) -> torch.Tensor:
        """
        Create a block-diagonal covariance matrix.
        
        Parameters
        ----------
        block_sizes : list of int
            Sizes of each diagonal block.
        correlations : list of float
            Within-block correlations for each block.
        variances : list of float, optional
            Diagonal variances for each block. If None, uses 1.0 for all.
            
        Returns
        -------
        cov : torch.Tensor
            Block-diagonal covariance matrix.
            
        Examples
        --------
        >>> model = ConcreteAMEModel(n_nodes=10)
        >>> # Create 2x2 block with corr=0.5, then 4x4 block with corr=0.3
        >>> cov = model._block_diagonal_covariance([2, 4], [0.5, 0.3])
        """
        if variances is None:
            variances = [1.0] * len(block_sizes)
            
        total_dim = sum(block_sizes)
        cov = torch.zeros(total_dim, total_dim)
        
        start_idx = 0
        for size, corr, var in zip(block_sizes, correlations, variances):
            end_idx = start_idx + size
            cov[start_idx:end_idx, start_idx:end_idx] = \
                self._generate_covariance_matrix(size, corr, var)
            start_idx = end_idx
            
        return cov