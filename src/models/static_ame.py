"""
Static (non-temporal) AME network model.

This module implements the standard Additive and Multiplicative Effects (AME)
model for static network data as described in Hoff (2021).

Classes
-------
StaticAMEModel
    Static AME model for single-snapshot network data.

References
----------
Hoff, P. D. (2021). Additive and multiplicative effects network models.
Statistical Science, 36(1), 34-50.

Author: Sean Plummer
Date: October 2025
"""

from typing import Optional, Tuple

import torch
import numpy as np
from torch.distributions import MultivariateNormal

from .base import BaseAMEModel


class StaticAMEModel(BaseAMEModel):
    """
    Static AME model for network data.
    
    The model structure is:
        Y_ij = [y_ij, y_ji]' ~ N(mu_ij, R)
        mu_ij = [a_i + b_j + U_i' V_j, a_j + b_i + U_j' V_i]'
        A_i = [a_i, b_i]' ~ N(0, Sigma)
        M_i = [U_i, V_i]' ~ N(0, Psi)
        
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    latent_dim : int, default=2
        Dimension of latent space (r).
    rho_additive : float, default=0.5
        Correlation between sender and receiver effects.
    rho_multiplicative : float, default=0.3
        Correlation within multiplicative effects.
    rho_dyadic : float, default=0.5
        Correlation between reciprocal edges.
    seed : int, default=42
        Random seed for reproducibility.
        
    Attributes
    ----------
    Sigma : torch.Tensor
        Covariance matrix for additive effects (2x2).
    Psi : torch.Tensor
        Covariance matrix for multiplicative effects (2r x 2r).
    A : torch.Tensor
        Additive effects matrix (n x 2), where A[i] = [a_i, b_i].
    M : torch.Tensor
        Multiplicative effects matrix (n x 2r), where M[i] = [U_i, V_i].
    Y : torch.Tensor
        Generated network data (n x n x 2).
        
    Examples
    --------
    >>> model = StaticAMEModel(n_nodes=20, latent_dim=2, seed=42)
    >>> Y = model.generate_data()
    >>> print(Y.shape)  # (20, 20, 2)
    >>> 
    >>> # Access generated parameters
    >>> additive_effects = model.A  # (20, 2)
    >>> latent_positions = model.M  # (20, 4)
    """
    
    def __init__(
        self,
        n_nodes: int,
        latent_dim: int = 2,
        rho_additive: float = 0.5,
        rho_multiplicative: float = 0.3,
        rho_dyadic: float = 0.5,
        seed: int = 42
    ):
        """Initialize static AME model."""
        super().__init__(n_nodes, latent_dim, seed)

        self.rho_additive = rho_additive
        self.rho_multiplicative = rho_multiplicative
        self.rho_dyadic = rho_dyadic
        
        # Update dyadic covariance with specified correlation
        self.R = self._generate_covariance_matrix(
            dim=2,
            correlation=rho_dyadic,
            variance=0.1
        )
        self.R_inv = torch.linalg.inv(self.R)
        
        # Initialize covariance matrices
        self._initialize_covariances()
        
        # Storage for generated parameters
        self.A: Optional[torch.Tensor] = None
        self.M: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None
        
    def _initialize_covariances(self) -> None:
        """Initialize covariance matrices for additive and multiplicative effects."""
        # Sigma: covariance for [a_i, b_i]
        self.Sigma = self._generate_covariance_matrix(
            dim=2,
            correlation=self.rho_additive,
            variance=1.0
        )
        
        # Psi: block-diagonal covariance for [U_i, V_i]
        # Structure: U_i (r-dim) correlated internally, V_i (r-dim) correlated internally
        # but U_i and V_i are independent
        self.Psi = self._block_diagonal_covariance(
            block_sizes=[self.r, self.r],
            correlations=[self.rho_multiplicative, self.rho_multiplicative],
            variances=[1.0, 1.0]
        )
        
    def generate_data(
        self,
        return_latents: bool = False
    ) -> torch.Tensor:
        """
        Generate synthetic network data from the AME model.
        
        Parameters
        ----------
        return_latents : bool, default=False
            If True, also return the latent parameters (A, M).
            
        Returns
        -------
        Y : torch.Tensor
            Network data of shape (n, n, 2), where Y[i,j] = [y_ij, y_ji].
        A : torch.Tensor, optional
            Additive effects (n, 2). Only returned if return_latents=True.
        M : torch.Tensor, optional
            Multiplicative effects (n, 2r). Only returned if return_latents=True.
            
        Notes
        -----
        The diagonal entries Y[i,i] are set to zero (no self-loops).
        Only the upper triangle is independently sampled; Y[i,j,1] = Y[j,i,0].
        """
        # Sample additive effects A_i = [a_i, b_i]
        dist_A = MultivariateNormal(
            torch.zeros(2),
            self.Sigma
        )
        self.A = torch.stack([dist_A.sample() for _ in range(self.n)])
        
        # Sample multiplicative effects M_i = [U_i, V_i]
        dist_M = MultivariateNormal(
            torch.zeros(2 * self.r),
            self.Psi
        )
        self.M = torch.stack([dist_M.sample() for _ in range(self.n)])
        
        # Compute mean structure
        mu = self.compute_mean(self.A, self.M)
        
        # Sample network data
        self.Y = torch.zeros(self.n, self.n, 2)
        dist_Y = MultivariateNormal(torch.zeros(2), self.R)
        
        for i in range(self.n):
            for j in range(i + 1, self.n):  # Upper triangle only
                # Sample dyad (y_ij, y_ji)
                dyad = mu[i, j] + dist_Y.sample()
                self.Y[i, j] = dyad
                # Ensure consistency: y_ji in Y[j,i,0]
                self.Y[j, i, 0] = dyad[1]
                self.Y[j, i, 1] = dyad[0]
                
        if return_latents:
            return self.Y, self.A, self.M
        return self.Y
    
    def compute_mean(
        self,
        A: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expected network structure given parameters.
        
        Parameters
        ----------
        A : torch.Tensor
            Additive effects of shape (n, 2).
        M : torch.Tensor
            Multiplicative effects of shape (n, 2r).
            
        Returns
        -------
        mu : torch.Tensor
            Expected network structure of shape (n, n, 2).
            mu[i,j] = [E[y_ij], E[y_ji]]
            
        Notes
        -----
        The mean structure is:
            mu_ij[0] = a_i + b_j + U_i' V_j
            mu_ij[1] = a_j + b_i + U_j' V_i
        """
        mu = torch.zeros(self.n, self.n, 2)
        
        # Extract components
        a = A[:, 0]  # Sender effects
        b = A[:, 1]  # Receiver effects
        U = M[:, :self.r]  # Sender latent positions
        V = M[:, self.r:]  # Receiver latent positions
        
        # Compute additive contributions
        # Broadcasting: a[i] + b[j] for all i,j
        additive = a.unsqueeze(1) + b.unsqueeze(0)
        
        # Compute multiplicative contributions
        # U_i' V_j for all i,j
        multiplicative = torch.matmul(U, V.t())
        
        # Combine for y_ij (first component)
        mu[:, :, 0] = additive + multiplicative
        
        # For y_ji (second component), swap indices
        mu[:, :, 1] = additive.t() + multiplicative.t()
        
        return mu
    
    def compute_reconstruction_error(
        self,
        A_est: torch.Tensor,
        M_est: torch.Tensor
    ) -> float:
        """
        Compute mean squared error between true and estimated parameters.
        
        Parameters
        ----------
        A_est : torch.Tensor
            Estimated additive effects (n, 2).
        M_est : torch.Tensor
            Estimated multiplicative effects (n, 2r).
            
        Returns
        -------
        mse : float
            Mean squared reconstruction error.
            
        Notes
        -----
        Computes ||Y - mu(A_est, M_est)||^2 / (n^2) where mu is the
        expected network structure.
        """
        if self.Y is None:
            raise ValueError("No data generated yet. Call generate_data() first.")
            
        mu_est = self.compute_mean(A_est, M_est)
        
        # Only compute error for off-diagonal elements
        mask = 1 - torch.eye(self.n).unsqueeze(-1)
        error = ((self.Y - mu_est) ** 2) * mask
        
        mse = error.sum().item() / (self.n * (self.n - 1))
        return mse
    
    def compute_additive_contribution(
        self,
        A: torch.Tensor
    ) -> float:
        """
        Compute the variance explained by additive effects.
        
        Parameters
        ----------
        A : torch.Tensor
            Additive effects (n, 2).
            
        Returns
        -------
        variance : float
            Variance of additive effects contribution.
        """
        a = A[:, 0]
        b = A[:, 1]
        additive = a.unsqueeze(1) + b.unsqueeze(0)
        
        # Exclude diagonal
        mask = 1 - torch.eye(self.n)
        return (additive ** 2 * mask).sum().item() / (self.n * (self.n - 1))
    
    def compute_multiplicative_contribution(
        self,
        M: torch.Tensor
    ) -> float:
        """
        Compute the variance explained by multiplicative effects.
        
        Parameters
        ----------
        M : torch.Tensor
            Multiplicative effects (n, 2r).
            
        Returns
        -------
        variance : float
            Variance of multiplicative effects contribution.
        """
        U = M[:, :self.r]
        V = M[:, self.r:]
        multiplicative = torch.matmul(U, V.t())
        
        # Exclude diagonal
        mask = 1 - torch.eye(self.n)
        return (multiplicative ** 2 * mask).sum().item() / (self.n * (self.n - 1))