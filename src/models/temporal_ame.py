"""
Temporal AME network model with AR(1) dynamics.

This module implements the temporal extension of the AME model with first-order
autoregressive dynamics for the latent states.

Classes
-------
TemporalAMEModel
    Temporal AME model with AR(1) dynamics.

Author: Sean Plummer
Date: October 2025
"""

from typing import Optional, Tuple

import torch
import numpy as np
from torch.distributions import MultivariateNormal

from .static_ame import StaticAMEModel


class TemporalAMEModel(StaticAMEModel):
    """
    Temporal AME model with AR(1) dynamics.
    
    Extends the static AME model to temporal networks by adding autoregressive
    dynamics to the latent states:
    
        Y_ij^t = [y_ij^t, y_ji^t]' ~ N(mu_ij^t, R)
        mu_ij^t = [a_i^t + b_j^t + (U_i^t)' V_j^t, ...]'
        
        X_i^t = [a_i^t, b_i^t, U_i^t, V_i^t]' (stacked state vector)
        X_i^t = Phi * X_i^{t-1} + epsilon_i^t
        
    where Phi is the AR(1) transition matrix and epsilon ~ N(0, Q) is
    the process noise.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    n_time : int
        Number of time steps.
    latent_dim : int, default=2
        Dimension of latent space (r).
    ar_coefficient : float, default=0.8
        AR(1) coefficient (diagonal of Phi matrix).
    rho_additive : float, default=0.5
        Correlation between sender and receiver effects.
    rho_multiplicative : float, default=0.3
        Correlation within multiplicative effects.
    rho_dyadic : float, default=0.5
        Correlation between reciprocal edges.
    process_noise_scale : float, default=0.1
        Scale of process noise.
    seed : int, default=42
        Random seed for reproducibility.
        
    Attributes
    ----------
    T : int
        Number of time steps.
    d : int
        State dimension (2 + 2r).
    Phi : torch.Tensor
        AR(1) transition matrix (d x d).
    Q : torch.Tensor
        Process noise covariance (d x d).
    X : torch.Tensor
        Latent states (n, T, d).
    Y : torch.Tensor
        Generated network data (n, n, T, 2).
        
    Examples
    --------
    >>> model = TemporalAMEModel(
    ...     n_nodes=15,
    ...     n_time=10,
    ...     latent_dim=2,
    ...     ar_coefficient=0.8,
    ...     seed=42
    ... )
    >>> Y = model.generate_data()
    >>> print(Y.shape)  # (15, 15, 10, 2)
    >>> 
    >>> # Access latent trajectories
    >>> X = model.X  # (15, 10, 6) where 6 = 2 + 2*2
    """
    
    def __init__(
        self,
        n_nodes: int,
        n_time: int,
        latent_dim: int = 2,
        ar_coefficient: float = 0.8,
        rho_additive: float = 0.5,
        rho_multiplicative: float = 0.3,
        rho_dyadic: float = 0.5,
        process_noise_scale: float = 0.1,
        seed: int = 42
    ):
        """Initialize temporal AME model."""
        super().__init__(
            n_nodes=n_nodes,
            latent_dim=latent_dim,
            rho_additive=rho_additive,
            rho_multiplicative=rho_multiplicative,
            rho_dyadic=rho_dyadic,
            seed=seed
        )
        self.r = latent_dim
        self.T = n_time
        self.ar_coefficient = ar_coefficient
        self.process_noise_scale = process_noise_scale
        
        # State dimension: d = 2 (additive) + 2r (multiplicative)
        self.d = 2 + 2 * self.r
        
        # Initialize dynamics
        self._initialize_dynamics()
        
        # Storage for temporal data
        self.X: Optional[torch.Tensor] = None  # Latent states (n, T, d)
        self.Y: Optional[torch.Tensor] = None  # Observations (n, n, T, 2)
        
    def _initialize_dynamics(self) -> None:
        """Initialize AR(1) dynamics matrices."""
        # Transition matrix: Phi = ar_coefficient * I
        self.Phi = torch.eye(self.d) * self.ar_coefficient
        
        # Process noise covariance Q
        # Use the stationary covariance adjusted for the AR dynamics
        # Q = (1 - phi^2) * Sigma_stationary to ensure stationarity
        
        # Stack Sigma and Psi to get stationary covariance
        Sigma_stationary = torch.zeros(self.d, self.d)
        Sigma_stationary[:2, :2] = self.Sigma
        Sigma_stationary[2:, 2:] = self.Psi
        
        # Adjust for AR process to maintain stationarity
        self.Q = (1 - self.ar_coefficient ** 2) * Sigma_stationary
        self.Q = self.Q * self.process_noise_scale
        
    def generate_data(
        self,
        return_latents: bool = False
    ) -> torch.Tensor:
        """
        Generate temporal network data with AR(1) dynamics.
        
        Parameters
        ----------
        return_latents : bool, default=False
            If True, also return the latent trajectories X.
            
        Returns
        -------
        Y : torch.Tensor
            Network data of shape (n, n, T, 2).
        X : torch.Tensor, optional
            Latent state trajectories (n, T, d). Only returned if
            return_latents=True.
            
        Notes
        -----
        The initial states X_i^0 are sampled from the stationary distribution.
        Subsequent states follow X_i^t = Phi * X_i^{t-1} + epsilon_i^t.
        """
        # Initialize storage
        self.X = torch.zeros(self.n, self.T, self.d)
        self.Y = torch.zeros(self.n, self.n, self.T, 2)
        
        # Initial distribution: stationary distribution of AR(1) process
        # For AR(1): X_0 ~ N(0, Sigma_stationary)
        Sigma_0 = torch.zeros(self.d, self.d)
        Sigma_0[:2, :2] = self.Sigma
        Sigma_0[2:, 2:] = self.Psi
        
        dist_init = MultivariateNormal(torch.zeros(self.d), Sigma_0)
        dist_process = MultivariateNormal(torch.zeros(self.d), self.Q)
        dist_obs = MultivariateNormal(torch.zeros(2), self.R)
        
        # Generate trajectories for each node
        for i in range(self.n):
            # Sample initial state
            self.X[i, 0] = dist_init.sample()
            
            # Generate trajectory
            for t in range(1, self.T):
                # AR(1) dynamics: X_t = Phi * X_{t-1} + epsilon_t
                self.X[i, t] = (
                    torch.matmul(self.Phi, self.X[i, t - 1]) +
                    dist_process.sample()
                )
        
        # Generate observations at each time point
        for t in range(self.T):
            # Extract A_t and M_t from states at time t
            A_t = self.X[:, t, :2]  # Shape: (n, 2)
            M_t = self.X[:, t, 2:]  # Shape: (n, 2r)
            
            # Compute mean structure at time t
            mu_t = self.compute_mean(A_t, M_t)
            
            # Sample network at time t
            for i in range(self.n):
                for j in range(i + 1, self.n):  # Upper triangle only
                    # Sample dyad (y_ij^t, y_ji^t)
                    dyad = mu_t[i, j] + dist_obs.sample()
                    self.Y[i, j, t] = dyad
                    # Ensure consistency
                    self.Y[j, i, t, 0] = dyad[1]
                    self.Y[j, i, t, 1] = dyad[0]
        
        if return_latents:
            return self.Y, self.X
        return self.Y
    
    def get_states_at_time(
        self,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract A_t and M_t from states at time t.
        
        Parameters
        ----------
        t : int
            Time index.
            
        Returns
        -------
        A_t : torch.Tensor
            Additive effects at time t, shape (n, 2).
        M_t : torch.Tensor
            Multiplicative effects at time t, shape (n, 2r).
            
        Raises
        ------
        ValueError
            If data has not been generated yet or t is out of bounds.
        """
        if self.X is None:
            raise ValueError("No data generated yet. Call generate_data() first.")
        if t < 0 or t >= self.T:
            raise ValueError(f"Time index {t} out of bounds [0, {self.T}).")
            
        A_t = self.X[:, t, :2]
        M_t = self.X[:, t, 2:]
        return A_t, M_t
    
    def compute_temporal_reconstruction_error(
        self,
        X_est: torch.Tensor
    ) -> float:
        """
        Compute mean squared error across all time steps.
        
        Parameters
        ----------
        X_est : torch.Tensor
            Estimated latent trajectories of shape (n, T, d).
            
        Returns
        -------
        mse : float
            Mean squared reconstruction error averaged over time and nodes.
            
        Notes
        -----
        For each time t, computes ||Y^t - mu(X^t)||^2 and averages.
        """
        if self.Y is None:
            raise ValueError("No data generated yet. Call generate_data() first.")
            
        total_error = 0.0
        mask = 1 - torch.eye(self.n).unsqueeze(-1)
        
        for t in range(self.T):
            A_t_est = X_est[:, t, :2]
            M_t_est = X_est[:, t, 2:]
            mu_t_est = self.compute_mean(A_t_est, M_t_est)
            
            error_t = ((self.Y[:, :, t] - mu_t_est) ** 2) * mask
            total_error += error_t.sum().item()
            
        mse = total_error / (self.n * (self.n - 1) * self.T)
        return mse
    
    def compute_state_prediction_error(
        self,
        X_est: torch.Tensor
    ) -> float:
        """
        Compute MSE between true and estimated state trajectories.
        
        Parameters
        ----------
        X_est : torch.Tensor
            Estimated states of shape (n, T, d).
            
        Returns
        -------
        mse : float
            Mean squared error in state space.
        """
        if self.X is None:
            raise ValueError("No data generated yet. Call generate_data() first.")
            
        return ((self.X - X_est) ** 2).mean().item()
    
    def compute_temporal_additive_contribution(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute additive effects contribution over time.
        
        Parameters
        ----------
        X : torch.Tensor
            State trajectories (n, T, d).
            
        Returns
        -------
        contributions : torch.Tensor
            Additive contribution at each time step, shape (T,).
        """
        contributions = torch.zeros(self.T)
        
        for t in range(self.T):
            A_t = X[:, t, :2]
            contributions[t] = self.compute_additive_contribution(A_t)
            
        return contributions
    
    def compute_temporal_multiplicative_contribution(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multiplicative effects contribution over time.
        
        Parameters
        ----------
        X : torch.Tensor
            State trajectories (n, T, d).
            
        Returns
        -------
        contributions : torch.Tensor
            Multiplicative contribution at each time step, shape (T,).
        """
        contributions = torch.zeros(self.T)
        
        for t in range(self.T):
            M_t = X[:, t, 2:]
            contributions[t] = self.compute_multiplicative_contribution(M_t)
            
        return contributions