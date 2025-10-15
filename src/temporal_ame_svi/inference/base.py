"""
Base classes for variational inference in AME models.

This module provides abstract base classes for variational inference algorithms
applied to AME network models.

Classes
-------
BaseVariationalInference
    Abstract base class for all VI algorithms.

Author: Sean Plummer
Date: October 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import torch
import numpy as np


class BaseVariationalInference(ABC):
    """
    Abstract base class for variational inference in AME models.
    
    All VI implementations should inherit from this class and implement
    the required abstract methods.
    
    Parameters
    ----------
    model : BaseAMEModel
        The AME model instance containing the observed data.
    learning_rate : float, default=0.01
        Learning rate for gradient-based optimization.
    seed : int, default=42
        Random seed for reproducibility.
        
    Attributes
    ----------
    model : BaseAMEModel
        Reference to the model.
    Y : torch.Tensor
        Observed network data.
    n : int
        Number of nodes.
    lr : float
        Learning rate.
    history : dict
        Dictionary storing optimization history (loss, metrics, etc.).
        
    Notes
    -----
    Subclasses must implement:
    - _initialize_variational_params()
    - _compute_elbo()
    - _update_step()
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 0.01,
        seed: int = 42
    ):
        """Initialize base variational inference."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.model = model
        self.Y = model.Y
        self.n = model.n
        self.lr = learning_rate
        
        # Storage for optimization history
        self.history: Dict[str, List[float]] = {
            'elbo': [],
            'reconstruction_error': []
        }
        
        # Initialize variational parameters (implemented by subclasses)
        self._initialize_variational_params()
        
    @abstractmethod
    def _initialize_variational_params(self) -> None:
        """
        Initialize variational distribution parameters.
        
        Notes
        -----
        This method must be implemented by subclasses to set up the
        variational parameters specific to their factorization structure.
        """
        pass
    
    @abstractmethod
    def _compute_elbo(self) -> float:
        """
        Compute the Evidence Lower Bound (ELBO).
        
        Returns
        -------
        elbo : float
            Current ELBO value (to be maximized).
            
        Notes
        -----
        The ELBO is given by:
            L = E_q[log p(Y, theta)] - E_q[log q(theta)]
        where theta represents all latent variables.
        """
        pass
    
    @abstractmethod
    def _update_step(self) -> None:
        """
        Perform one update step of the variational parameters.
        
        Notes
        -----
        This method should update the variational parameters using
        the chosen optimization strategy (coordinate ascent, gradient
        ascent, natural gradients, etc.).
        """
        pass
    
    def fit(
        self,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        verbose: bool = True,
        check_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        Fit the variational distribution to the data.
        
        Parameters
        ----------
        max_iter : int, default=100
            Maximum number of iterations.
        tolerance : float, default=1e-4
            Convergence tolerance for relative change in ELBO.
        verbose : bool, default=True
            If True, print progress during optimization.
        check_every : int, default=10
            Print progress every this many iterations.
            
        Returns
        -------
        history : dict
            Dictionary containing optimization history with keys:
            - 'elbo': ELBO values over iterations
            - 'reconstruction_error': Reconstruction errors
            - Additional metrics added by subclasses
            
        Notes
        -----
        Optimization stops when either:
        1. max_iter is reached, or
        2. Relative change in ELBO < tolerance for 3 consecutive iterations
        """
        if verbose:
            print(f"Starting {self.__class__.__name__} optimization...")
            print("=" * 60)
            
        converged = False
        patience_counter = 0
        prev_elbo = -np.inf
        
        for iteration in range(max_iter):
            # Perform update step
            self._update_step()
            
            # Compute ELBO
            elbo = self._compute_elbo()
            self.history['elbo'].append(elbo)
            
            # Compute reconstruction error
            recon_error = self._compute_reconstruction_error()
            self.history['reconstruction_error'].append(recon_error)
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(elbo - prev_elbo) / (abs(prev_elbo) + 1e-8)
                if rel_change < tolerance:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    
                if patience_counter >= 3:
                    converged = True
                    
            prev_elbo = elbo
            
            # Print progress
            if verbose and (iteration % check_every == 0 or iteration == max_iter - 1):
                self._print_progress(iteration, elbo, recon_error)
                
            # Check convergence
            if converged:
                if verbose:
                    print(f"\nConverged at iteration {iteration}")
                break
                
        if verbose and not converged:
            print("\nReached maximum iterations without convergence")
            
        return self.history
    
    def _compute_reconstruction_error(self) -> float:
        """
        Compute reconstruction error using current variational parameters.
        
        Returns
        -------
        error : float
            Mean squared reconstruction error.
            
        Notes
        -----
        This default implementation should be overridden by subclasses
        if they have a more efficient computation method.
        """
        # Get variational means (subclass should store these)
        if hasattr(self, 'get_variational_means'):
            params = self.get_variational_means()
            return self.model.compute_reconstruction_error(*params)
        else:
            return 0.0
    
    def _print_progress(
        self,
        iteration: int,
        elbo: float,
        recon_error: float
    ) -> None:
        """
        Print optimization progress.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        elbo : float
            Current ELBO value.
        recon_error : float
            Current reconstruction error.
        """
        print(f"Iter {iteration:4d} | ELBO: {elbo:10.2f} | "
              f"MSE: {recon_error:.6f}")
    
    def get_elbo_history(self) -> List[float]:
        """
        Get the ELBO history.
        
        Returns
        -------
        elbo_history : list of float
            ELBO values over iterations.
        """
        return self.history['elbo']
    
    def get_reconstruction_history(self) -> List[float]:
        """
        Get the reconstruction error history.
        
        Returns
        -------
        recon_history : list of float
            Reconstruction errors over iterations.
        """
        return self.history['reconstruction_error']


class BaseTemporalVariationalInference(BaseVariationalInference):
    """
    Base class for temporal variational inference.
    
    Extends BaseVariationalInference for temporal AME models with
    additional functionality for handling time-varying latent states.
    
    Parameters
    ----------
    model : TemporalAMEModel
        The temporal AME model instance.
    learning_rate : float, default=0.01
        Learning rate for optimization.
    seed : int, default=42
        Random seed.
        
    Attributes
    ----------
    T : int
        Number of time steps.
    d : int
        State dimension.
    r : int
        Latent dimension. 
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 0.01,
        seed: int = 42
    ):
        """Initialize temporal variational inference."""
        self.T = model.T
        self.d = model.d
        self.r = model.r

        super().__init__(model, learning_rate, seed)
        
    def _compute_reconstruction_error(self) -> float:
        """
        Compute temporal reconstruction error.
        
        Returns
        -------
        error : float
            Mean squared reconstruction error across all time steps.
        """
        if hasattr(self, 'X_mean'):
            return self.model.compute_temporal_reconstruction_error(self.X_mean)
        else:
            return 0.0
    
    def _print_progress(
        self,
        iteration: int,
        elbo: float,
        recon_error: float
    ) -> None:
        """Print progress with temporal-specific metrics."""
        output = f"Iter {iteration:4d} | ELBO: {elbo:10.2f} | MSE: {recon_error:.6f}"
        
        # Add temporal-specific metrics if available
        if hasattr(self, 'history') and 'state_error' in self.history:
            if len(self.history['state_error']) > 0:
                state_err = self.history['state_error'][-1]
                output += f" | State MSE: {state_err:.6f}"
                
        print(output)