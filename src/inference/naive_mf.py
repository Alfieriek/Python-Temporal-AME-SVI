"""
Naive mean-field variational inference for temporal AME models.

This module implements a fully factorized (naive) mean-field approximation
using coordinate ascent with closed-form updates:
    q(X) = ∏_i ∏_t q(X_i^t)

Based on the approach of Zhao et al. (2024) for temporal latent space models,
adapted to the temporal AME setting.

Classes
-------
TemporalAMENaiveMFVI
    Naive mean-field VI with coordinate ascent updates.

Author: Sean Plummer
Date: October 2025
"""

from typing import Optional, Tuple

import torch
import numpy as np
from torch.distributions import MultivariateNormal

from .base import BaseTemporalVariationalInference


class TemporalAMENaiveMFVI(BaseTemporalVariationalInference):
    """
    Naive mean-field variational inference for temporal AME.
    
    Uses coordinate ascent with closed-form updates for each node's
    trajectory. Each update solves a linear Gaussian inference problem.
    
    Assumes a fully factorized posterior:
        q(X) = ∏_{i=1}^n ∏_{t=0}^{T-1} q(X_i^t)
    where each q(X_i^t) = N(mu_i^t, Sigma_i^t) with diagonal Sigma_i^t.
    
    Parameters
    ----------
    model : TemporalAMEModel
        The temporal AME model instance with observed data.
    learning_rate : float, default=1.0
        Learning rate (damping factor) for updates.
    init_scale : float, default=0.1
        Scale for random initialization of means.
    seed : int, default=42
        Random seed for reproducibility.
        
    Examples
    --------
    >>> model = TemporalAMEModel(n_nodes=15, n_time=10)
    >>> Y = model.generate_data()
    >>> vi = TemporalAMENaiveMFVI(model)
    >>> history = vi.fit(max_iter=100, verbose=True)
    >>> X_est = vi.X_mean
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 1.0,
        init_scale: float = 0.1,
        seed: int = 42
    ):
        """Initialize naive mean-field VI with coordinate ascent."""
        self.init_scale = init_scale
        super().__init__(model, learning_rate, seed)
        
    def _initialize_variational_params(self) -> None:
        """
        Initialize variational parameters.
        
        Notes
        -----
        Initializes means with small random values and covariances
        as diagonal matrices (naive MF assumption).
        """
        # Initialize means
        self.X_mean = torch.randn(self.n, self.T, self.d) * self.init_scale
        
        # Initialize covariances as diagonal
        self.X_cov = torch.zeros(self.n, self.T, self.d, self.d)
        for i in range(self.n):
            for t in range(self.T):
                self.X_cov[i, t] = torch.eye(self.d) * 0.5
                
    def _compute_elbo(self) -> float:
        """
        Compute the Evidence Lower Bound.
        
        Returns
        -------
        elbo : float
            Current ELBO value.
        """
        elbo = 0.0
        
        # Expected log-likelihood
        elbo += self._compute_expected_log_likelihood()
        
        # Prior on initial states
        elbo += self._compute_log_prior_initial()
        
        # Transition priors
        elbo += self._compute_log_prior_transitions()
        
        # Entropy
        elbo += self._compute_entropy()
        
        return elbo
    
    def _compute_expected_log_likelihood(self) -> float:
        """Compute E_q[log p(Y|X)]."""
        log_lik = 0.0
        R_inv = self.model.R_inv
        log_det_R = torch.logdet(self.model.R)
        
        for t in range(self.T):
            A_t = self.X_mean[:, t, :2]
            M_t = self.X_mean[:, t, 2:]
            mu_t = self.model.compute_mean(A_t, M_t)
            Y_t = self.Y[:, :, t]
            
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    resid = Y_t[i, j] - mu_t[i, j]
                    quad_form = torch.matmul(resid, torch.matmul(R_inv, resid))
                    log_lik += -0.5 * (log_det_R + quad_form + 2 * np.log(2 * np.pi))
                    
        return log_lik
    
    def _compute_log_prior_initial(self) -> float:
        """Compute E_q[log p(X^0)]."""
        Sigma_0 = torch.zeros(self.d, self.d)
        Sigma_0[:2, :2] = self.model.Sigma
        Sigma_0[2:, 2:] = self.model.Psi
        Sigma_0_inv = torch.linalg.inv(Sigma_0)
        log_det_Sigma_0 = torch.logdet(Sigma_0)
        
        log_prior = 0.0
        for i in range(self.n):
            mu_0 = self.X_mean[i, 0]
            Sigma_0_q = self.X_cov[i, 0]
            
            quad_form = torch.matmul(mu_0, torch.matmul(Sigma_0_inv, mu_0))
            trace_term = torch.trace(torch.matmul(Sigma_0_inv, Sigma_0_q))
            
            log_prior += -0.5 * (
                log_det_Sigma_0 + quad_form + trace_term + 
                self.d * np.log(2 * np.pi)
            )
            
        return log_prior
    
    def _compute_log_prior_transitions(self) -> float:
        """Compute E_q[log p(X^{1:T} | X^{0:T-1})]."""
        Phi = self.model.Phi
        Q = self.model.Q
        Q_inv = torch.linalg.inv(Q)
        log_det_Q = torch.logdet(Q)
        
        log_prior = 0.0
        for i in range(self.n):
            for t in range(1, self.T):
                mu_t = self.X_mean[i, t]
                mu_prev = self.X_mean[i, t - 1]
                Sigma_t = self.X_cov[i, t]
                
                expected_mean = torch.matmul(Phi, mu_prev)
                resid = mu_t - expected_mean
                
                quad_form = torch.matmul(resid, torch.matmul(Q_inv, resid))
                trace_term = torch.trace(torch.matmul(Q_inv, Sigma_t))
                
                log_prior += -0.5 * (
                    log_det_Q + quad_form + trace_term + 
                    self.d * np.log(2 * np.pi)
                )
                
        return log_prior
    
    def _compute_entropy(self) -> float:
        """Compute entropy H[q(X)]."""
        entropy = 0.0
        for i in range(self.n):
            for t in range(self.T):
                log_det = torch.logdet(self.X_cov[i, t])
                entropy += 0.5 * (self.d * (1 + np.log(2 * np.pi)) + log_det)
        return entropy
    
    def _update_step(self) -> None:
        """
        Perform one coordinate ascent update step.
        
        Notes
        -----
        Cycles through all nodes and updates each node's trajectory
        using closed-form solutions. For naive MF with diagonal covariances,
        we update each time point independently.
        """
        # Cycle through all nodes
        for i in range(self.n):
            self._update_node_i(i)
    
    def _update_node_i(self, i: int) -> None:
        """
        Update node i's trajectory using coordinate ascent.
        
        Parameters
        ----------
        i : int
            Node index to update.
            
        Notes
        -----
        For naive MF with diagonal covariances, we update each time point
        independently. The update for X_i^t comes from:
        - Observations at time t involving node i
        - Prior on X_i^0 (if t=0)
        - Transitions from t-1 to t and from t to t+1
        """
        R_inv = self.model.R_inv
        Phi = self.model.Phi
        Q_inv = torch.linalg.inv(self.model.Q)
        
        # Initial state prior
        Sigma_0 = torch.zeros(self.d, self.d)
        Sigma_0[:2, :2] = self.model.Sigma
        Sigma_0[2:, 2:] = self.model.Psi
        Sigma_0_inv = torch.linalg.inv(Sigma_0)
        
        # Update each time point for node i
        for t in range(self.T):
            # Compute precision (inverse covariance) and natural parameter
            precision = torch.zeros(self.d, self.d)
            nat_param = torch.zeros(self.d)
            
            # 1. Contribution from observations at time t
            prec_obs, nat_obs = self._compute_observation_terms(i, t)
            precision += prec_obs
            nat_param += nat_obs
            
            # 2. Contribution from prior (if t=0)
            if t == 0:
                precision += Sigma_0_inv
                # nat_param += 0 (prior mean is zero)
            
            # 3. Contribution from transition from t-1 to t
            if t > 0:
                precision += Q_inv
                # E[X_{t-1}] from current estimate
                mu_prev = self.X_mean[i, t - 1]
                nat_param += torch.matmul(Q_inv, torch.matmul(Phi, mu_prev))
            
            # 4. Contribution from transition from t to t+1
            if t < self.T - 1:
                # X_t appears in transition: X_{t+1} = Phi * X_t + eps
                # This contributes Phi^T Q^{-1} Phi to precision
                precision += torch.matmul(Phi.t(), torch.matmul(Q_inv, Phi))
                # And Phi^T Q^{-1} E[X_{t+1}] to natural parameter
                mu_next = self.X_mean[i, t + 1]
                nat_param += torch.matmul(Phi.t(), torch.matmul(Q_inv, mu_next))
            
            # Solve for new mean and covariance
            # First solve for the mean using the full precision
            mu_new = torch.linalg.solve(precision, nat_param)

            # Then extract diagonal variances for naive MF
            precision_diag = torch.diag(precision)
            # Add small constant for stability
            var_diag = 1.0 / (precision_diag + 1e-8)  
            cov_new = torch.diag(var_diag)
                        
            # Damped update
            self.X_mean[i, t] = (
                self.lr * mu_new + (1 - self.lr) * self.X_mean[i, t]
            )
            self.X_cov[i, t] = (
                self.lr * cov_new + (1 - self.lr) * self.X_cov[i, t]
            )
    
    def _compute_observation_terms(
        self, 
        i: int, 
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute observation contribution to precision and natural parameter.
        
        Parameters
        ----------
        i : int
            Node index.
        t : int
            Time index.
            
        Returns
        -------
        precision : torch.Tensor
            Precision contribution from observations, shape (d, d).
        nat_param : torch.Tensor
            Natural parameter contribution, shape (d,).
            
        Notes
        -----
        For node i at time t, observations Y_{ij}^t and Y_{ji}^t depend on
        X_i^t through the mean structure:
            mu_{ij}^t[0] = a_i^t + b_j^t + U_i^t' V_j^t
            mu_{ij}^t[1] = a_j^t + b_i^t + U_j^t' V_i^t
        """
        R_inv = self.model.R_inv
        precision = torch.zeros(self.d, self.d)
        nat_param = torch.zeros(self.d)
        
        # Extract current estimates for other nodes
        A_t = self.X_mean[:, t, :2]  # (n, 2)
        M_t = self.X_mean[:, t, 2:]  # (n, 2r)
        
        Y_t = self.Y[:, :, t]  # (n, n, 2)
        
        # Node i sends to all other nodes j
        for j in range(self.n):
            if i == j:
                continue
                
            # Observation Y_{ij}^t
            y_ij = Y_t[i, j]  # (2,): [y_ij, y_ji]
            
            # Linearize mean structure around current estimates
            # mu_{ij}[0] = a_i + b_j + U_i' V_j
            # mu_{ij}[1] = a_j + b_i + U_j' V_i
            
            # Jacobian of mu_{ij} w.r.t. X_i = [a_i, b_i, U_i, V_i]
            J = torch.zeros(2, self.d)
            
            # Derivative of mu_{ij}[0] w.r.t. [a_i, b_i, U_i, V_i]
            J[0, 0] = 1.0  # d/da_i
            J[0, 1] = 0.0  # d/db_i
            V_j = M_t[j, self.r:]  # Receiver latent position of j
            J[0, 2:2+self.r] = V_j  # d/dU_i = V_j
            J[0, 2+self.r:] = 0.0  # d/dV_i
            
            # Derivative of mu_{ij}[1] w.r.t. [a_i, b_i, U_i, V_i]
            J[1, 0] = 0.0  # d/da_i
            J[1, 1] = 1.0  # d/db_i
            U_j = M_t[j, :self.r]  # Sender latent position of j
            J[1, 2:2+self.r] = 0.0  # d/dU_i
            J[1, 2+self.r:] = U_j  # d/dV_i = U_j
            
            # Compute expected mean without X_i contribution
            a_j, b_j = A_t[j]
            U_j = M_t[j, :self.r]
            V_j = M_t[j, self.r:]
            
            # Current estimate of mu_{ij}
            a_i, b_i = A_t[i]
            U_i = M_t[i, :self.r]
            V_i = M_t[i, self.r:]
            
            mu_ij = torch.zeros(2)
            mu_ij[0] = a_i + b_j + torch.dot(U_i, V_j)
            mu_ij[1] = a_j + b_i + torch.dot(U_j, V_i)
            
            # Residual
            resid = y_ij - mu_ij
            
            # Update precision: J^T R^{-1} J
            precision += torch.matmul(J.t(), torch.matmul(R_inv, J))
            
            # Update natural parameter: J^T R^{-1} (y - mu + J * X_i)
            # Simplifies to: J^T R^{-1} y (since we'll add back the X_i term)
            nat_param += torch.matmul(J.t(), torch.matmul(R_inv, y_ij))
        
        return precision, nat_param
    
    def get_variational_means(self) -> torch.Tensor:
        """Get variational means."""
        return self.X_mean
    
    def get_variational_covariances(self) -> torch.Tensor:
        """Get variational covariances."""
        return self.X_cov
    
    def predict_forward(self, n_steps: int = 1) -> torch.Tensor:
        """Predict future states using AR(1) dynamics."""
        X_pred = torch.zeros(self.n, n_steps, self.d)
        Phi = self.model.Phi
        
        for i in range(self.n):
            x_current = self.X_mean[i, -1].clone()
            for s in range(n_steps):
                x_current = torch.matmul(Phi, x_current)
                X_pred[i, s] = x_current
                
        return X_pred