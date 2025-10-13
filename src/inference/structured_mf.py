"""
Structured mean-field variational inference for temporal AME models.

This module implements structured mean-field approximations using coordinate
ascent with closed-form updates. Preserves within-(node, time) correlations:
    q(X) = ∏_i ∏_t q(X_i^t)  where q(X_i^t) is multivariate normal

Based on the approach of Zhao et al. (2024), adapted for temporal AME.

Classes
-------
TemporalAMEStructuredMFVI
    Structured mean-field VI with configurable factorization.

Author: Sean Plummer
Date: October 2025
"""

from typing import Optional, Literal

import torch
import numpy as np
from torch.distributions import MultivariateNormal

from .base import BaseTemporalVariationalInference


class TemporalAMEStructuredMFVI(BaseTemporalVariationalInference):
    """
    Structured mean-field variational inference for temporal AME.
    
    Uses coordinate ascent with closed-form updates, preserving
    within-(node, time) correlations.
    
    Parameters
    ----------
    model : TemporalAMEModel
        The temporal AME model instance with observed data.
    factorization : {"good", "bad"}, default="good"
        Type of structured factorization:
        - "good": Full covariance for [a_i, b_i, U_i, V_i]
        - "bad": Block-diagonal with [a_i, b_i] and [U_i, V_i] independent
    learning_rate : float, default=1.0
        Learning rate (damping factor) for updates.
    init_scale : float, default=0.1
        Scale for random initialization.
    seed : int, default=42
        Random seed.
        
    Examples
    --------
    >>> model = TemporalAMEModel(n_nodes=15, n_time=10)
    >>> Y = model.generate_data()
    >>> vi = TemporalAMEStructuredMFVI(model, factorization="good")
    >>> history = vi.fit(max_iter=100, verbose=True)
    """
    
    def __init__(
        self,
        model,
        factorization: Literal["good", "bad"] = "good",
        learning_rate: float = 1.0,
        init_scale: float = 0.1,
        cov_init_scale: float = 0.5,
        seed: int = 42
    ):
        """Initialize structured mean-field VI."""
        self.factorization = factorization
        self.init_scale = init_scale
        self.cov_init_scale = cov_init_scale
        
        super().__init__(model, learning_rate, seed)
        
    def _initialize_variational_params(self) -> None:
        """Initialize variational parameters."""
        # Initialize means
        self.X_mean = torch.randn(self.n, self.T, self.d) * self.init_scale
        
        # Initialize covariances based on factorization
        self.X_cov = torch.zeros(self.n, self.T, self.d, self.d)
        
        if self.factorization == "good":
            # Full covariance
            for i in range(self.n):
                for t in range(self.T):
                    cov = torch.eye(self.d) * self.cov_init_scale
                    cov += torch.randn(self.d, self.d) * 0.01
                    cov = (cov + cov.t()) / 2
                    cov += torch.eye(self.d) * 0.1  # Ensure PD
                    self.X_cov[i, t] = cov
                    
        elif self.factorization == "bad":
            # Block-diagonal structure
            for i in range(self.n):
                for t in range(self.T):
                    cov = torch.zeros(self.d, self.d)
                    
                    # Block 1: [a_i, b_i]
                    cov[:2, :2] = torch.eye(2) * self.cov_init_scale
                    cov[:2, :2] += torch.randn(2, 2) * 0.01
                    cov[:2, :2] = (cov[:2, :2] + cov[:2, :2].t()) / 2
                    cov[:2, :2] += torch.eye(2) * 0.05
                    
                    # Block 2: [U_i, V_i]
                    r2 = 2 * self.r
                    cov[2:, 2:] = torch.eye(r2) * self.cov_init_scale
                    cov[2:, 2:] += torch.randn(r2, r2) * 0.01
                    cov[2:, 2:] = (cov[2:, 2:] + cov[2:, 2:].t()) / 2
                    cov[2:, 2:] += torch.eye(r2) * 0.05
                    
                    self.X_cov[i, t] = cov
        else:
            raise ValueError(f"Unknown factorization '{self.factorization}'")
            
    def _compute_elbo(self) -> float:
        """Compute the Evidence Lower Bound."""
        elbo = 0.0
        elbo += self._compute_expected_log_likelihood()
        elbo += self._compute_log_prior_initial()
        elbo += self._compute_log_prior_transitions()
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
                    
                    # Correction for uncertainty (simplified)
                    trace_i = torch.trace(self.X_cov[i, t])
                    trace_j = torch.trace(self.X_cov[j, t])
                    correction = 0.1 * (trace_i + trace_j) * torch.trace(R_inv) / self.d
                    
                    log_lik += -0.5 * (
                        log_det_R + quad_form + correction + 2 * np.log(2 * np.pi)
                    )
                    
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
        
        Cycles through all nodes and updates each with closed-form solutions.
        """
        for i in range(self.n):
            self._update_node_i(i)
    
    def _update_node_i(self, i: int) -> None:
        """
        Update node i's trajectory with full covariances.
        
        Parameters
        ----------
        i : int
            Node index to update.
        """
        R_inv = self.model.R_inv
        Phi = self.model.Phi
        Q_inv = torch.linalg.inv(self.model.Q)
        
        # Initial state prior
        Sigma_0 = torch.zeros(self.d, self.d)
        Sigma_0[:2, :2] = self.model.Sigma
        Sigma_0[2:, 2:] = self.model.Psi
        Sigma_0_inv = torch.linalg.inv(Sigma_0)
        
        # Update each time point
        for t in range(self.T):
            # Compute precision and natural parameter
            precision = torch.zeros(self.d, self.d)
            nat_param = torch.zeros(self.d)
            
            # Observations at time t
            prec_obs, nat_obs = self._compute_observation_terms(i, t)
            precision += prec_obs
            nat_param += nat_obs
            
            # Prior (if t=0)
            if t == 0:
                precision += Sigma_0_inv
            
            # Transition from t-1 to t
            if t > 0:
                precision += Q_inv
                mu_prev = self.X_mean[i, t - 1]
                nat_param += torch.matmul(Q_inv, torch.matmul(Phi, mu_prev))
            
            # Transition from t to t+1
            if t < self.T - 1:
                precision += torch.matmul(Phi.t(), torch.matmul(Q_inv, Phi))
                mu_next = self.X_mean[i, t + 1]
                nat_param += torch.matmul(Phi.t(), torch.matmul(Q_inv, mu_next))
            
            # Solve for new mean and covariance
            cov_new = torch.linalg.inv(precision)
            
            # Enforce factorization structure
            if self.factorization == "bad":
                # Zero out off-diagonal blocks
                cov_new[:2, 2:] = 0
                cov_new[2:, :2] = 0
            
            # Ensure positive definite
            cov_new = (cov_new + cov_new.t()) / 2
            cov_new += torch.eye(self.d) * 1e-6
            
            mu_new = torch.matmul(cov_new, nat_param)
            
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
    ) -> tuple:
        """Compute observation contribution to precision and natural parameter."""
        R_inv = self.model.R_inv
        precision = torch.zeros(self.d, self.d)
        nat_param = torch.zeros(self.d)
        
        A_t = self.X_mean[:, t, :2]
        M_t = self.X_mean[:, t, 2:]
        Y_t = self.Y[:, :, t]
        
        for j in range(self.n):
            if i == j:
                continue
                
            y_ij = Y_t[i, j]
            
            # Jacobian
            J = torch.zeros(2, self.d)
            
            # mu_{ij}[0] = a_i + b_j + U_i' V_j
            J[0, 0] = 1.0  # d/da_i
            V_j = M_t[j, self.r:]
            J[0, 2:2+self.r] = V_j  # d/dU_i
            
            # mu_{ij}[1] = a_j + b_i + U_j' V_i
            J[1, 1] = 1.0  # d/db_i
            U_j = M_t[j, :self.r]
            J[1, 2+self.r:] = U_j  # d/dV_i
            
            # Update terms
            precision += torch.matmul(J.t(), torch.matmul(R_inv, J))
            nat_param += torch.matmul(J.t(), torch.matmul(R_inv, y_ij))
        
        return precision, nat_param
    
    def get_factorization_type(self) -> str:
        """Get the factorization structure type."""
        return self.factorization
    
    def get_variational_means(self) -> torch.Tensor:
        """Get variational means."""
        return self.X_mean
    
    def get_variational_covariances(self) -> torch.Tensor:
        """Get variational covariances."""
        return self.X_cov