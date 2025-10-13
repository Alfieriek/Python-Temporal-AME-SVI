"""
Diagnostic utilities for AME models.

This module provides functions for computing diagnostics, summaries, and
model quality assessments for AME network models and inference results.

Functions
---------
compute_reconstruction_error
    Compute MSE between observed and predicted networks.
compute_additive_contribution
    Compute variance explained by additive effects.
compute_multiplicative_contribution
    Compute variance explained by multiplicative effects.
compute_temporal_contributions
    Compute time-varying contributions of effects.
print_diagnostic_summary
    Print formatted summary of inference results.
compare_methods
    Compare multiple inference methods on the same data.

Author: Sean Plummer
Date: October 2025
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np


def compute_reconstruction_error(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    exclude_diagonal: bool = True
) -> float:
    """
    Compute mean squared reconstruction error.
    
    Parameters
    ----------
    Y_true : torch.Tensor
        True network data. Shape (n, n, 2) for static or (n, n, T, 2) for temporal.
    Y_pred : torch.Tensor
        Predicted network data. Same shape as Y_true.
    exclude_diagonal : bool, default=True
        If True, exclude diagonal entries (self-loops) from computation.
        
    Returns
    -------
    mse : float
        Mean squared reconstruction error.
        
    Examples
    --------
    >>> Y_true = torch.randn(20, 20, 2)
    >>> Y_pred = torch.randn(20, 20, 2)
    >>> mse = compute_reconstruction_error(Y_true, Y_pred)
    """
    # Compute squared error
    squared_error = (Y_true - Y_pred) ** 2
    
    if exclude_diagonal:
        # Create mask to exclude diagonal
        n = Y_true.shape[0]
        if Y_true.ndim == 3:  # Static: (n, n, 2)
            mask = 1 - torch.eye(n).unsqueeze(-1)
            n_elements = n * (n - 1) * 2
        else:  # Temporal: (n, n, T, 2)
            T = Y_true.shape[2]
            mask = (1 - torch.eye(n)).unsqueeze(-1).unsqueeze(-1)
            n_elements = n * (n - 1) * T * 2
            
        squared_error = squared_error * mask
    else:
        n_elements = squared_error.numel()
        
    mse = squared_error.sum().item() / n_elements
    return mse


def compute_additive_contribution(
    A: torch.Tensor,
    exclude_diagonal: bool = True
) -> float:
    """
    Compute variance explained by additive effects.
    
    Parameters
    ----------
    A : torch.Tensor
        Additive effects. Shape (n, 2) where A[:, 0] are sender effects
        and A[:, 1] are receiver effects.
    exclude_diagonal : bool, default=True
        If True, exclude diagonal entries from computation.
        
    Returns
    -------
    variance : float
        Variance explained by additive effects.
        
    Notes
    -----
    Computes the variance of the additive component:
        a_i + b_j for all pairs (i,j)
    """
    n = A.shape[0]
    a = A[:, 0]  # Sender effects
    b = A[:, 1]  # Receiver effects
    
    # Broadcast to compute a_i + b_j for all pairs
    additive = a.unsqueeze(1) + b.unsqueeze(0)
    
    if exclude_diagonal:
        mask = 1 - torch.eye(n)
        n_elements = n * (n - 1)
    else:
        mask = torch.ones(n, n)
        n_elements = n * n
        
    variance = ((additive ** 2) * mask).sum().item() / n_elements
    return variance


def compute_multiplicative_contribution(
    M: torch.Tensor,
    exclude_diagonal: bool = True
) -> float:
    """
    Compute variance explained by multiplicative effects.
    
    Parameters
    ----------
    M : torch.Tensor
        Multiplicative effects. Shape (n, 2r) where M[:, :r] are sender
        latent positions and M[:, r:] are receiver latent positions.
    exclude_diagonal : bool, default=True
        If True, exclude diagonal entries from computation.
        
    Returns
    -------
    variance : float
        Variance explained by multiplicative effects.
        
    Notes
    -----
    Computes the variance of the multiplicative component:
        U_i^T V_j for all pairs (i,j)
    """
    n = M.shape[0]
    r = M.shape[1] // 2
    
    U = M[:, :r]  # Sender latent positions
    V = M[:, r:]  # Receiver latent positions
    
    # Compute U_i^T V_j for all pairs
    multiplicative = torch.matmul(U, V.t())
    
    if exclude_diagonal:
        mask = 1 - torch.eye(n)
        n_elements = n * (n - 1)
    else:
        mask = torch.ones(n, n)
        n_elements = n * n
        
    variance = ((multiplicative ** 2) * mask).sum().item() / n_elements
    return variance


def compute_temporal_contributions(
    X: torch.Tensor,
    latent_dim: int,
    exclude_diagonal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute additive and multiplicative contributions over time.
    
    Parameters
    ----------
    X : torch.Tensor
        State trajectories of shape (n, T, d) where d = 2 + 2r.
    latent_dim : int
        Latent space dimension (r).
    exclude_diagonal : bool, default=True
        If True, exclude diagonal entries from computation.
        
    Returns
    -------
    additive_contribs : torch.Tensor
        Additive contributions over time, shape (T,).
    multiplicative_contribs : torch.Tensor
        Multiplicative contributions over time, shape (T,).
        
    Examples
    --------
    >>> X = torch.randn(15, 10, 6)  # n=15, T=10, d=6 (r=2)
    >>> add_contrib, mult_contrib = compute_temporal_contributions(X, latent_dim=2)
    >>> print(add_contrib.shape)  # (10,)
    """
    n, T, d = X.shape
    additive_contribs = torch.zeros(T)
    multiplicative_contribs = torch.zeros(T)
    
    for t in range(T):
        # Extract A_t and M_t
        A_t = X[:, t, :2]
        M_t = X[:, t, 2:]
        
        # Compute contributions at time t
        additive_contribs[t] = compute_additive_contribution(
            A_t, exclude_diagonal
        )
        multiplicative_contribs[t] = compute_multiplicative_contribution(
            M_t, exclude_diagonal
        )
        
    return additive_contribs, multiplicative_contribs


def compute_contribution_ratio(
    A: torch.Tensor,
    M: torch.Tensor
) -> float:
    """
    Compute ratio of additive to multiplicative contributions.
    
    Parameters
    ----------
    A : torch.Tensor
        Additive effects, shape (n, 2).
    M : torch.Tensor
        Multiplicative effects, shape (n, 2r).
        
    Returns
    -------
    ratio : float
        Square root of variance ratio: sqrt(Var(A) / Var(M)).
        
    Notes
    -----
    A ratio > 1 indicates additive effects dominate.
    A ratio < 1 indicates multiplicative effects dominate.
    """
    var_additive = compute_additive_contribution(A)
    var_multiplicative = compute_multiplicative_contribution(M)
    
    if var_multiplicative < 1e-10:
        return float('inf')
        
    ratio = np.sqrt(var_additive / var_multiplicative)
    return ratio


def compute_state_prediction_error(
    X_true: torch.Tensor,
    X_pred: torch.Tensor
) -> float:
    """
    Compute MSE between true and predicted state trajectories.
    
    Parameters
    ----------
    X_true : torch.Tensor
        True states, shape (n, T, d).
    X_pred : torch.Tensor
        Predicted states, shape (n, T, d).
        
    Returns
    -------
    mse : float
        Mean squared error in state space.
    """
    return ((X_true - X_pred) ** 2).mean().item()


def print_diagnostic_summary(
    method_name: str,
    history: Dict[str, List[float]],
    X_true: Optional[torch.Tensor] = None,
    X_est: Optional[torch.Tensor] = None,
    latent_dim: Optional[int] = None,
    final_only: bool = False
) -> None:
    """
    Print formatted diagnostic summary for inference results.
    
    Parameters
    ----------
    method_name : str
        Name of the inference method.
    history : dict
        Optimization history containing 'elbo', 'reconstruction_error', etc.
    X_true : torch.Tensor, optional
        True state trajectories for computing prediction error.
    X_est : torch.Tensor, optional
        Estimated state trajectories.
    latent_dim : int, optional
        Latent space dimension (needed for contribution analysis).
    final_only : bool, default=False
        If True, only print final iteration metrics.
        
    Examples
    --------
    >>> print_diagnostic_summary(
    ...     "Naive MF",
    ...     history={'elbo': elbo_hist, 'reconstruction_error': mse_hist},
    ...     X_true=X_true,
    ...     X_est=X_est,
    ...     latent_dim=2
    ... )
    """
    print("\n" + "=" * 70)
    print(f"Diagnostic Summary: {method_name}")
    print("=" * 70)
    
    # Basic convergence info
    n_iter = len(history['elbo'])
    print(f"Number of iterations: {n_iter}")
    
    if not final_only and n_iter > 0:
        print(f"Initial ELBO: {history['elbo'][0]:10.2f}")
        print(f"Final ELBO:   {history['elbo'][-1]:10.2f}")
        
        if n_iter > 1:
            elbo_change = history['elbo'][-1] - history['elbo'][0]
            print(f"ELBO change:  {elbo_change:10.2f}")
    
    # Reconstruction error
    if 'reconstruction_error' in history and len(history['reconstruction_error']) > 0:
        print(f"\nFinal reconstruction MSE: {history['reconstruction_error'][-1]:.6f}")
        
        if not final_only and n_iter > 1:
            init_mse = history['reconstruction_error'][0]
            final_mse = history['reconstruction_error'][-1]
            improvement = (1 - final_mse / init_mse) * 100 if init_mse > 0 else 0
            print(f"MSE improvement: {improvement:.1f}%")
    
    # State prediction error
    if X_true is not None and X_est is not None:
        state_mse = compute_state_prediction_error(X_true, X_est)
        print(f"\nState prediction MSE: {state_mse:.6f}")
    
    # Contribution analysis
    if X_est is not None and latent_dim is not None:
        if X_est.ndim == 3:  # Temporal: (n, T, d)
            # Use final time step
            A_final = X_est[:, -1, :2]
            M_final = X_est[:, -1, 2:]
        else:  # Static: (n, d)
            A_final = X_est[:, :2]
            M_final = X_est[:, 2:]
            
        add_contrib = compute_additive_contribution(A_final)
        mult_contrib = compute_multiplicative_contribution(M_final)
        ratio = compute_contribution_ratio(A_final, M_final)
        
        print(f"\nEffect contributions (final):")
        print(f"  Additive:       {add_contrib:.4f}")
        print(f"  Multiplicative: {mult_contrib:.4f}")
        print(f"  A/M ratio:      {ratio:.2f}")
    
    # Additional metrics from history
    if not final_only:
        additional_metrics = [k for k in history.keys() 
                            if k not in ['elbo', 'reconstruction_error']]
        if additional_metrics:
            print(f"\nAdditional metrics:")
            for metric in additional_metrics:
                if len(history[metric]) > 0:
                    print(f"  {metric}: {history[metric][-1]:.6f}")
    
    print("=" * 70)


def compare_methods(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'reconstruction_error',
    X_true: Optional[torch.Tensor] = None
) -> None:
    """
    Compare multiple inference methods on the same data.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain 'history' and 'X_est'.
    metric : str, default='reconstruction_error'
        Metric to compare. Should be a key in the history dict.
    X_true : torch.Tensor, optional
        True states for computing prediction errors.
        
    Examples
    --------
    >>> results = {
    ...     'Naive MF': {'history': hist1, 'X_est': X1},
    ...     'Good SMF': {'history': hist2, 'X_est': X2},
    ...     'Bad SMF': {'history': hist3, 'X_est': X3}
    ... }
    >>> compare_methods(results, metric='reconstruction_error')
    """
    print("\n" + "=" * 70)
    print("Method Comparison")
    print("=" * 70)
    
    # Extract final metric values
    method_scores = {}
    for method_name, result in results.items():
        history = result['history']
        if metric in history and len(history[metric]) > 0:
            method_scores[method_name] = history[metric][-1]
    
    # Sort by metric (lower is better for errors)
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1])
    
    print(f"\nFinal {metric}:")
    for rank, (method_name, score) in enumerate(sorted_methods, 1):
        print(f"  {rank}. {method_name:20s}: {score:.6f}")
    
    # State prediction errors if available
    if X_true is not None:
        print(f"\nState prediction MSE:")
        state_errors = {}
        for method_name, result in results.items():
            if 'X_est' in result:
                state_mse = compute_state_prediction_error(
                    X_true, result['X_est']
                )
                state_errors[method_name] = state_mse
        
        sorted_state_errors = sorted(state_errors.items(), key=lambda x: x[1])
        for rank, (method_name, error) in enumerate(sorted_state_errors, 1):
            print(f"  {rank}. {method_name:20s}: {error:.6f}")
    
    # Relative improvements
    if len(sorted_methods) > 1:
        baseline_name, baseline_score = sorted_methods[-1]  # Worst method
        print(f"\nImprovement over {baseline_name}:")
        for method_name, score in sorted_methods[:-1]:
            improvement = (1 - score / baseline_score) * 100
            print(f"  {method_name:20s}: {improvement:+.1f}%")
    
    print("=" * 70)


def track_convergence(
    history: Dict[str, List[float]],
    window_size: int = 10
) -> Dict[str, bool]:
    """
    Check convergence status for various metrics.
    
    Parameters
    ----------
    history : dict
        Optimization history.
    window_size : int, default=10
        Window size for computing moving statistics.
        
    Returns
    -------
    convergence_status : dict
        Dictionary mapping metric names to convergence booleans.
        
    Notes
    -----
    A metric is considered converged if the relative change over the
    last `window_size` iterations is below 1e-4.
    """
    status = {}
    
    for metric_name, values in history.items():
        if len(values) < window_size + 1:
            status[metric_name] = False
            continue
            
        # Compute relative change over window
        recent = values[-window_size:]
        rel_changes = []
        for i in range(1, len(recent)):
            if abs(recent[i-1]) > 1e-8:
                rel_change = abs(recent[i] - recent[i-1]) / abs(recent[i-1])
                rel_changes.append(rel_change)
        
        # Converged if all recent relative changes are small
        if rel_changes:
            max_rel_change = max(rel_changes)
            status[metric_name] = max_rel_change < 1e-4
        else:
            status[metric_name] = False
    
    return status


def compute_elbo_gap(
    elbo_history: List[float],
    true_log_likelihood: Optional[float] = None
) -> Optional[float]:
    """
    Compute gap between ELBO and true log-likelihood (if known).
    
    Parameters
    ----------
    elbo_history : list of float
        ELBO values over iterations.
    true_log_likelihood : float, optional
        True log p(Y) if available.
        
    Returns
    -------
    gap : float or None
        Gap between final ELBO and true log-likelihood.
        Returns None if true log-likelihood not provided.
        
    Notes
    -----
    The gap should always be non-negative (ELBO ≤ log p(Y)).
    A smaller gap indicates a tighter approximation.
    """
    if true_log_likelihood is None or len(elbo_history) == 0:
        return None
        
    final_elbo = elbo_history[-1]
    gap = true_log_likelihood - final_elbo
    
    return gap

def compute_uv_product_correlation(
    M_est: torch.Tensor,
    M_true: torch.Tensor,
    latent_dim: int
) -> float:
    """
    Compute correlation of U'V products (the identified quantity).
    
    Parameters
    ----------
    M_est : torch.Tensor
        Estimated multiplicative effects, shape (n, 2r).
    M_true : torch.Tensor
        True multiplicative effects, shape (n, 2r).
    latent_dim : int
        Latent space dimension (r).
        
    Returns
    -------
    correlation : float
        Correlation between true and estimated U'V products.
    """
    r = latent_dim
    U_est = M_est[:, :r]
    V_est = M_est[:, r:]
    U_true = M_true[:, :r]
    V_true = M_true[:, r:]
    
    # Compute products
    UV_est = (U_est @ V_est.T).flatten()
    UV_true = (U_true @ V_true.T).flatten()
    
    # Correlation
    corr = torch.corrcoef(torch.stack([UV_true, UV_est]))[0, 1]
    return corr.item()