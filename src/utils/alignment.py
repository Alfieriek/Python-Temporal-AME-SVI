"""
Alignment utilities for comparing estimated and true parameters.

This module provides functions for aligning estimated parameters with true
parameters to handle identifiability issues in latent space models:
- Rotation ambiguity in latent positions
- Sign/reflection ambiguity
- Permutation ambiguity

Functions
---------
procrustes_alignment
    Align latent positions using Procrustes transformation.
align_signs
    Align signs of latent vectors.
align_temporal_states
    Align state trajectories over time.
compute_alignment_error
    Compute error after optimal alignment.

Author: Sean Plummer
Date: October 2025
"""

from typing import Tuple, Optional

import torch
import numpy as np


def procrustes_alignment(
    X_est: torch.Tensor,
    X_true: torch.Tensor,
    scaling: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align estimated matrix to true matrix using Procrustes transformation.
    
    Finds rotation matrix R (and optionally scaling s) that minimizes:
        ||X_true - s * X_est @ R||^2
    
    Parameters
    ----------
    X_est : torch.Tensor
        Estimated matrix to be aligned, shape (n, d).
    X_true : torch.Tensor
        True matrix (target), shape (n, d).
    scaling : bool, default=False
        If True, also compute optimal scaling factor.
        
    Returns
    -------
    X_aligned : torch.Tensor
        Aligned version of X_est, shape (n, d).
    R : torch.Tensor
        Rotation matrix, shape (d, d).
        
    Notes
    -----
    The Procrustes solution is given by:
        R = V @ U^T where U, S, V = SVD(X_true^T @ X_est)
    
    References
    ----------
    Schonemann, P. H. (1966). A generalized solution of the orthogonal
    Procrustes problem. Psychometrika, 31(1), 1-10.
    
    Examples
    --------
    >>> X_true = torch.randn(20, 3)
    >>> X_est = torch.randn(20, 3)
    >>> X_aligned, R = procrustes_alignment(X_est, X_true)
    >>> error = torch.norm(X_true - X_aligned)
    """
    # Compute cross-covariance matrix
    M = torch.matmul(X_true.t(), X_est)
    
    # SVD of cross-covariance
    U, S, Vt = torch.linalg.svd(M)
    
    # Optimal rotation
    R = torch.matmul(U, Vt)
    
    # Handle reflection (ensure det(R) = 1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = torch.matmul(U, Vt)
    
    # Apply rotation
    X_aligned = torch.matmul(X_est, R)
    
    # Optionally compute and apply scaling
    if scaling:
        numerator = torch.trace(torch.matmul(X_true.t(), X_aligned))
        denominator = torch.trace(torch.matmul(X_aligned.t(), X_aligned))
        if denominator > 1e-10:
            s = numerator / denominator
            X_aligned = X_aligned * s
    
    return X_aligned, R


def align_signs(
    X_est: torch.Tensor,
    X_true: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Align signs of vectors to minimize L2 distance.
    
    For each vector (along specified dimension), flips sign if doing so
    reduces the distance to the corresponding true vector.
    
    Parameters
    ----------
    X_est : torch.Tensor
        Estimated parameters.
    X_true : torch.Tensor
        True parameters (target).
    dim : int, default=-1
        Dimension along which to align signs.
        
    Returns
    -------
    X_aligned : torch.Tensor
        Sign-aligned version of X_est.
        
    Examples
    --------
    >>> # Align signs of each row
    >>> X_est = torch.randn(20, 3)
    >>> X_true = torch.randn(20, 3)
    >>> X_aligned = align_signs(X_est, X_true, dim=1)
    """
    X_aligned = X_est.clone()
    
    if dim == -1 or dim == X_est.ndim - 1:
        # Align last dimension (most common case)
        for i in range(X_est.shape[0]):
            # Compute distance with and without sign flip
            dist_positive = torch.norm(X_est[i] - X_true[i])
            dist_negative = torch.norm(-X_est[i] - X_true[i])
            
            if dist_negative < dist_positive:
                X_aligned[i] = -X_aligned[i]
    else:
        # General case: align along specified dimension
        n_slices = X_est.shape[dim]
        for i in range(n_slices):
            # Get slice along dimension
            idx = [slice(None)] * X_est.ndim
            idx[dim] = i
            
            slice_est = X_est[tuple(idx)]
            slice_true = X_true[tuple(idx)]
            
            # Check if sign flip improves alignment
            dist_positive = torch.norm(slice_est - slice_true)
            dist_negative = torch.norm(-slice_est - slice_true)
            
            if dist_negative < dist_positive:
                X_aligned[tuple(idx)] = -X_aligned[tuple(idx)]
    
    return X_aligned


def align_latent_positions(
    M_est: torch.Tensor,
    M_true: torch.Tensor,
    latent_dim: int
) -> torch.Tensor:
    """
    Align estimated multiplicative effects with true values.
    
    Handles both rotation (via Procrustes) and sign ambiguity for
    multiplicative effects M = [U, V] where U and V are latent positions.
    
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
    M_aligned : torch.Tensor
        Aligned multiplicative effects, shape (n, 2r).
        
    Notes
    -----
    Aligns U and V separately using Procrustes, then applies sign alignment.
    
    Examples
    --------
    >>> M_est = torch.randn(20, 4)  # n=20, r=2
    >>> M_true = torch.randn(20, 4)
    >>> M_aligned = align_latent_positions(M_est, M_true, latent_dim=2)
    """
    r = latent_dim
    
    # Separate U and V
    U_est = M_est[:, :r]
    V_est = M_est[:, r:]
    U_true = M_true[:, :r]
    V_true = M_true[:, r:]
    
    # Procrustes alignment for U and V
    U_aligned, _ = procrustes_alignment(U_est, U_true)
    V_aligned, _ = procrustes_alignment(V_est, V_true)
    
    # Sign alignment
    U_aligned = align_signs(U_aligned, U_true, dim=1)
    V_aligned = align_signs(V_aligned, V_true, dim=1)
    
    # Concatenate back
    M_aligned = torch.cat([U_aligned, V_aligned], dim=1)
    
    return M_aligned


def align_temporal_states(
    X_est: torch.Tensor,
    X_true: torch.Tensor,
    latent_dim: int,
    align_each_time: bool = True
) -> torch.Tensor:
    """
    Align estimated state trajectories with true trajectories.
    
    Parameters
    ----------
    X_est : torch.Tensor
        Estimated states, shape (n, T, d) where d = 2 + 2r.
    X_true : torch.Tensor
        True states, shape (n, T, d).
    latent_dim : int
        Latent space dimension (r).
    align_each_time : bool, default=True
        If True, align at each time step separately.
        If False, compute a single global alignment.
        
    Returns
    -------
    X_aligned : torch.Tensor
        Aligned state trajectories, shape (n, T, d).
        
    Notes
    -----
    For temporal data, we can either:
    1. Align each time step independently (align_each_time=True)
    2. Use a single global alignment across all time (align_each_time=False)
    
    Option 1 is more flexible but may introduce discontinuities.
    Option 2 maintains temporal smoothness.
    
    Examples
    --------
    >>> X_est = torch.randn(15, 10, 6)  # n=15, T=10, d=6
    >>> X_true = torch.randn(15, 10, 6)
    >>> X_aligned = align_temporal_states(X_est, X_true, latent_dim=2)
    """
    n, T, d = X_est.shape
    X_aligned = X_est.clone()
    
    if align_each_time:
        # Align each time step independently
        for t in range(T):
            # Extract states at time t
            X_t_est = X_est[:, t, :]
            X_t_true = X_true[:, t, :]
            
            # Align additive effects (simple sign flip)
            A_t_est = X_t_est[:, :2]
            A_t_true = X_t_true[:, :2]
            A_t_aligned = align_signs(A_t_est, A_t_true, dim=1)
            
            # Align multiplicative effects (Procrustes + signs)
            M_t_est = X_t_est[:, 2:]
            M_t_true = X_t_true[:, 2:]
            M_t_aligned = align_latent_positions(
                M_t_est, M_t_true, latent_dim
            )
            
            # Combine
            X_aligned[:, t, :2] = A_t_aligned
            X_aligned[:, t, 2:] = M_t_aligned
    else:
        # Global alignment using time-averaged states
        # Compute temporal means
        X_est_mean = X_est.mean(dim=1)  # (n, d)
        X_true_mean = X_true.mean(dim=1)  # (n, d)
        
        # Find alignment based on means
        A_mean_est = X_est_mean[:, :2]
        A_mean_true = X_true_mean[:, :2]
        M_mean_est = X_est_mean[:, 2:]
        M_mean_true = X_true_mean[:, 2:]
        
        # Get rotation matrices
        _, R_M = procrustes_alignment(M_mean_est, M_mean_true)
        
        # Apply same rotation to all time steps
        for t in range(T):
            # Align additive (sign only)
            A_t_aligned = align_signs(
                X_est[:, t, :2],
                X_true[:, t, :2],
                dim=1
            )
            X_aligned[:, t, :2] = A_t_aligned
            
            # Align multiplicative (use global rotation)
            M_t = X_est[:, t, 2:]
            M_t_rotated = torch.matmul(M_t, R_M)
            M_t_aligned = align_signs(M_t_rotated, X_true[:, t, 2:], dim=1)
            X_aligned[:, t, 2:] = M_t_aligned
    
    return X_aligned


def compute_alignment_error(
    X_est: torch.Tensor,
    X_true: torch.Tensor,
    latent_dim: Optional[int] = None,
    align: bool = True
) -> Tuple[float, torch.Tensor]:
    """
    Compute alignment error between estimated and true parameters.
    
    Parameters
    ----------
    X_est : torch.Tensor
        Estimated parameters.
    X_true : torch.Tensor
        True parameters.
    latent_dim : int, optional
        Latent dimension (required for temporal alignment).
    align : bool, default=True
        If True, perform alignment before computing error.
        
    Returns
    -------
    error : float
        Mean squared error after alignment.
    X_aligned : torch.Tensor
        Aligned version of X_est.
        
    Examples
    --------
    >>> X_est = torch.randn(15, 10, 6)
    >>> X_true = torch.randn(15, 10, 6)
    >>> error, X_aligned = compute_alignment_error(
    ...     X_est, X_true, latent_dim=2, align=True
    ... )
    """
    if align:
        if X_est.ndim == 3:  # Temporal: (n, T, d)
            if latent_dim is None:
                raise ValueError(
                    "latent_dim must be provided for temporal alignment"
                )
            X_aligned = align_temporal_states(X_est, X_true, latent_dim)
        elif X_est.ndim == 2:  # Static: (n, d)
            if latent_dim is not None:
                # Has multiplicative effects
                M_aligned = align_latent_positions(
                    X_est[:, 2:], X_true[:, 2:], latent_dim
                )
                A_aligned = align_signs(X_est[:, :2], X_true[:, :2], dim=1)
                X_aligned = torch.cat([A_aligned, M_aligned], dim=1)
            else:
                # Additive only or unknown structure
                X_aligned = align_signs(X_est, X_true, dim=1)
        else:
            X_aligned = X_est
    else:
        X_aligned = X_est
    
    # Compute MSE
    error = ((X_aligned - X_true) ** 2).mean().item()
    
    return error, X_aligned


def compute_correlation_after_alignment(
    X_est: torch.Tensor,
    X_true: torch.Tensor,
    latent_dim: Optional[int] = None
) -> float:
    """
    Compute correlation between aligned estimates and true values.
    
    Parameters
    ----------
    X_est : torch.Tensor
        Estimated parameters.
    X_true : torch.Tensor
        True parameters.
    latent_dim : int, optional
        Latent dimension for alignment.
        
    Returns
    -------
    correlation : float
        Pearson correlation coefficient after alignment.
        
    Notes
    -----
    A correlation near 1.0 indicates strong recovery of true parameters.
    """
    # Align first
    _, X_aligned = compute_alignment_error(
        X_est, X_true, latent_dim, align=True
    )
    
    # Flatten both tensors
    x_aligned_flat = X_aligned.flatten()
    x_true_flat = X_true.flatten()
    
    # Compute correlation
    x_aligned_centered = x_aligned_flat - x_aligned_flat.mean()
    x_true_centered = x_true_flat - x_true_flat.mean()
    
    numerator = (x_aligned_centered * x_true_centered).sum()
    denominator = torch.sqrt(
        (x_aligned_centered ** 2).sum() * (x_true_centered ** 2).sum()
    )
    
    if denominator < 1e-10:
        return 0.0
    
    correlation = (numerator / denominator).item()
    return correlation