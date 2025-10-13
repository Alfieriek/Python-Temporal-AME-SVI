"""
Metric utilities for evaluating AME model performance.

This module provides various metrics for assessing model fit, prediction
accuracy, and uncertainty quantification.

Functions
---------
mean_squared_error
    Compute MSE between predictions and observations.
root_mean_squared_error
    Compute RMSE.
mean_absolute_error
    Compute MAE.
r_squared
    Compute coefficient of determination.
temporal_consistency_score
    Measure smoothness of temporal trajectories.
link_prediction_metrics
    Compute classification metrics for link prediction.
calibration_error
    Assess calibration of uncertainty estimates.

Author: Sean Plummer
Date: October 2025
"""

from typing import Tuple, Optional, Dict

import torch
import numpy as np
from scipy.stats import pearsonr


def mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute mean squared error.
    
    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    mask : torch.Tensor, optional
        Binary mask for selecting elements (1 = include, 0 = exclude).
        
    Returns
    -------
    mse : float
        Mean squared error.
    """
    squared_error = (y_true - y_pred) ** 2
    
    if mask is not None:
        squared_error = squared_error * mask
        n_elements = mask.sum().item()
    else:
        n_elements = squared_error.numel()
    
    if n_elements == 0:
        return 0.0
        
    mse = squared_error.sum().item() / n_elements
    return mse


def root_mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute root mean squared error.
    
    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    mask : torch.Tensor, optional
        Binary mask for selecting elements.
        
    Returns
    -------
    rmse : float
        Root mean squared error.
    """
    mse = mean_squared_error(y_true, y_pred, mask)
    return np.sqrt(mse)


def mean_absolute_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute mean absolute error.
    
    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    mask : torch.Tensor, optional
        Binary mask for selecting elements.
        
    Returns
    -------
    mae : float
        Mean absolute error.
    """
    absolute_error = torch.abs(y_true - y_pred)
    
    if mask is not None:
        absolute_error = absolute_error * mask
        n_elements = mask.sum().item()
    else:
        n_elements = absolute_error.numel()
    
    if n_elements == 0:
        return 0.0
        
    mae = absolute_error.sum().item() / n_elements
    return mae


def r_squared(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute coefficient of determination (R²).
    
    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    mask : torch.Tensor, optional
        Binary mask for selecting elements.
        
    Returns
    -------
    r2 : float
        Coefficient of determination. Values close to 1 indicate good fit.
        
    Notes
    -----
    R² = 1 - SS_res / SS_tot
    where SS_res is residual sum of squares and SS_tot is total sum of squares.
    """
    if mask is not None:
        y_true_masked = y_true[mask > 0]
        y_pred_masked = y_pred[mask > 0]
    else:
        y_true_masked = y_true.flatten()
        y_pred_masked = y_pred.flatten()
    
    if len(y_true_masked) == 0:
        return 0.0
    
    # Mean of true values
    y_mean = y_true_masked.mean()
    
    # Sum of squares
    ss_tot = ((y_true_masked - y_mean) ** 2).sum()
    ss_res = ((y_true_masked - y_pred_masked) ** 2).sum()
    
    if ss_tot < 1e-10:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def pearson_correlation(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Compute Pearson correlation coefficient.
    
    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    mask : torch.Tensor, optional
        Binary mask for selecting elements.
        
    Returns
    -------
    corr : float
        Pearson correlation coefficient.
    """
    if mask is not None:
        y_true_masked = y_true[mask > 0]
        y_pred_masked = y_pred[mask > 0]
    else:
        y_true_masked = y_true.flatten()
        y_pred_masked = y_pred.flatten()
    
    if len(y_true_masked) < 2:
        return 0.0
    
    # Convert to numpy for scipy
    y_true_np = y_true_masked.cpu().numpy()
    y_pred_np = y_pred_masked.cpu().numpy()
    
    corr, _ = pearsonr(y_true_np, y_pred_np)
    return float(corr)


def temporal_consistency_score(
    X: torch.Tensor,
    order: int = 1
) -> float:
    """
    Measure temporal smoothness of trajectories.
    
    Parameters
    ----------
    X : torch.Tensor
        State trajectories, shape (n, T, d).
    order : int, default=1
        Order of finite difference (1 = velocity, 2 = acceleration).
        
    Returns
    -------
    consistency : float
        Temporal consistency score. Lower values indicate smoother trajectories.
        
    Notes
    -----
    Computes the average magnitude of finite differences across time.
    For order=1, measures average "velocity" of state changes.
    For order=2, measures average "acceleration".
    
    Examples
    --------
    >>> X = torch.randn(15, 10, 6)
    >>> smoothness = temporal_consistency_score(X, order=1)
    """
    n, T, d = X.shape
    
    if T < order + 1:
        return 0.0
    
    # Compute finite differences
    diffs = X[:, 1:, :] - X[:, :-1, :]
    
    # Higher order differences
    for _ in range(order - 1):
        if diffs.shape[1] < 2:
            break
        diffs = diffs[:, 1:, :] - diffs[:, :-1, :]
    
    # Average magnitude
    consistency = torch.norm(diffs, dim=-1).mean().item()
    
    return consistency


def link_prediction_metrics(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    threshold: float = 0.0
) -> Dict[str, float]:
    """
    Compute classification metrics for link prediction.
    
    Parameters
    ----------
    Y_true : torch.Tensor
        True edge weights, shape (n, n).
    Y_pred : torch.Tensor
        Predicted edge weights, shape (n, n).
    threshold : float, default=0.0
        Threshold for binarizing predictions (edge exists if > threshold).
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'accuracy': Overall accuracy
        - 'precision': Precision (TP / (TP + FP))
        - 'recall': Recall (TP / (TP + FN))
        - 'f1': F1 score
        - 'auc': Area under ROC curve (if applicable)
        
    Notes
    -----
    Treats edge prediction as a binary classification problem.
    Excludes diagonal (self-loops) from computation.
    
    Examples
    --------
    >>> Y_true = torch.randn(20, 20)
    >>> Y_pred = torch.randn(20, 20)
    >>> metrics = link_prediction_metrics(Y_true, Y_pred, threshold=0.0)
    >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    n = Y_true.shape[0]
    
    # Exclude diagonal
    mask = 1 - torch.eye(n)
    Y_true_masked = Y_true * mask
    Y_pred_masked = Y_pred * mask
    
    # Binarize
    Y_true_binary = (Y_true_masked > threshold).float()
    Y_pred_binary = (Y_pred_masked > threshold).float()
    
    # Compute confusion matrix components
    tp = ((Y_true_binary == 1) & (Y_pred_binary == 1)).sum().item()
    tn = ((Y_true_binary == 0) & (Y_pred_binary == 0)).sum().item()
    fp = ((Y_true_binary == 0) & (Y_pred_binary == 1)).sum().item()
    fn = ((Y_true_binary == 1) & (Y_pred_binary == 0)).sum().item()
    
    # Compute metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def calibration_error(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    Compute expected calibration error for uncertainty estimates.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Point predictions.
    uncertainties : torch.Tensor
        Predicted standard deviations or variances.
    targets : torch.Tensor
        True target values.
    n_bins : int, default=10
        Number of bins for calibration curve.
        
    Returns
    -------
    ece : float
        Expected calibration error. Lower is better (0 = perfect calibration).
        
    Notes
    -----
    Measures whether predicted uncertainties match empirical errors.
    A well-calibrated model has uncertainties that correctly reflect
    the actual prediction errors.
    
    References
    ----------
    Guo, C., et al. (2017). On calibration of modern neural networks.
    ICML 2017.
    """
    # Compute prediction errors
    errors = torch.abs(predictions - targets)
    
    # Sort by uncertainty
    sorted_indices = torch.argsort(uncertainties)
    errors_sorted = errors[sorted_indices]
    uncertainties_sorted = uncertainties[sorted_indices]
    
    # Create bins
    n = len(predictions)
    bin_size = n // n_bins
    
    ece = 0.0
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n
        
        if end_idx <= start_idx:
            continue
        
        # Bin statistics
        bin_errors = errors_sorted[start_idx:end_idx]
        bin_uncertainties = uncertainties_sorted[start_idx:end_idx]
        
        # Expected error vs predicted uncertainty
        expected_error = bin_errors.mean().item()
        predicted_uncertainty = bin_uncertainties.mean().item()
        
        # Contribution to ECE
        weight = (end_idx - start_idx) / n
        ece += weight * abs(expected_error - predicted_uncertainty)
    
    return ece


def compute_coverage(
    predictions: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute empirical coverage of prediction intervals.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Point predictions (not used, kept for interface consistency).
    lower_bounds : torch.Tensor
        Lower bounds of prediction intervals.
    upper_bounds : torch.Tensor
        Upper bounds of prediction intervals.
    targets : torch.Tensor
        True target values.
        
    Returns
    -------
    coverage : float
        Fraction of targets falling within prediction intervals.
        
    Notes
    -----
    For a nominal 95% prediction interval, coverage should be close to 0.95.
    
    Examples
    --------
    >>> preds = torch.randn(100)
    >>> targets = torch.randn(100)
    >>> lower = preds - 1.96  # Assuming unit variance
    >>> upper = preds + 1.96
    >>> coverage = compute_coverage(preds, lower, upper, targets)
    """
    # Check if targets fall within intervals
    in_interval = (targets >= lower_bounds) & (targets <= upper_bounds)
    coverage = in_interval.float().mean().item()
    
    return coverage


def temporal_prediction_metrics(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    horizon: int = 1
) -> Dict[str, float]:
    """
    Compute prediction metrics at different time horizons.
    
    Parameters
    ----------
    Y_true : torch.Tensor
        True observations, shape (n, n, T, 2).
    Y_pred : torch.Tensor
        Predictions, shape (n, n, T, 2).
    horizon : int, default=1
        Number of steps ahead to evaluate.
        
    Returns
    -------
    metrics : dict
        Dictionary of prediction metrics at the specified horizon.
        
    Notes
    -----
    Evaluates predictions at time t using information up to t-horizon.
    """
    n, _, T, _ = Y_true.shape
    
    if T <= horizon:
        return {
            'mse': float('inf'),
            'mae': float('inf'),
            'r2': 0.0
        }
    
    # Extract predictions at specified horizon
    Y_true_h = Y_true[:, :, horizon:, :]
    Y_pred_h = Y_pred[:, :, horizon:, :]
    
    # Exclude diagonal
    mask = (1 - torch.eye(n)).unsqueeze(-1).unsqueeze(-1)
    
    # Compute metrics
    mse = mean_squared_error(Y_true_h, Y_pred_h, mask)
    mae = mean_absolute_error(Y_true_h, Y_pred_h, mask)
    r2 = r_squared(Y_true_h, Y_pred_h, mask)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics


def relative_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    epsilon: float = 1e-8
) -> float:
    """
    Compute mean relative error.
    
    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.
    epsilon : float, default=1e-8
        Small constant to avoid division by zero.
        
    Returns
    -------
    rel_error : float
        Mean relative error: mean(|y_true - y_pred| / (|y_true| + epsilon))
    """
    abs_error = torch.abs(y_true - y_pred)
    denominator = torch.abs(y_true) + epsilon
    rel_error = (abs_error / denominator).mean().item()
    
    return rel_error