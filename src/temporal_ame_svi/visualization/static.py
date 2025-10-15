"""
Static visualization utilities for AME models.

This module provides functions for creating static plots including convergence
curves, network visualizations, and latent space plots.

Functions
---------
plot_convergence
    Plot ELBO and reconstruction error over iterations.
plot_network
    Visualize network adjacency matrix.
plot_latent_space
    Plot 2D latent space positions.
plot_contribution_breakdown
    Bar plot of additive vs multiplicative contributions.
plot_parameter_comparison
    Scatter plot comparing true vs estimated parameters.

Author: Sean Plummer
Date: October 2025
"""

from typing import Optional, Dict, List, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_convergence(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot convergence curves for ELBO and reconstruction error.
    
    Parameters
    ----------
    history : dict
        Optimization history containing 'elbo' and 'reconstruction_error'.
    figsize : tuple, default=(12, 5)
        Figure size (width, height) in inches.
    title : str, optional
        Overall title for the figure.
    save_path : str, optional
        If provided, save figure to this path.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Examples
    --------
    >>> history = {'elbo': [...], 'reconstruction_error': [...]}
    >>> fig = plot_convergence(history, title="Naive MF Convergence")
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot ELBO
    if 'elbo' in history and len(history['elbo']) > 0:
        axes[0].plot(history['elbo'], linewidth=2, color='#2E86AB')
        axes[0].set_xlabel('Iteration', fontsize=11)
        axes[0].set_ylabel('ELBO', fontsize=11)
        axes[0].set_title('Evidence Lower Bound', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
    
    # Plot reconstruction error
    if 'reconstruction_error' in history and len(history['reconstruction_error']) > 0:
        axes[1].plot(
            history['reconstruction_error'],
            linewidth=2,
            color='#A23B72'
        )
        axes[1].set_xlabel('Iteration', fontsize=11)
        axes[1].set_ylabel('MSE', fontsize=11)
        axes[1].set_title('Reconstruction Error', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_network(
    Y: torch.Tensor,
    time_index: Optional[int] = None,
    component: int = 0,
    figsize: Tuple[int, int] = (8, 7),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> Figure:
    """
    Visualize network adjacency matrix as a heatmap.
    
    Parameters
    ----------
    Y : torch.Tensor
        Network data. Shape (n, n, 2) for static or (n, n, T, 2) for temporal.
    time_index : int, optional
        For temporal data, which time step to plot.
    component : int, default=0
        Which dyadic component to plot (0 for y_ij, 1 for y_ji).
    figsize : tuple, default=(8, 7)
        Figure size.
    title : str, optional
        Title for the plot.
    save_path : str, optional
        Path to save figure.
    cmap : str, default='RdBu_r'
        Colormap name.
    vmin, vmax : float, optional
        Color scale limits.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Examples
    --------
    >>> Y = torch.randn(20, 20, 10, 2)
    >>> fig = plot_network(Y, time_index=5, title="Network at t=5")
    >>> plt.show()
    """
    # Extract appropriate slice
    if Y.ndim == 4:  # Temporal
        if time_index is None:
            time_index = Y.shape[2] - 1  # Last time step
        Y_plot = Y[:, :, time_index, component]
    else:  # Static
        Y_plot = Y[:, :, component]
    
    # Convert to numpy
    Y_np = Y_plot.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color limits
    if vmin is None:
        vmin = Y_np.min()
    if vmax is None:
        vmax = Y_np.max()
    
    # Plot heatmap
    im = ax.imshow(Y_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Edge Weight', fontsize=11)
    
    # Labels
    ax.set_xlabel('Node j', fontsize=11)
    ax.set_ylabel('Node i', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    elif Y.ndim == 4:
        comp_name = 'y_ij' if component == 0 else 'y_ji'
        ax.set_title(
            f'Network at t={time_index} ({comp_name})',
            fontsize=12,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latent_space(
    M: torch.Tensor,
    labels: Optional[np.ndarray] = None,
    time_index: Optional[int] = None,
    plot_U: bool = True,
    plot_V: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot 2D latent space positions for sender (U) and receiver (V).
    
    Parameters
    ----------
    M : torch.Tensor
        Multiplicative effects. Shape (n, 2r) for static or (n, T, 2r) for temporal.
    labels : np.ndarray, optional
        Node labels for coloring points.
    time_index : int, optional
        For temporal data, which time step to plot.
    plot_U : bool, default=True
        Whether to plot sender positions (U).
    plot_V : bool, default=True
        Whether to plot receiver positions (V).
    figsize : tuple, default=(10, 5)
        Figure size.
    title : str, optional
        Title for the figure.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Notes
    -----
    Only works for 2D latent spaces (r=2).
    
    Examples
    --------
    >>> M = torch.randn(20, 4)  # n=20, r=2
    >>> fig = plot_latent_space(M, title="Latent Space")
    >>> plt.show()
    """
    # Extract appropriate slice
    if M.ndim == 3:  # Temporal
        if time_index is None:
            time_index = M.shape[1] - 1
        M_plot = M[:, time_index, :]
    else:  # Static
        M_plot = M
    
    n = M_plot.shape[0]
    r = M_plot.shape[1] // 2
    
    if r != 2:
        raise ValueError("Latent space plotting only supported for r=2")
    
    U = M_plot[:, :2].cpu().numpy()
    V = M_plot[:, 2:].cpu().numpy()
    
    # Determine number of subplots
    n_plots = int(plot_U) + int(plot_V)
    if n_plots == 0:
        raise ValueError("Must plot at least one of U or V")
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot U (sender positions)
    if plot_U:
        ax = axes[plot_idx]
        if labels is not None:
            scatter = ax.scatter(
                U[:, 0], U[:, 1],
                c=labels,
                cmap='tab10',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            plt.colorbar(scatter, ax=ax, label='Group')
        else:
            ax.scatter(
                U[:, 0], U[:, 1],
                s=100,
                alpha=0.7,
                color='#2E86AB',
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add node labels
        for i in range(n):
            ax.annotate(
                str(i),
                (U[i, 0], U[i, 1]),
                fontsize=8,
                ha='center',
                va='center'
            )
        
        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
        ax.set_title('Sender Positions (U)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plot_idx += 1
    
    # Plot V (receiver positions)
    if plot_V:
        ax = axes[plot_idx]
        if labels is not None:
            scatter = ax.scatter(
                V[:, 0], V[:, 1],
                c=labels,
                cmap='tab10',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            plt.colorbar(scatter, ax=ax, label='Group')
        else:
            ax.scatter(
                V[:, 0], V[:, 1],
                s=100,
                alpha=0.7,
                color='#A23B72',
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add node labels
        for i in range(n):
            ax.annotate(
                str(i),
                (V[i, 0], V[i, 1]),
                fontsize=8,
                ha='center',
                va='center'
            )
        
        ax.set_xlabel('Dimension 1', fontsize=11)
        ax.set_ylabel('Dimension 2', fontsize=11)
        ax.set_title('Receiver Positions (V)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_contribution_breakdown(
    additive_contrib: float,
    multiplicative_contrib: float,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Create bar plot showing additive vs multiplicative contributions.
    
    Parameters
    ----------
    additive_contrib : float
        Variance explained by additive effects.
    multiplicative_contrib : float
        Variance explained by multiplicative effects.
    figsize : tuple, default=(8, 6)
        Figure size.
    title : str, optional
        Title for the plot.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Examples
    --------
    >>> fig = plot_contribution_breakdown(0.3, 0.7, title="Effect Contributions")
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    categories = ['Additive\nEffects', 'Multiplicative\nEffects']
    values = [additive_contrib, multiplicative_contrib]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{value:.3f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    # Compute ratio
    if multiplicative_contrib > 1e-10:
        ratio = np.sqrt(additive_contrib / multiplicative_contrib)
        ratio_text = f'A/M Ratio: {ratio:.2f}'
    else:
        ratio_text = 'A/M Ratio: ∞'
    
    ax.text(
        0.95, 0.95,
        ratio_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title(
        title if title else 'Effect Contributions',
        fontsize=13,
        fontweight='bold'
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_comparison(
    X_true: torch.Tensor,
    X_est: torch.Tensor,
    parameter_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Scatter plot comparing true vs estimated parameters.
    
    Parameters
    ----------
    X_true : torch.Tensor
        True parameter values.
    X_est : torch.Tensor
        Estimated parameter values.
    parameter_names : list of str, optional
        Names for each parameter dimension.
    figsize : tuple, default=(10, 8)
        Figure size.
    title : str, optional
        Title for the figure.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Examples
    --------
    >>> X_true = torch.randn(20, 6)
    >>> X_est = torch.randn(20, 6)
    >>> fig = plot_parameter_comparison(X_true, X_est)
    >>> plt.show()
    """
    # Flatten for comparison
    true_flat = X_true.flatten().cpu().numpy()
    est_flat = X_est.flatten().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(
        true_flat,
        est_flat,
        alpha=0.5,
        s=20,
        color='#2E86AB',
        edgecolors='none'
    )
    
    # Perfect prediction line
    min_val = min(true_flat.min(), est_flat.min())
    max_val = max(true_flat.max(), est_flat.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        linewidth=2,
        label='Perfect Prediction'
    )
    
    # Compute correlation
    correlation = np.corrcoef(true_flat, est_flat)[0, 1]
    
    # Add text box with statistics
    textstr = f'Correlation: {correlation:.3f}\nMSE: {((true_flat - est_flat)**2).mean():.4f}'
    ax.text(
        0.05, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Estimated Values', fontsize=12)
    ax.set_title(
        title if title else 'Parameter Recovery',
        fontsize=13,
        fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_residuals(
    Y_true: torch.Tensor,
    Y_pred: torch.Tensor,
    figsize: Tuple[int, int] = (12, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot residual analysis: histogram and Q-Q plot.
    
    Parameters
    ----------
    Y_true : torch.Tensor
        True observations.
    Y_pred : torch.Tensor
        Predictions.
    figsize : tuple, default=(12, 5)
        Figure size.
    title : str, optional
        Title for the figure.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Examples
    --------
    >>> Y_true = torch.randn(20, 20, 2)
    >>> Y_pred = torch.randn(20, 20, 2)
    >>> fig = plot_residuals(Y_true, Y_pred)
    >>> plt.show()
    """
    residuals = (Y_true - Y_pred).flatten().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(residuals, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
    axes[0].set_xlabel('Residual', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig