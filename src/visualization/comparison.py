"""
Comparison visualization utilities for AME models.

This module provides functions for comparing multiple inference methods
side-by-side, including convergence comparisons, performance metrics,
and parameter recovery assessments.

Functions
---------
plot_method_comparison
    Compare multiple methods on various metrics.
plot_convergence_comparison
    Compare ELBO convergence across methods.
plot_reconstruction_comparison
    Compare reconstruction errors across methods.
plot_parameter_recovery_grid
    Grid of scatter plots comparing parameter recovery.
plot_computational_efficiency
    Compare runtime and iterations to convergence.

Author: Sean Plummer
Date: October 2025
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def add_correlation_panels_to_figure(
    fig,
    gs,
    results: Dict[str, Dict[str, Any]],
    method_names: List[str],
    row_index: int = 2,
    t_plot: int = 0,
    node_plot: int = 0,
    latent_dim: int = 2
) -> None:
    """
    Add correlation structure panels to an existing figure.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add to.
    gs : matplotlib.gridspec.GridSpec
        GridSpec object for subplot layout.
    results : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain 'vi' with the inference object.
    method_names : list of str
        List of method names to plot (e.g., ['Naive MF', 'Good SMF', 'Bad SMF']).
    row_index : int, default=2
        Which row of the GridSpec to use.
    t_plot : int, default=0
        Time index to plot correlation for.
    node_plot : int, default=0
        Node index to plot correlation for.
    latent_dim : int, default=2
        Latent space dimension (r).
        
    Examples
    --------
    >>> fig = plt.figure(figsize=(16, 12))
    >>> gs = GridSpec(3, 3, figure=fig)
    >>> # ... create other panels ...
    >>> add_correlation_panels_to_figure(fig, gs, results, method_names)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    d = 2 + 2 * latent_dim
    
    # Create state labels
    state_labels = ['a', 'b']
    for k in range(latent_dim):
        state_labels.extend([f'U_{k+1}', f'V_{k+1}'])
    
    for idx, method_name in enumerate(method_names):
        if method_name not in results:
            continue
        
        ax = fig.add_subplot(gs[row_index, idx])
        
        vi = results[method_name].get('vi')
        if vi is None:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16)
            ax.set_title(method_name, fontweight='bold')
            ax.axis('off')
            continue
        
        # Get covariance
        try:
            if hasattr(vi, 'get_variational_covariances'):
                cov_all = vi.get_variational_covariances()
                cov = cov_all[node_plot, t_plot].detach()
            elif hasattr(vi, 'X_cov'):
                cov = vi.X_cov[node_plot, t_plot].detach()
            else:
                raise AttributeError("No covariance available")
                
            # Convert to correlation
            std = torch.sqrt(torch.diag(cov))
            corr = cov / (std.unsqueeze(1) @ std.unsqueeze(0))
            corr = corr.cpu().numpy()
            corr = np.nan_to_num(corr, nan=0.0)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)}', transform=ax.transAxes,
                   ha='center', va='center', fontsize=9)
            ax.set_title(method_name, fontweight='bold')
            ax.axis('off')
            continue
        
        # Plot heatmap
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', fontsize=9)
        
        # Labels
        ax.set_xticks(range(d))
        ax.set_yticks(range(d))
        ax.set_xticklabels(state_labels, fontsize=9)
        ax.set_yticklabels(state_labels, fontsize=9)
        
        # Title
        ax.set_title(
            f'{method_name}\nPosterior Correlation\n(Node {node_plot}, t={t_plot})',
            fontsize=10,
            fontweight='bold'
        )
        
        # Grid
        for i in range(d + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # Highlight structures
        if 'bad' in method_name.lower():
            # Block structure boxes
            ax.add_patch(Rectangle(
                (-0.5, -0.5), 2, 2,
                fill=False, edgecolor='yellow', linewidth=2.5, linestyle='--'
            ))
            ax.add_patch(Rectangle(
                (1.5, 1.5), d - 2, d - 2,
                fill=False, edgecolor='yellow', linewidth=2.5, linestyle='--'
            ))
            ax.text(
                0.5, 0.02,
                'Block-Diagonal',
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
            )
        
        elif 'naive' in method_name.lower():
            # Check if diagonal
            off_diag = corr - np.diag(np.diag(corr))
            if np.abs(off_diag).max() < 0.05:
                ax.text(
                    0.5, 0.02,
                    'Diagonal',
                    transform=ax.transAxes,
                    ha='center', va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
                )
        
        else:  # Good SMF
            ax.text(
                0.5, 0.02,
                'Full Covariance',
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
            )

def plot_method_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['reconstruction_error', 'state_error'],
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare multiple methods on various metrics using bar plots.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain 'history' with metric values.
    metrics : list of str, default=['reconstruction_error', 'state_error']
        Which metrics to compare.
    figsize : tuple, default=(12, 6)
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
    >>> results = {
    ...     'Naive MF': {'history': {'reconstruction_error': [...]}},
    ...     'Good SMF': {'history': {'reconstruction_error': [...]}},
    ...     'Bad SMF': {'history': {'reconstruction_error': [...]}}
    ... }
    >>> fig = plot_method_comparison(results)
    >>> plt.show()
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    method_names = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62828']
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        # Extract final values for this metric
        values = []
        for method_name in method_names:
            history = results[method_name]['history']
            if metric in history and len(history[metric]) > 0:
                values.append(history[metric][-1])
            else:
                values.append(0.0)
        
        # Create bar plot
        x_pos = np.arange(len(method_names))
        bars = ax.bar(
            x_pos,
            values,
            color=[colors[i % len(colors)] for i in range(len(method_names))],
            alpha=0.8,
            edgecolor='black'
        )
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{value:.4f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(
            metric.replace('_', ' ').title(),
            fontsize=12,
            fontweight='bold'
        )
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_convergence_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'elbo',
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_legend: bool = True
) -> Figure:
    """
    Compare convergence curves across multiple methods.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping method names to result dictionaries.
    metric : str, default='elbo'
        Which metric to plot ('elbo' or 'reconstruction_error').
    figsize : tuple, default=(12, 6)
        Figure size.
    title : str, optional
        Title for the figure.
    save_path : str, optional
        Path to save figure.
    show_legend : bool, default=True
        Whether to show legend.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Examples
    --------
    >>> results = {
    ...     'Naive MF': {'history': {'elbo': [...]}},
    ...     'Good SMF': {'history': {'elbo': [...]}}
    ... }
    >>> fig = plot_convergence_comparison(results, metric='elbo')
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62828']
    linestyles = ['-', '--', '-.', ':', '-']
    
    for idx, (method_name, result) in enumerate(results.items()):
        history = result['history']
        if metric in history and len(history[metric]) > 0:
            values = history[metric]
            ax.plot(
                range(len(values)),
                values,
                label=method_name,
                linewidth=2.5,
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                alpha=0.8
            )
    
    # Formatting
    ax.set_xlabel('Iteration', fontsize=12)
    
    if metric == 'elbo':
        ylabel = 'ELBO'
        plot_title = 'ELBO Convergence Comparison'
    elif metric == 'reconstruction_error':
        ylabel = 'Reconstruction MSE'
        plot_title = 'Reconstruction Error Comparison'
    else:
        ylabel = metric.replace('_', ' ').title()
        plot_title = f'{ylabel} Comparison'
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        title if title else plot_title,
        fontsize=13,
        fontweight='bold'
    )
    
    if show_legend:
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_reconstruction_comparison(
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare final reconstruction errors with improvement percentages.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping method names to result dictionaries.
    figsize : tuple, default=(10, 6)
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
    >>> results = {
    ...     'Naive MF': {'history': {'reconstruction_error': [...]}},
    ...     'Good SMF': {'history': {'reconstruction_error': [...]}}
    ... }
    >>> fig = plot_reconstruction_comparison(results)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    method_names = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # Extract final reconstruction errors
    final_errors = []
    for method_name in method_names:
        history = results[method_name]['history']
        if 'reconstruction_error' in history and len(history['reconstruction_error']) > 0:
            final_errors.append(history['reconstruction_error'][-1])
        else:
            final_errors.append(float('inf'))
    
    # Sort by error (best to worst)
    sorted_indices = np.argsort(final_errors)
    sorted_methods = [method_names[i] for i in sorted_indices]
    sorted_errors = [final_errors[i] for i in sorted_indices]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_methods))
    bars = ax.barh(
        y_pos,
        sorted_errors,
        color=[colors[i % len(colors)] for i in range(len(sorted_methods))],
        alpha=0.8,
        edgecolor='black'
    )
    
    # Add value labels and improvement percentages
    baseline_error = sorted_errors[-1]  # Worst method
    for bar, error, method in zip(bars, sorted_errors, sorted_methods):
        width = bar.get_width()
        
        # Value label
        label = f'{error:.4f}'
        
        # Improvement percentage (if not baseline)
        if error < baseline_error and baseline_error > 0:
            improvement = (1 - error / baseline_error) * 100
            label += f' ({improvement:+.1f}%)'
        
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha='left',
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_methods, fontsize=11)
    ax.set_xlabel('Final Reconstruction MSE', fontsize=12)
    ax.set_title(
        title if title else 'Method Comparison: Reconstruction Error',
        fontsize=13,
        fontweight='bold'
    )
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_parameter_recovery_grid(
    X_true: torch.Tensor,
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (15, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Create grid of scatter plots comparing parameter recovery across methods.
    
    Parameters
    ----------
    X_true : torch.Tensor
        True parameters for comparison.
    results : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain 'X_est'.
    figsize : tuple, default=(15, 5)
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
    >>> X_true = torch.randn(15, 10, 6)
    >>> results = {
    ...     'Naive MF': {'X_est': torch.randn(15, 10, 6)},
    ...     'Good SMF': {'X_est': torch.randn(15, 10, 6)}
    ... }
    >>> fig = plot_parameter_recovery_grid(X_true, results)
    >>> plt.show()
    """
    method_names = list(results.keys())
    n_methods = len(method_names)
    
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    true_flat = X_true.flatten().cpu().numpy()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    for idx, (ax, method_name) in enumerate(zip(axes, method_names)):
        if 'X_est' not in results[method_name]:
            continue
            
        X_est = results[method_name]['X_est']
        est_flat = X_est.flatten().cpu().numpy()
        
        # Scatter plot
        ax.scatter(
            true_flat,
            est_flat,
            alpha=0.4,
            s=10,
            color=colors[idx % len(colors)],
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
            alpha=0.7
        )
        
        # Compute correlation
        correlation = np.corrcoef(true_flat, est_flat)[0, 1]
        mse = ((true_flat - est_flat) ** 2).mean()
        
        # Add text box
        textstr = f'r = {correlation:.3f}\nMSE = {mse:.4f}'
        ax.text(
            0.05, 0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.set_xlabel('True', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Estimated', fontsize=10)
        ax.set_title(method_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_computational_efficiency(
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare computational efficiency across methods.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping method names to result dictionaries.
        Each result dict should contain 'runtime' and 'iterations'.
    figsize : tuple, default=(12, 6)
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
    >>> results = {
    ...     'Naive MF': {'runtime': 10.5, 'iterations': 100},
    ...     'Good SMF': {'runtime': 15.2, 'iterations': 80}
    ... }
    >>> fig = plot_computational_efficiency(results)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    method_names = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # Runtime comparison
    if all('runtime' in results[m] for m in method_names):
        runtimes = [results[m]['runtime'] for m in method_names]
        x_pos = np.arange(len(method_names))
        
        bars = axes[0].bar(
            x_pos,
            runtimes,
            color=[colors[i % len(colors)] for i in range(len(method_names))],
            alpha=0.8,
            edgecolor='black'
        )
        
        # Add value labels
        for bar, runtime in zip(bars, runtimes):
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{runtime:.2f}s',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(method_names, rotation=45, ha='right')
        axes[0].set_ylabel('Runtime (seconds)', fontsize=11)
        axes[0].set_title('Runtime Comparison', fontsize=12, fontweight='bold')
        axes[0].grid(True, axis='y', alpha=0.3)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
    
    # Iterations to convergence
    if all('iterations' in results[m] for m in method_names):
        iterations = [results[m]['iterations'] for m in method_names]
        x_pos = np.arange(len(method_names))
        
        bars = axes[1].bar(
            x_pos,
            iterations,
            color=[colors[i % len(colors)] for i in range(len(method_names))],
            alpha=0.8,
            edgecolor='black'
        )
        
        # Add value labels
        for bar, iters in zip(bars, iterations):
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{iters}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(method_names, rotation=45, ha='right')
        axes[1].set_ylabel('Iterations', fontsize=11)
        axes[1].set_title('Iterations to Convergence', fontsize=12, fontweight='bold')
        axes[1].grid(True, axis='y', alpha=0.3)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_three_way_comparison(
    results: Dict[str, Dict[str, Any]],
    X_true: Optional[torch.Tensor] = None,
    figsize: Tuple[int, int] = (16, 14),
    save_path: Optional[str] = None
) -> Figure:
    """
    Comprehensive three-way comparison plot with multiple panels.
    
    Parameters
    ----------
    results : dict
        Dictionary with 'Naive MF', 'Good SMF', and 'Bad SMF' results.
    X_true : torch.Tensor, optional
        True parameters for recovery analysis.
    figsize : tuple, default=(16, 10)
        Figure size.
    save_path : str, optional
        Path to save figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Notes
    -----
    Creates a comprehensive comparison with:
    - ELBO convergence
    - Reconstruction error
    - Parameter recovery (if X_true provided)
    - Final metrics bar chart
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    method_names = ['Naive MF', 'Good SMF', 'Bad SMF']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Panel 1: ELBO convergence
    ax1 = fig.add_subplot(gs[0, 0])
    for method_name, color in zip(method_names, colors):
        if method_name in results:
            elbo = results[method_name]['history'].get('elbo', [])
            if elbo:
                ax1.plot(elbo, label=method_name, color=color, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('ELBO')
    ax1.set_title('ELBO Convergence', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Reconstruction error
    ax2 = fig.add_subplot(gs[0, 1])
    for method_name, color in zip(method_names, colors):
        if method_name in results:
            recon = results[method_name]['history'].get('reconstruction_error', [])
            if recon:
                ax2.plot(recon, label=method_name, color=color, linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE')
    ax2.set_title('Reconstruction Error', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Final metrics comparison
    ax3 = fig.add_subplot(gs[0, 2])
    final_errors = []
    for method_name in method_names:
        if method_name in results:
            recon = results[method_name]['history'].get('reconstruction_error', [0])
            final_errors.append(recon[-1] if recon else 0)
        else:
            final_errors.append(0)
    
    x_pos = np.arange(len(method_names))
    bars = ax3.bar(x_pos, final_errors, color=colors, alpha=0.8, edgecolor='black')
    for bar, error in zip(bars, final_errors):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{error:.4f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(method_names, rotation=45, ha='right')
    ax3.set_ylabel('Final MSE')
    ax3.set_title('Final Performance', fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Panels 4-6: Parameter recovery (if X_true provided)
    if X_true is not None:
        true_flat = X_true.flatten().cpu().numpy()
        for idx, (method_name, color) in enumerate(zip(method_names, colors)):
            ax = fig.add_subplot(gs[1, idx])
            if method_name in results and 'X_est' in results[method_name]:
                est_flat = results[method_name]['X_est'].flatten().cpu().numpy()
                ax.scatter(true_flat, est_flat, alpha=0.3, s=10, color=color)
                
                min_val = min(true_flat.min(), est_flat.min())
                max_val = max(true_flat.max(), est_flat.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                
                corr = np.corrcoef(true_flat, est_flat)[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('True')
            ax.set_ylabel('Estimated')
            ax.set_title(f'{method_name} Recovery', fontweight='bold')
            ax.grid(True, alpha=0.3)

    if X_true is not None:
        n, T, d = X_true.shape
        latent_dim = (d - 2) // 2
    else:
        T, d, latent_dim = 10, 6, 2

    add_correlation_panels_to_figure(
        fig=fig,
        gs=gs,
        results=results,
        method_names=method_names,
        row_index=2,
        t_plot=T // 2,
        node_plot=0,
        latent_dim=latent_dim
    )
    
    fig.suptitle('Three-Way Method Comparison', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig