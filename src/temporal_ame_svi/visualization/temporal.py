"""
Temporal visualization utilities for AME models.

This module provides functions for visualizing temporal dynamics including
trajectories, time-varying contributions, and network evolution.

Functions
---------
plot_state_trajectories
    Plot latent state trajectories over time.
plot_temporal_contributions
    Plot additive and multiplicative contributions over time.
plot_trajectory_comparison
    Compare true vs estimated trajectories.
plot_network_evolution
    Create grid of network snapshots over time.
plot_latent_trajectory_2d
    Plot 2D latent space trajectories with time arrows.

Author: Sean Plummer
Date: October 2025
"""

from typing import Optional, List, Tuple, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch


def plot_state_trajectories(
    X: torch.Tensor,
    node_indices: Optional[List[int]] = None,
    state_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot latent state trajectories over time for selected nodes.
    
    Parameters
    ----------
    X : torch.Tensor
        State trajectories, shape (n, T, d).
    node_indices : list of int, optional
        Which nodes to plot. If None, plot first 5 nodes.
    state_indices : list of int, optional
        Which state dimensions to plot. If None, plot all.
    figsize : tuple, default=(12, 8)
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
    >>> X = torch.randn(15, 10, 6)
    >>> fig = plot_state_trajectories(X, node_indices=[0, 1, 2])
    >>> plt.show()
    """
    n, T, d = X.shape
    
    if node_indices is None:
        node_indices = list(range(min(5, n)))
    if state_indices is None:
        state_indices = list(range(d))
    
    n_states = len(state_indices)
    n_nodes = len(node_indices)
    
    # Create subplots
    fig, axes = plt.subplots(n_states, 1, figsize=figsize, sharex=True)
    if n_states == 1:
        axes = [axes]
    
    # State names
    state_names = [f'$a$', f'$b$', f'$U_1$', f'$U_2$', f'$V_1$', f'$V_2$']
    if d > 6:
        state_names = [f'State {i}' for i in range(d)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_nodes))
    
    for plot_idx, state_idx in enumerate(state_indices):
        ax = axes[plot_idx]
        
        for node_idx, color in zip(node_indices, colors):
            trajectory = X[node_idx, :, state_idx].cpu().numpy()
            ax.plot(
                range(T),
                trajectory,
                label=f'Node {node_idx}',
                linewidth=2,
                alpha=0.8,
                color=color
            )
        
        ax.set_ylabel(state_names[state_idx], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if plot_idx == 0:
            ax.legend(
                loc='upper right',
                fontsize=9,
                ncol=min(3, n_nodes),
                framealpha=0.9
            )
    
    axes[-1].set_xlabel('Time', fontsize=11)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_temporal_contributions(
    additive_contribs: torch.Tensor,
    multiplicative_contribs: torch.Tensor,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot additive and multiplicative contributions over time.
    
    Parameters
    ----------
    additive_contribs : torch.Tensor
        Additive contributions over time, shape (T,).
    multiplicative_contribs : torch.Tensor
        Multiplicative contributions over time, shape (T,).
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
    >>> add_contrib = torch.randn(10).abs()
    >>> mult_contrib = torch.randn(10).abs()
    >>> fig = plot_temporal_contributions(add_contrib, mult_contrib)
    >>> plt.show()
    """
    T = len(additive_contribs)
    time = np.arange(T)
    
    add_np = additive_contribs.cpu().numpy()
    mult_np = multiplicative_contribs.cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot contributions
    axes[0].plot(time, add_np, linewidth=2, color='#2E86AB', label='Additive')
    axes[0].plot(time, mult_np, linewidth=2, color='#A23B72', label='Multiplicative')
    axes[0].set_ylabel('Contribution', fontsize=11)
    axes[0].set_title('Effect Contributions Over Time', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Plot ratio
    ratio = np.sqrt(add_np / (mult_np + 1e-10))
    axes[1].plot(time, ratio, linewidth=2, color='#F18F01')
    axes[1].axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('A/M Ratio', fontsize=11)
    axes[1].set_title('Additive/Multiplicative Ratio', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_trajectory_comparison(
    X_true: torch.Tensor,
    X_est: torch.Tensor,
    node_index: int = 0,
    state_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Compare true vs estimated trajectories for a single node.
    
    Parameters
    ----------
    X_true : torch.Tensor
        True state trajectories, shape (n, T, d).
    X_est : torch.Tensor
        Estimated state trajectories, shape (n, T, d).
    node_index : int, default=0
        Which node to plot.
    state_indices : list of int, optional
        Which state dimensions to plot. If None, plot all.
    figsize : tuple, default=(12, 8)
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
    >>> X_est = torch.randn(15, 10, 6)
    >>> fig = plot_trajectory_comparison(X_true, X_est, node_index=0)
    >>> plt.show()
    """
    n, T, d = X_true.shape
    
    if state_indices is None:
        state_indices = list(range(d))
    
    n_states = len(state_indices)
    
    fig, axes = plt.subplots(n_states, 1, figsize=figsize, sharex=True)
    if n_states == 1:
        axes = [axes]
    
    # State names
    state_names = [f'$a$', f'$b$', f'$U_1$', f'$U_2$', f'$V_1$', f'$V_2$']
    if d > 6:
        state_names = [f'State {i}' for i in range(d)]
    
    time = np.arange(T)
    
    for plot_idx, state_idx in enumerate(state_indices):
        ax = axes[plot_idx]
        
        # True trajectory
        true_traj = X_true[node_index, :, state_idx].cpu().numpy()
        ax.plot(
            time,
            true_traj,
            linewidth=2.5,
            color='#2E86AB',
            label='True',
            alpha=0.8
        )
        
        # Estimated trajectory
        est_traj = X_est[node_index, :, state_idx].cpu().numpy()
        ax.plot(
            time,
            est_traj,
            linewidth=2,
            color='#A23B72',
            label='Estimated',
            linestyle='--',
            alpha=0.8
        )
        
        ax.set_ylabel(state_names[state_idx], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if plot_idx == 0:
            ax.legend(fontsize=10, loc='upper right')
    
    axes[-1].set_xlabel('Time', fontsize=11)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        fig.suptitle(
            f'Trajectory Comparison: Node {node_index}',
            fontsize=14,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_network_evolution(
    Y: torch.Tensor,
    time_indices: Optional[List[int]] = None,
    component: int = 0,
    figsize: Tuple[int, int] = (15, 4),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cmap: str = 'RdBu_r'
) -> Figure:
    """
    Create grid of network snapshots over time.
    
    Parameters
    ----------
    Y : torch.Tensor
        Network data, shape (n, n, T, 2).
    time_indices : list of int, optional
        Which time steps to plot. If None, plot 5 evenly spaced steps.
    component : int, default=0
        Which dyadic component to plot.
    figsize : tuple, default=(15, 4)
        Figure size.
    title : str, optional
        Title for the figure.
    save_path : str, optional
        Path to save figure.
    cmap : str, default='RdBu_r'
        Colormap.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Examples
    --------
    >>> Y = torch.randn(20, 20, 10, 2)
    >>> fig = plot_network_evolution(Y, time_indices=[0, 3, 6, 9])
    >>> plt.show()
    """
    n, _, T, _ = Y.shape
    
    if time_indices is None:
        # Select 5 evenly spaced time points
        time_indices = [int(t) for t in np.linspace(0, T-1, min(5, T))]
    
    n_plots = len(time_indices)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Determine global color scale
    Y_component = Y[:, :, :, component]
    vmin = Y_component.min().item()
    vmax = Y_component.max().item()
    
    for ax, t in zip(axes, time_indices):
        Y_t = Y[:, :, t, component].cpu().numpy()
        
        im = ax.imshow(Y_t, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f't = {t}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add colorbar
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        comp_name = 'y_ij' if component == 0 else 'y_ji'
        fig.suptitle(f'Network Evolution ({comp_name})', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latent_trajectory_2d(
    M: torch.Tensor,
    node_indices: Optional[List[int]] = None,
    plot_type: str = 'U',
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_arrows: bool = True
) -> Figure:
    """
    Plot 2D latent space trajectories with time arrows.
    
    Parameters
    ----------
    M : torch.Tensor
        Multiplicative effects, shape (n, T, 2r) where r=2.
    node_indices : list of int, optional
        Which nodes to plot. If None, plot first 5 nodes.
    plot_type : str, default='U'
        Which latent positions to plot: 'U' (sender) or 'V' (receiver).
    figsize : tuple, default=(10, 8)
        Figure size.
    title : str, optional
        Title for the figure.
    save_path : str, optional
        Path to save figure.
    show_arrows : bool, default=True
        Whether to show directional arrows along trajectories.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
        
    Notes
    -----
    Only works for 2D latent spaces (r=2).
    
    Examples
    --------
    >>> M = torch.randn(15, 10, 4)  # n=15, T=10, r=2
    >>> fig = plot_latent_trajectory_2d(M, node_indices=[0, 1, 2])
    >>> plt.show()
    """
    n, T, dim = M.shape
    r = dim // 2
    
    if r != 2:
        raise ValueError("2D trajectory plotting only supported for r=2")
    
    if node_indices is None:
        node_indices = list(range(min(5, n)))
    
    # Extract U or V
    if plot_type.upper() == 'U':
        positions = M[:, :, :2]  # (n, T, 2)
        position_name = 'Sender (U)'
    elif plot_type.upper() == 'V':
        positions = M[:, :, 2:]  # (n, T, 2)
        position_name = 'Receiver (V)'
    else:
        raise ValueError("plot_type must be 'U' or 'V'")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(node_indices)))
    
    for node_idx, color in zip(node_indices, colors):
        traj = positions[node_idx].cpu().numpy()  # (T, 2)
        
        # Plot trajectory
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linewidth=2,
            alpha=0.6,
            color=color,
            label=f'Node {node_idx}'
        )
        
        # Mark start and end
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            s=100,
            color=color,
            marker='o',
            edgecolors='black',
            linewidth=2,
            zorder=5,
            label=f'Start {node_idx}' if node_idx == node_indices[0] else None
        )
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            s=100,
            color=color,
            marker='s',
            edgecolors='black',
            linewidth=2,
            zorder=5,
            label=f'End {node_idx}' if node_idx == node_indices[0] else None
        )
        
        # Add arrows along trajectory
        if show_arrows and T > 2:
            # Add arrows at every 20% of trajectory
            arrow_indices = [int(T * p) for p in [0.2, 0.4, 0.6, 0.8]]
            for idx in arrow_indices:
                if idx < T - 1:
                    dx = traj[idx+1, 0] - traj[idx, 0]
                    dy = traj[idx+1, 1] - traj[idx, 1]
                    ax.arrow(
                        traj[idx, 0],
                        traj[idx, 1],
                        dx * 0.5,
                        dy * 0.5,
                        head_width=0.05,
                        head_length=0.05,
                        fc=color,
                        ec=color,
                        alpha=0.7,
                        zorder=3
                    )
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(
        title if title else f'Latent Space Trajectories: {position_name}',
        fontsize=13,
        fontweight='bold'
    )
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_horizon(
    mse_by_horizon: Dict[int, float],
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot prediction error as a function of forecast horizon.
    
    Parameters
    ----------
    mse_by_horizon : dict
        Dictionary mapping horizon (int) to MSE (float).
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
    >>> mse_by_horizon = {1: 0.1, 2: 0.15, 3: 0.22, 4: 0.31, 5: 0.42}
    >>> fig = plot_prediction_horizon(mse_by_horizon)
    >>> plt.show()
    """
    horizons = sorted(mse_by_horizon.keys())
    mses = [mse_by_horizon[h] for h in horizons]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        horizons,
        mses,
        marker='o',
        markersize=8,
        linewidth=2,
        color='#2E86AB'
    )
    
    ax.set_xlabel('Forecast Horizon', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title(
        title if title else 'Prediction Error by Horizon',
        fontsize=13,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig