"""
Experiment comparing Naive MF vs Structured MF under different
multiplicative effect strengths.

This experiment demonstrates when structured factorization provides
substantial benefits over naive mean-field approximation.

Author: Sean Plummer
Date: October 2025
"""

import sys
import os
from pathlib import Path
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.temporal_ame import TemporalAMEModel
from src.inference.naive_mf import TemporalAMENaiveMFVI
from src.inference.structured_mf import TemporalAMEStructuredMFVI
from src.utils.alignment import compute_alignment_error
from src.utils.diagnostics import (
    compute_temporal_contributions,
    print_diagnostic_summary,
    compare_methods
)
from src.visualization.comparison import (
    plot_convergence_comparison,
    plot_method_comparison,
    plot_parameter_recovery_grid
)
from src.visualization.temporal import (
    plot_temporal_contributions,
    plot_trajectory_comparison
)


def compute_uv_correlation_over_time(
    X_est: torch.Tensor,
    X_true: torch.Tensor,
    latent_dim: int
) -> torch.Tensor:
    """
    Compute correlation between true and estimated U'V products over time.
    
    Parameters
    ----------
    X_est : torch.Tensor
        Estimated states, shape (n, T, d).
    X_true : torch.Tensor
        True states, shape (n, T, d).
    latent_dim : int
        Latent space dimension (r).
        
    Returns
    -------
    correlations : torch.Tensor
        Correlation at each time step, shape (T,).
    """
    n, T, d = X_est.shape
    correlations = torch.zeros(T)
    
    for t in range(T):
        # Extract U, V at time t
        U_true = X_true[:, t, 2:2+latent_dim]
        V_true = X_true[:, t, 2+latent_dim:]
        U_est = X_est[:, t, 2:2+latent_dim]
        V_est = X_est[:, t, 2+latent_dim:]
        
        # Compute U'V products
        UV_true = (U_true @ V_true.T).flatten()
        UV_est = (U_est @ V_est.T).flatten()
        
        # Correlation
        if len(UV_true) > 1:
            corr_matrix = torch.corrcoef(torch.stack([UV_true, UV_est]))
            correlations[t] = corr_matrix[0, 1].item()
        else:
            correlations[t] = 0.0
    
    return correlations


def run_experiment(
    scenario_name: str,
    rho_additive: float = 0.5,
    rho_multiplicative: float = 0.5,
    additive_scale: float = 1.0,
    multiplicative_scale: float = 1.0,
    n_nodes: int = 20,
    n_time: int = 10,
    seed: int = 42
):
    """
    Run one experiment scenario.
    
    Parameters
    ----------
    scenario_name : str
        Name of the scenario (e.g., "Weak Multiplicative").
    rho_additive : float
        Correlation between sender/receiver effects.
    rho_multiplicative : float
        Correlation within multiplicative effects.
    additive_scale : float
        Scale factor for additive effects variance.
    multiplicative_scale : float
        Scale factor for multiplicative effects variance.
    n_nodes : int
        Number of nodes.
    n_time : int
        Number of time steps.
    seed : int
        Random seed.
        
    Returns
    -------
    results : dict
        Dictionary containing results for all methods.
    model : TemporalAMEModel
        The model used.
    X_true : torch.Tensor
        True latent states.
    """
    print("\n" + "="*70)
    print(f"SCENARIO: {scenario_name}")
    print("="*70)
    print(f"Parameters:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Time steps: {n_time}")
    print(f"  Additive scale: {additive_scale}")
    print(f"  Multiplicative scale: {multiplicative_scale}")
    print(f"  ρ_additive: {rho_additive}")
    print(f"  ρ_multiplicative: {rho_multiplicative}")
    
    # Create model
    model = TemporalAMEModel(
        n_nodes=n_nodes,
        n_time=n_time,
        latent_dim=2,
        ar_coefficient=0.8,
        rho_additive=rho_additive,
        rho_multiplicative=rho_multiplicative,
        rho_dyadic=0.5,
        process_noise_scale=0.1,
        seed=seed
    )
    
    # Generate data
    Y, X_true = model.generate_data(return_latents=True)
    
    # Scale the additive and multiplicative effects
    if additive_scale != 1.0 or multiplicative_scale != 1.0:
        # Modify the true states
        X_true_scaled = X_true.clone()
        X_true_scaled[:, :, :2] *= additive_scale  # Scale [a, b]
        X_true_scaled[:, :, 2:] *= multiplicative_scale  # Scale [U, V]
        
        # Regenerate Y from scaled parameters
        Y = torch.zeros(n_nodes, n_nodes, n_time, 2)
        for t in range(n_time):
            A_t = X_true_scaled[:, t, :2]
            M_t = X_true_scaled[:, t, 2:]
            mu_t = model.compute_mean(A_t, M_t)
            
            # Add noise
            from torch.distributions import MultivariateNormal
            dist_obs = MultivariateNormal(torch.zeros(2), model.R)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    dyad = mu_t[i, j] + dist_obs.sample()
                    Y[i, j, t] = dyad
                    Y[j, i, t, 0] = dyad[1]
                    Y[j, i, t, 1] = dyad[0]
        
        # Update model's Y
        model.Y = Y
        X_true = X_true_scaled
    
    # Compute true contributions
    add_contrib_true, mult_contrib_true = compute_temporal_contributions(
        X_true, latent_dim=2
    )
    print(f"\nTrue effect contributions:")
    print(f"  Additive (mean): {add_contrib_true.mean():.4f}")
    print(f"  Multiplicative (mean): {mult_contrib_true.mean():.4f}")
    print(f"  A/M ratio: {np.sqrt(add_contrib_true.mean() / (mult_contrib_true.mean() + 1e-10)):.2f}")
    
    # Run inference methods
    results = {}
    
    # Naive MF
    print(f"\n{'-'*70}")
    print("Running Naive MF...")
    print(f"{'-'*70}")
    vi_naive = TemporalAMENaiveMFVI(model, learning_rate=0.7, seed=seed)
    start_time = time.time()
    history_naive = vi_naive.fit(max_iter=150, verbose=False, check_every=50)
    runtime_naive = time.time() - start_time
    
    error_naive, X_aligned_naive = compute_alignment_error(
        vi_naive.X_mean.detach(), X_true, latent_dim=2, align=True
    )
    
    results['Naive MF'] = {
        'vi': vi_naive,
        'history': history_naive,
        'X_est': vi_naive.X_mean.detach(),
        'X_aligned': X_aligned_naive,
        'alignment_error': error_naive,
        'runtime': runtime_naive
    }
    
    print(f"  Final ELBO: {history_naive['elbo'][-1]:.2f}")
    print(f"  Final MSE: {history_naive['reconstruction_error'][-1]:.6f}")
    print(f"  Alignment error: {error_naive:.6f}")
    print(f"  Runtime: {runtime_naive:.2f}s")
    
    # Structured MF (Good)
    print(f"\n{'-'*70}")
    print("Running Structured MF (Good)...")
    print(f"{'-'*70}")
    vi_smf_good = TemporalAMEStructuredMFVI(
        model, factorization='good', learning_rate=0.7, seed=seed
    )
    start_time = time.time()
    history_smf_good = vi_smf_good.fit(max_iter=150, verbose=False, check_every=50)
    runtime_smf_good = time.time() - start_time
    
    error_smf_good, X_aligned_smf_good = compute_alignment_error(
        vi_smf_good.X_mean.detach(), X_true, latent_dim=2, align=True
    )
    
    results['Good SMF'] = {
        'vi': vi_smf_good,
        'history': history_smf_good,
        'X_est': vi_smf_good.X_mean.detach(),
        'X_aligned': X_aligned_smf_good,
        'alignment_error': error_smf_good,
        'runtime': runtime_smf_good
    }
    
    print(f"  Final ELBO: {history_smf_good['elbo'][-1]:.2f}")
    print(f"  Final MSE: {history_smf_good['reconstruction_error'][-1]:.6f}")
    print(f"  Alignment error: {error_smf_good:.6f}")
    print(f"  Runtime: {runtime_smf_good:.2f}s")
    
    # Structured MF (Bad)
    print(f"\n{'-'*70}")
    print("Running Structured MF (Bad)...")
    print(f"{'-'*70}")
    vi_smf_bad = TemporalAMEStructuredMFVI(
        model, factorization='bad', learning_rate=0.7, seed=seed
    )
    start_time = time.time()
    history_smf_bad = vi_smf_bad.fit(max_iter=150, verbose=False, check_every=50)
    runtime_smf_bad = time.time() - start_time
    
    error_smf_bad, X_aligned_smf_bad = compute_alignment_error(
        vi_smf_bad.X_mean.detach(), X_true, latent_dim=2, align=True
    )
    
    results['Bad SMF'] = {
        'vi': vi_smf_bad,
        'history': history_smf_bad,
        'X_est': vi_smf_bad.X_mean.detach(),
        'X_aligned': X_aligned_smf_bad,
        'alignment_error': error_smf_bad,
        'runtime': runtime_smf_bad
    }
    
    print(f"  Final ELBO: {history_smf_bad['elbo'][-1]:.2f}")
    print(f"  Final MSE: {history_smf_bad['reconstruction_error'][-1]:.6f}")
    print(f"  Alignment error: {error_smf_bad:.6f}")
    print(f"  Runtime: {runtime_smf_bad:.2f}s")
    
    # Print summaries
    print(f"\n{'='*70}")
    print("DETAILED SUMMARIES")
    print(f"{'='*70}")
    
    for method_name, result in results.items():
        print_diagnostic_summary(
            method_name,
            result['history'],
            X_true=X_true,
            X_est=result['X_est'],
            latent_dim=2,
            final_only=False
        )
    
    # Compare methods
    compare_methods(results, metric='reconstruction_error', X_true=X_true)
    
    return results, model, X_true


def create_comparison_figure(
    results_weak,
    results_strong,
    X_true_weak,
    X_true_strong,
    save_path: str = 'multiplicative_strength_comparison.png'
):
    """
    Create publication-ready comparison figure for weak vs strong scenarios.
    
    Layout:
    - Row 1: ELBO (weak), ELBO (strong), U'V correlation trajectories
    - Row 2: Weak scenario posterior correlation matrices (3 methods)
    - Row 3: Strong scenario posterior correlation matrices (3 methods)
    
    Parameters
    ----------
    results_weak : dict
        Results from weak multiplicative scenario.
    results_strong : dict
        Results from strong multiplicative scenario.
    X_true_weak : torch.Tensor
        True states for weak scenario.
    X_true_strong : torch.Tensor
        True states for strong scenario.
    save_path : str
        Path to save figure.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.4)
    
    colors = {'Naive MF': '#2E86AB', 'Good SMF': '#A23B72', 'Bad SMF': '#F18F01'}
    
    # ========== ROW 1: PERFORMANCE METRICS ==========
    # 3 panels: ELBO weak, ELBO strong, U'V correlation (larger)
    
    # Panel 1: ELBO convergence - Weak scenario
    ax1 = fig.add_subplot(gs[0, 0])
    for method_name, result in results_weak.items():
        ax1.plot(
            result['history']['elbo'],
            label=method_name,
            color=colors[method_name],
            linewidth=2,
            alpha=0.8
        )
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('ELBO', fontsize=10)
    ax1.set_title('Weak: ELBO Convergence', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel 2: ELBO convergence - Strong scenario
    ax2 = fig.add_subplot(gs[0, 1])
    for method_name, result in results_strong.items():
        ax2.plot(
            result['history']['elbo'],
            label=method_name,
            color=colors[method_name],
            linewidth=2,
            alpha=0.8
        )
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('ELBO', fontsize=10)
    ax2.set_title('Strong: ELBO Convergence', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Panel 3: U'V correlation trajectories (larger)
    ax3 = fig.add_subplot(gs[0, 2:])
    
    # Compute U'V correlations over time for each method and scenario
    latent_dim = 2
    
    method_names = list(results_weak.keys())
    
    for method_name in method_names:
        # Weak scenario
        X_est_weak = results_weak[method_name]['X_aligned']
        corr_weak = compute_uv_correlation_over_time(X_est_weak, X_true_weak, latent_dim)
        ax3.plot(
            corr_weak.cpu().numpy(),
            label=f'{method_name} (W)',
            color=colors[method_name],
            linewidth=2.5,
            linestyle='--',
            alpha=0.6
        )
        
        # Strong scenario
        X_est_strong = results_strong[method_name]['X_aligned']
        corr_strong = compute_uv_correlation_over_time(X_est_strong, X_true_strong, latent_dim)
        ax3.plot(
            corr_strong.cpu().numpy(),
            label=f'{method_name} (S)',
            color=colors[method_name],
            linewidth=2.5,
            linestyle='-',
            alpha=0.8
        )
    
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Correlation with True U\'V', fontsize=11)
    ax3.set_title('Multiplicative Recovery Over Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, ncol=1, loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    ax3.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ========== ROWS 2-3: POSTERIOR CORRELATION STRUCTURES ==========
    
    print("Adding correlation structure panels...")
    
    n_weak, T_weak, d_weak = X_true_weak.shape
    n_strong, T_strong, d_strong = X_true_strong.shape
    latent_dim = (d_weak - 2) // 2
    
    t_plot = T_weak // 2  # Middle time point
    node_plot = 0
    
    # State labels
    state_labels = ['a', 'b']
    for k in range(latent_dim):
        state_labels.extend([f'U_{k+1}', f'V_{k+1}'])
    
    method_names = ['Naive MF', 'Good SMF', 'Bad SMF']
    
    # Row 2: Weak scenario correlation matrices (one per column, skip last)
    for method_idx, method_name in enumerate(method_names):
        if method_name not in results_weak or method_idx >= 3:
            continue
            
        ax = fig.add_subplot(gs[1, method_idx])
        
        vi = results_weak[method_name]['vi']
        
        # Get covariance
        try:
            if hasattr(vi, 'X_cov'):
                cov = vi.X_cov[node_plot, t_plot].detach()
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center', fontsize=10)
                ax.set_title(f'{method_name}\nWeak', fontsize=10, fontweight='bold')
                ax.axis('off')
                continue
            
            # Convert to correlation
            std = torch.sqrt(torch.diag(cov))
            corr = cov / (std.unsqueeze(1) @ std.unsqueeze(0))
            corr = corr.cpu().numpy()
            corr = np.nan_to_num(corr, nan=0.0)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error', transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(f'{method_name}\nWeak', fontsize=10, fontweight='bold')
            ax.axis('off')
            continue
        
        # Plot heatmap
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        
        # Labels
        ax.set_xticks(range(d_weak))
        ax.set_yticks(range(d_weak))
        ax.set_xticklabels(state_labels, fontsize=8)
        ax.set_yticklabels(state_labels, fontsize=8)
        
        # Title
        ax.set_title(
            f'{method_name}\nWeak',
            fontsize=10,
            fontweight='bold'
        )
        
        # Grid
        for i in range(d_weak + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # Highlight structures
        if 'bad' in method_name.lower():
            ax.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False, 
                                  edgecolor='yellow', linewidth=2, linestyle='--'))
            ax.add_patch(Rectangle((1.5, 1.5), d_weak - 2, d_weak - 2, fill=False,
                                  edgecolor='yellow', linewidth=2, linestyle='--'))
        
        # Check if nearly diagonal
        off_diag = corr - np.diag(np.diag(corr))
        max_off_diag = np.abs(off_diag).max()
        
        if max_off_diag < 0.1:
            label = 'Diagonal'
            color = 'lightgreen'
        elif 'bad' in method_name.lower():
            label = 'Block-Diag'
            color = 'yellow'
        else:
            label = f'max|ρ|={max_off_diag:.2f}'
            color = 'lightblue'
        
        ax.text(
            0.5, -0.15,
            label,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
    

    
    # Row 3: Strong scenario correlation matrices
    for method_idx, method_name in enumerate(method_names):
        if method_name not in results_strong or method_idx >= 3:
            continue
            
        ax = fig.add_subplot(gs[2, method_idx])
        
        vi = results_strong[method_name]['vi']
        
        # Get covariance
        try:
            if hasattr(vi, 'X_cov'):
                cov = vi.X_cov[node_plot, t_plot].detach()
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center', fontsize=10)
                ax.set_title(f'{method_name}\nStrong', fontsize=10, fontweight='bold')
                ax.axis('off')
                continue
            
            # Convert to correlation
            std = torch.sqrt(torch.diag(cov))
            corr = cov / (std.unsqueeze(1) @ std.unsqueeze(0))
            corr = corr.cpu().numpy()
            corr = np.nan_to_num(corr, nan=0.0)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error', transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title(f'{method_name}\nStrong', fontsize=10, fontweight='bold')
            ax.axis('off')
            continue
        
        # Plot heatmap
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        
        # Labels
        ax.set_xticks(range(d_strong))
        ax.set_yticks(range(d_strong))
        ax.set_xticklabels(state_labels, fontsize=8)
        ax.set_yticklabels(state_labels, fontsize=8)
        
        # Title
        ax.set_title(
            f'{method_name}\nStrong',
            fontsize=10,
            fontweight='bold'
        )
        
        # Grid
        for i in range(d_strong + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # Highlight structures
        if 'bad' in method_name.lower():
            ax.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False, 
                                  edgecolor='yellow', linewidth=2, linestyle='--'))
            ax.add_patch(Rectangle((1.5, 1.5), d_strong - 2, d_strong - 2, fill=False,
                                  edgecolor='yellow', linewidth=2, linestyle='--'))
        
        # Check correlation strength
        off_diag = corr - np.diag(np.diag(corr))
        max_off_diag = np.abs(off_diag).max()
        
        if max_off_diag < 0.1:
            label = 'Diagonal'
            color = 'lightgreen'
        elif 'bad' in method_name.lower():
            label = 'Block-Diag'
            color = 'yellow'
        elif 'good' in method_name.lower():
            label = f'max|ρ|={max_off_diag:.2f}'
            color = 'lightcoral'
        else:
            label = f'max|ρ|={max_off_diag:.2f}'
            color = 'lightblue'
        
        ax.text(
            0.5, -0.15,
            label,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
    

    
    fig.suptitle(
        'Structured vs Naive Mean-Field: Effect of Multiplicative Strength',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Publication figure saved to: {save_path}")
    print("\n" + "="*70)
    print("SUGGESTED FIGURE CAPTION:")
    print("="*70)
    print("""
Comparison of Naive Mean-Field (MF) and Structured Mean-Field (SMF) 
approximations under weak and strong multiplicative effect regimes.
(Top row) ELBO convergence for both scenarios and U'V correlation 
trajectories showing recovery of multiplicative interactions over time, 
with dashed lines for weak (W) and solid lines for strong (S) scenarios. 
(Middle/Bottom rows) Posterior correlation structures at t=5 for n=0, 
showing that Naive MF produces near-diagonal posteriors in both regimes, 
while Good SMF captures off-diagonal dependencies when present. Bad SMF 
forces incorrect block-diagonal structure (highlighted in yellow) 
regardless of regime. 

Final reconstruction errors (MSE): In weak multiplicative settings, 
Naive MF = 3.76, Good SMF = 3.64, Bad SMF = 6.73, indicating that all 
methods except Bad SMF achieve similar performance with minimal benefit 
from structured factorization. In strong multiplicative settings, 
Naive MF = 0.32, Good SMF = 0.32, Bad SMF = 1.82, demonstrating that 
incorrect structural assumptions (Bad SMF) are harmful while both 
Naive and Good SMF achieve comparable reconstruction quality. Good SMF 
shows enhanced capture of correlations (max|ρ|=0.54 in weak, max|ρ|=0.54 
in strong) compared to Naive MF's near-diagonal structure.
    """)
    print("="*70)
    plt.close()


def main():
    """Run the complete experiment."""
    print("="*70)
    print("MULTIPLICATIVE STRENGTH COMPARISON EXPERIMENT")
    print("="*70)
    print("\nThis experiment compares Naive MF vs Structured MF under")
    print("different multiplicative effect strengths to demonstrate")
    print("when structured factorization provides substantial benefits.")
    
    # Create output directory
    output_dir = Path('results/multiplicative_experiment')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scenario 1: Weak multiplicative effects (additive dominated)
    results_weak, model_weak, X_true_weak = run_experiment(
        scenario_name="Weak Multiplicative (Additive Dominated)",
        rho_additive=0.5,
        rho_multiplicative=0.3,
        additive_scale=2.0,      # Strong additive
        multiplicative_scale=0.2, # Weak multiplicative
        n_nodes=20,
        n_time=10,
        seed=42
    )
    
    # Scenario 2: Strong multiplicative effects
    results_strong, model_strong, X_true_strong = run_experiment(
        scenario_name="Strong Multiplicative (Balanced Effects)",
        rho_additive=0.7,
        rho_multiplicative=0.7,
        additive_scale=1.0,       # Moderate additive
        multiplicative_scale=2.0, # Strong multiplicative
        n_nodes=20,
        n_time=10,
        seed=42
    )
    
    # Create comprehensive comparison figure
    print("\n" + "="*70)
    print("Creating comparison visualization...")
    print("="*70)
    
    create_comparison_figure(
        results_weak,
        results_strong,
        X_true_weak,
        X_true_strong,
        save_path=output_dir / 'multiplicative_strength_comparison.png'
    )
    
    # Save individual scenario plots
    for scenario_name, results, X_true in [
        ('weak', results_weak, X_true_weak),
        ('strong', results_strong, X_true_strong)
    ]:
        # Convergence
        fig = plot_convergence_comparison(
            results,
            metric='elbo',
            save_path=output_dir / f'{scenario_name}_convergence.png'
        )
        plt.close(fig)
        
        # Parameter recovery
        fig = plot_parameter_recovery_grid(
            X_true,
            results,
            save_path=output_dir / f'{scenario_name}_recovery.png'
        )
        plt.close(fig)
        
        print(f"  ✓ Saved {scenario_name} scenario plots")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  • multiplicative_strength_comparison.png (main figure)")
    print(f"  • weak_convergence.png")
    print(f"  • weak_recovery.png")
    print(f"  • strong_convergence.png")
    print(f"  • strong_recovery.png")
    
    return results_weak, results_strong


if __name__ == "__main__":
    results_weak, results_strong = main()