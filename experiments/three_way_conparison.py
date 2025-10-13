"""
Three-Way Comparison Experiment: Naive MF vs Good SMF vs Bad SMF.

This experiment demonstrates the importance of correct factorization design
by comparing three inference methods:
1. Naive MF: Fully factorized (baseline)
2. Good SMF: Correct blocks [a_i, b_i, U_i, V_i] together
3. Bad SMF: Wrong blocks with [U_i, V_i] independent from [a_i, b_i]

The goal is to show that:
- Structured MF can substantially improve over naive MF
- Incorrect factorization (Bad SMF) can perform worse than naive MF
- Proper block structure matters

Usage
-----
python experiments/three_way_comparison.py

Author: Sean Plummer
Date: October 2025
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt

from src.models import TemporalAMEModel
from src.inference import TemporalAMENaiveMFVI, TemporalAMEStructuredMFVI
from src.utils import (
    print_diagnostic_summary,
    compare_methods,
    align_temporal_states,
    compute_temporal_contributions
)
from src.visualization import (
    plot_three_way_comparison,
    plot_convergence_comparison,
    plot_temporal_contributions
)
from utils import (
    setup_experiment_dir,
    save_results,
    run_method_with_timing,
    generate_experiment_report,
    set_random_seeds,
    print_experiment_header
)


def run_three_way_comparison(
    n_nodes: int = 15,
    n_time: int = 10,
    latent_dim: int = 2,
    rho_dyadic: float = 0.5,
    ar_coefficient: float = 0.8,
    max_iter: int = 200,
    learning_rate: float = 0.01,
    seed: int = 42,
    save_outputs: bool = True
):
    """
    Run three-way comparison experiment.
    
    Parameters
    ----------
    n_nodes : int, default=15
        Number of nodes in network.
    n_time : int, default=10
        Number of time steps.
    latent_dim : int, default=2
        Latent space dimension.
    rho_dyadic : float, default=0.5
        Dyadic error correlation.
    ar_coefficient : float, default=0.8
        AR(1) coefficient for temporal dynamics.
    max_iter : int, default=200
        Maximum optimization iterations.
    learning_rate : float, default=0.01
        Learning rate for all methods.
    seed : int, default=42
        Random seed for reproducibility.
    save_outputs : bool, default=True
        Whether to save results and figures.
        
    Returns
    -------
    results : dict
        Dictionary containing results for all three methods.
    exp_dir : Path or None
        Experiment directory (if save_outputs=True).
    """
    # Set random seed
    set_random_seeds(seed)
    
    # Print experiment header
    params = {
        'n_nodes': n_nodes,
        'n_time': n_time,
        'latent_dim': latent_dim,
        'rho_dyadic': rho_dyadic,
        'ar_coefficient': ar_coefficient,
        'max_iter': max_iter,
        'learning_rate': learning_rate,
        'seed': seed
    }
    print_experiment_header("Three-Way Comparison", params)
    
    # Setup output directory
    exp_dir = None
    if save_outputs:
        exp_dir = setup_experiment_dir("three_way_comparison")
        print(f"Results will be saved to: {exp_dir}\n")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    model = TemporalAMEModel(
        n_nodes=n_nodes,
        n_time=n_time,
        latent_dim=latent_dim,
        ar_coefficient=ar_coefficient,
        rho_dyadic=rho_dyadic,
        seed=seed
    )
    Y, X_true = model.generate_data(return_latents=True)
    print(f"Generated network: shape {Y.shape}")
    print(f"True states: shape {X_true.shape}\n")
    
    # Run three methods
    results = {}
    
    # 1. Naive Mean-Field
    print("\n" + "="*70)
    print("METHOD 1: Naive Mean-Field VI")
    print("="*70)
    result_naive = run_method_with_timing(
        TemporalAMENaiveMFVI,
        model,
        "Naive MF",
        max_iter=max_iter,
        learning_rate=learning_rate,
        verbose=True
    )
    results['Naive MF'] = result_naive
    
    # 2. Good Structured MF
    print("\n" + "="*70)
    print("METHOD 2: Structured MF (Good Factorization)")
    print("="*70)
    result_good = run_method_with_timing(
        TemporalAMEStructuredMFVI,
        model,
        "Good SMF",
        max_iter=max_iter,
        learning_rate=learning_rate,
        factorization="good",
        verbose=True
    )
    results['Good SMF'] = result_good
    
    # 3. Bad Structured MF
    print("\n" + "="*70)
    print("METHOD 3: Structured MF (Bad Factorization)")
    print("="*70)
    result_bad = run_method_with_timing(
        TemporalAMEStructuredMFVI,
        model,
        "Bad SMF",
        max_iter=max_iter,
        learning_rate=learning_rate,
        factorization="bad",
        verbose=True
    )
    results['Bad SMF'] = result_bad
    
    # Align estimates with true parameters
    print("\n" + "="*70)
    print("ALIGNING ESTIMATES WITH TRUE PARAMETERS")
    print("="*70)
    for method_name in results.keys():
        if results[method_name]['X_est'] is not None:
            X_est = results[method_name]['X_est']
            X_aligned = align_temporal_states(X_est, X_true, latent_dim)
            results[method_name]['X_aligned'] = X_aligned
            print(f"✓ Aligned {method_name}")
    
    # Print diagnostic summaries
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARIES")
    print("="*70)
    for method_name in results.keys():
        X_est = results[method_name].get('X_aligned', results[method_name]['X_est'])
        print_diagnostic_summary(
            method_name,
            results[method_name]['history'],
            X_true=X_true,
            X_est=X_est,
            latent_dim=latent_dim,
            final_only=False
        )
    
    # Compare methods
    print("\n")
    compare_methods(results, metric='reconstruction_error', X_true=X_true)
    
    # Generate visualizations
    if save_outputs:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        figures_dir = exp_dir / "figures"
        
        # Comprehensive three-way comparison
        print("Creating three-way comparison figure...")
        fig1 = plot_three_way_comparison(
            results,
            X_true=X_true,
            save_path=figures_dir / "three_way_comparison.png"
        )
        plt.close(fig1)
        
        # Convergence comparison
        print("Creating convergence comparison figures...")
        fig2 = plot_convergence_comparison(
            results,
            metric='elbo',
            title='ELBO Convergence Comparison',
            save_path=figures_dir / "convergence_elbo.png"
        )
        plt.close(fig2)
        
        fig3 = plot_convergence_comparison(
            results,
            metric='reconstruction_error',
            title='Reconstruction Error Comparison',
            save_path=figures_dir / "convergence_mse.png"
        )
        plt.close(fig3)
        
        # Temporal contributions for each method
        print("Creating temporal contribution figures...")
        for method_name in results.keys():
            X_est = results[method_name].get('X_aligned', results[method_name]['X_est'])
            if X_est is not None:
                add_contrib, mult_contrib = compute_temporal_contributions(
                    X_est,
                    latent_dim
                )
                fig = plot_temporal_contributions(
                    add_contrib,
                    mult_contrib,
                    title=f'Temporal Contributions: {method_name}',
                    save_path=figures_dir / f"contributions_{method_name.replace(' ', '_')}.png"
                )
                plt.close(fig)
        
        print(f"\n✓ All figures saved to {figures_dir}")
        
        # Save results
        print("\nSaving results...")
        save_results(results, exp_dir, "results.pkl")
        
        # Generate report
        print("Generating experiment report...")
        generate_experiment_report(
            results,
            exp_dir,
            X_true=X_true,
            experiment_name="Three-Way Comparison: Naive MF vs Good SMF vs Bad SMF"
        )
    
    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    # Extract key findings
    method_mses = []
    for method_name, result in results.items():
        history = result['history']
        if 'reconstruction_error' in history:
            final_mse = history['reconstruction_error'][-1]
            method_mses.append((method_name, final_mse))
    
    method_mses.sort(key=lambda x: x[1])
    
    print("\nKey Findings:")
    print(f"1. Best method: {method_mses[0][0]} (MSE: {method_mses[0][1]:.6f})")
    print(f"2. Worst method: {method_mses[-1][0]} (MSE: {method_mses[-1][1]:.6f})")
    
    if len(method_mses) >= 2:
        baseline_mse = method_mses[-1][1]
        best_mse = method_mses[0][1]
        improvement = (1 - best_mse / baseline_mse) * 100
        print(f"3. Improvement of best over worst: {improvement:.1f}%")
    
    # Check if bad SMF is worse than naive
    naive_rank = next((i for i, (name, _) in enumerate(method_mses) 
                      if name == 'Naive MF'), None)
    bad_rank = next((i for i, (name, _) in enumerate(method_mses) 
                    if name == 'Bad SMF'), None)
    
    if naive_rank is not None and bad_rank is not None and bad_rank > naive_rank:
        print("4. ✓ Bad SMF performed worse than Naive MF (demonstrates importance of structure)")
    
    if save_outputs:
        print(f"\n✓ All results saved to: {exp_dir}")
        print(f"✓ View report at: {exp_dir / 'report.md'}")
    
    print("="*70 + "\n")
    
    return results, exp_dir


if __name__ == "__main__":
    # Run experiment with default parameters
    results, exp_dir = run_three_way_comparison(
        n_nodes=15,
        n_time=10,
        latent_dim=2,
        rho_dyadic=0.8,
        ar_coefficient=0.8,
        max_iter=500,
        learning_rate=0.01,
        seed=42,
        save_outputs=True
    )
    
    print("\n✓ Experiment completed successfully!")
    print(f"  Run 'cat {exp_dir / 'report.md'}' to view the report")