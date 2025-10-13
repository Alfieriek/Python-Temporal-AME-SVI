"""
Quick Start Demo: Temporal AME with Structured Variational Inference

This script demonstrates the complete workflow:
1. Generate synthetic temporal network data
2. Fit models with different inference methods
3. Visualize and compare results

Usage
-----
python demo.py

Author: Sean Plummer
Date: October 2025
"""

import torch
import matplotlib.pyplot as plt

# Import models, inference, and utilities
from src.models import TemporalAMEModel
from src.inference import TemporalAMENaiveMFVI, TemporalAMEStructuredMFVI
from src.utils import (
    print_diagnostic_summary,
    compare_methods,
    align_temporal_states,
    compute_temporal_contributions
)
from src.visualization import (
    plot_convergence,
    plot_state_trajectories,
    plot_temporal_contributions,
    plot_three_way_comparison
)


def main():
    """Run complete demo workflow."""
    
    print("\n" + "="*70)
    print("TEMPORAL AME MODEL: QUICK START DEMO")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # STEP 1: Generate Synthetic Data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Generating synthetic temporal network data...")
    
    model = TemporalAMEModel(
        n_nodes=15,          # 15 nodes
        n_time=10,           # 10 time steps
        latent_dim=2,        # 2D latent space
        ar_coefficient=0.8,  # Strong temporal dependence
        rho_dyadic=0.5,      # Moderate reciprocity
        seed=42
    )
    
    # Generate data and store true parameters
    Y, X_true = model.generate_data(return_latents=True)
    
    print(f"✓ Generated network data:")
    print(f"  - Shape: {Y.shape} (n_nodes × n_nodes × n_time × 2)")
    print(f"  - True latent states: {X_true.shape} (n_nodes × n_time × state_dim)")
    
    # -------------------------------------------------------------------------
    # STEP 2: Naive Mean-Field VI (Baseline)
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Running Naive Mean-Field VI (baseline)...")
    
    vi_naive = TemporalAMENaiveMFVI(
        model,
        learning_rate=0.01
    )
    
    history_naive = vi_naive.fit(
        max_iter=100,
        verbose=False,
        check_every=20
    )
    
    X_est_naive = vi_naive.get_variational_means()
    print(f"✓ Naive MF completed:")
    print(f"  - Final ELBO: {history_naive['elbo'][-1]:.2f}")
    print(f"  - Final MSE: {history_naive['reconstruction_error'][-1]:.6f}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Structured Mean-Field VI (Good Factorization)
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Running Structured MF VI (good factorization)...")
    
    vi_good = TemporalAMEStructuredMFVI(
        model,
        factorization="good",  # Correct block structure
        learning_rate=0.01
    )
    
    history_good = vi_good.fit(
        max_iter=100,
        verbose=False,
        check_every=20
    )
    
    X_est_good = vi_good.get_variational_means()
    print(f"✓ Good SMF completed:")
    print(f"  - Final ELBO: {history_good['elbo'][-1]:.2f}")
    print(f"  - Final MSE: {history_good['reconstruction_error'][-1]:.6f}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Structured Mean-Field VI (Bad Factorization)
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Running Structured MF VI (bad factorization)...")
    
    vi_bad = TemporalAMEStructuredMFVI(
        model,
        factorization="bad",   # Wrong block structure
        learning_rate=0.01
    )
    
    history_bad = vi_bad.fit(
        max_iter=100,
        verbose=False,
        check_every=20
    )
    
    X_est_bad = vi_bad.get_variational_means()
    print(f"✓ Bad SMF completed:")
    print(f"  - Final ELBO: {history_bad['elbo'][-1]:.2f}")
    print(f"  - Final MSE: {history_bad['reconstruction_error'][-1]:.6f}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Align Estimates with True Parameters
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Aligning estimates with true parameters...")
    
    X_aligned_naive = align_temporal_states(X_est_naive, X_true, latent_dim=2)
    X_aligned_good = align_temporal_states(X_est_good, X_true, latent_dim=2)
    X_aligned_bad = align_temporal_states(X_est_bad, X_true, latent_dim=2)
    
    print("✓ Parameter alignment completed")
    
    # -------------------------------------------------------------------------
    # STEP 6: Print Diagnostic Summaries
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Diagnostic summaries...")
    
    results = {
        'Naive MF': {
            'history': history_naive,
            'X_est': X_aligned_naive
        },
        'Good SMF': {
            'history': history_good,
            'X_est': X_aligned_good
        },
        'Bad SMF': {
            'history': history_bad,
            'X_est': X_aligned_bad
        }
    }
    
    for method_name, result in results.items():
        print_diagnostic_summary(
            method_name,
            result['history'],
            X_true=X_true,
            X_est=result['X_est'],
            latent_dim=2,
            final_only=True
        )
    
    # -------------------------------------------------------------------------
    # STEP 7: Compare Methods
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Method comparison...")
    
    compare_methods(results, metric='reconstruction_error', X_true=X_true)
    
    # -------------------------------------------------------------------------
    # STEP 8: Visualizations
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Creating visualizations...")
    
    # Convergence comparison
    print("  - Creating convergence plot...")
    fig1 = plot_convergence(
        history_good,
        title="Good SMF: Convergence"
    )
    plt.show(block=False)
    
    # State trajectories
    print("  - Creating trajectory plot...")
    fig2 = plot_state_trajectories(
        X_aligned_good,
        node_indices=[0, 1, 2],
        state_indices=[0, 1, 2, 3],  # a, b, U_1, U_2
        title="Good SMF: State Trajectories (Nodes 0-2)"
    )
    plt.show(block=False)
    
    # Temporal contributions
    print("  - Creating contribution plot...")
    add_contrib, mult_contrib = compute_temporal_contributions(
        X_aligned_good,
        latent_dim=2
    )
    fig3 = plot_temporal_contributions(
        add_contrib,
        mult_contrib,
        title="Good SMF: Effect Contributions Over Time"
    )
    plt.show(block=False)
    
    # Three-way comparison
    print("  - Creating comprehensive comparison...")
    fig4 = plot_three_way_comparison(
        results,
        X_true=X_true
    )
    plt.show(block=False)
    
    # -------------------------------------------------------------------------
    # STEP 9: Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    
    # Extract final MSEs
    mse_naive = history_naive['reconstruction_error'][-1]
    mse_good = history_good['reconstruction_error'][-1]
    mse_bad = history_bad['reconstruction_error'][-1]
    
    print("\nKey Findings:")
    print(f"1. Naive MF MSE:  {mse_naive:.6f}")
    print(f"2. Good SMF MSE:  {mse_good:.6f} ({(1-mse_good/mse_naive)*100:+.1f}%)")
    print(f"3. Bad SMF MSE:   {mse_bad:.6f} ({(1-mse_bad/mse_naive)*100:+.1f}%)")
    
    if mse_good < mse_naive:
        print("\n✓ Good SMF outperforms Naive MF (structured factorization helps!)")
    
    if mse_bad > mse_naive:
        print("✓ Bad SMF performs worse than Naive MF (wrong structure hurts!)")
    
    print("\n" + "="*70)
    print("\nFour figures have been created. Close them to exit.")
    print("\nTo run full experiments with saved outputs, try:")
    print("  python experiments/three_way_comparison.py")
    print("  python experiments/sensitivity_analysis.py")
    print("="*70 + "\n")
    
    # Keep figures open
    plt.show()


if __name__ == "__main__":
    main()