"""
Sensitivity Analysis: Impact of Problem Parameters on Method Performance.

This experiment examines how different problem characteristics affect the
performance of Naive MF vs Structured MF:
- Network size (n_nodes)
- Temporal length (n_time)
- AR coefficient (temporal dependence)
- Dyadic correlation (reciprocity)

Usage
-----
python experiments/sensitivity_analysis.py

Author: Sean Plummer
Date: October 2025
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models import TemporalAMEModel
from src.inference import TemporalAMENaiveMFVI, TemporalAMEStructuredMFVI
from src.utils import align_temporal_states
from utils import (
    setup_experiment_dir,
    save_results,
    run_method_with_timing,
    set_random_seeds,
    print_experiment_header
)


def run_sensitivity_analysis(
    parameter: str = 'n_nodes',
    values: list = [10, 15, 20, 25, 30],
    base_params: dict = None,
    max_iter: int = 150,
    learning_rate: float = 0.01,
    n_replicates: int = 3,
    seed: int = 42,
    save_outputs: bool = True
):
    """
    Run sensitivity analysis over a parameter range.
    
    Parameters
    ----------
    parameter : str, default='n_nodes'
        Which parameter to vary. Options: 'n_nodes', 'n_time', 'ar_coefficient', 
        'rho_dyadic'.
    values : list, default=[10, 15, 20, 25, 30]
        Values to test for the parameter.
    base_params : dict, optional
        Base parameters for the experiment. If None, uses defaults.
    max_iter : int, default=150
        Maximum optimization iterations.
    learning_rate : float, default=0.01
        Learning rate for inference.
    n_replicates : int, default=3
        Number of replicates per parameter value.
    seed : int, default=42
        Random seed for reproducibility.
    save_outputs : bool, default=True
        Whether to save results.
        
    Returns
    -------
    results : dict
        Dictionary containing results for each parameter value.
    exp_dir : Path or None
        Experiment directory (if save_outputs=True).
    """
    # Default base parameters
    if base_params is None:
        base_params = {
            'n_nodes': 15,
            'n_time': 10,
            'latent_dim': 2,
            'ar_coefficient': 0.8,
            'rho_dyadic': 0.5
        }
    
    # Print experiment header
    params = base_params.copy()
    params.update({
        'varied_parameter': parameter,
        'parameter_values': values,
        'max_iter': max_iter,
        'n_replicates': n_replicates,
        'seed': seed
    })
    print_experiment_header(f"Sensitivity Analysis: {parameter}", params)
    
    # Setup output directory
    exp_dir = None
    if save_outputs:
        exp_dir = setup_experiment_dir(f"sensitivity_{parameter}")
        print(f"Results will be saved to: {exp_dir}\n")
    
    # Storage for results
    results = {
        'parameter': parameter,
        'values': values,
        'base_params': base_params,
        'data': {}
    }
    
    # Run experiments for each parameter value
    for value in values:
        print("\n" + "="*70)
        print(f"{parameter.upper()} = {value}")
        print("="*70)
        
        # Update parameters
        current_params = base_params.copy()
        current_params[parameter] = value
        
        # Storage for this parameter value
        value_results = {
            'Naive MF': {'mse': [], 'runtime': [], 'iterations': []},
            'Good SMF': {'mse': [], 'runtime': [], 'iterations': []}
        }
        
        # Run replicates
        for rep in range(n_replicates):
            rep_seed = seed + rep
            set_random_seeds(rep_seed)
            
            print(f"\n--- Replicate {rep + 1}/{n_replicates} ---")
            
            # Generate data
            model = TemporalAMEModel(
                n_nodes=current_params['n_nodes'],
                n_time=current_params['n_time'],
                latent_dim=current_params['latent_dim'],
                ar_coefficient=current_params['ar_coefficient'],
                rho_dyadic=current_params['rho_dyadic'],
                seed=rep_seed
            )
            Y, X_true = model.generate_data(return_latents=True)
            
            # Naive MF
            print("\nNaive MF...")
            result_naive = run_method_with_timing(
                TemporalAMENaiveMFVI,
                model,
                "Naive MF",
                max_iter=max_iter,
                learning_rate=learning_rate,
                verbose=False
            )
            
            final_mse_naive = result_naive['history']['reconstruction_error'][-1]
            value_results['Naive MF']['mse'].append(final_mse_naive)
            value_results['Naive MF']['runtime'].append(result_naive['runtime'])
            value_results['Naive MF']['iterations'].append(result_naive['iterations'])
            print(f"  MSE: {final_mse_naive:.6f}, Runtime: {result_naive['runtime']:.2f}s")
            
            # Good Structured MF
            print("Good SMF...")
            result_good = run_method_with_timing(
                TemporalAMEStructuredMFVI,
                model,
                "Good SMF",
                max_iter=max_iter,
                learning_rate=learning_rate,
                factorization="good",
                verbose=False
            )
            
            final_mse_good = result_good['history']['reconstruction_error'][-1]
            value_results['Good SMF']['mse'].append(final_mse_good)
            value_results['Good SMF']['runtime'].append(result_good['runtime'])
            value_results['Good SMF']['iterations'].append(result_good['iterations'])
            print(f"  MSE: {final_mse_good:.6f}, Runtime: {result_good['runtime']:.2f}s")
        
        # Compute statistics
        for method in ['Naive MF', 'Good SMF']:
            value_results[method]['mse_mean'] = np.mean(value_results[method]['mse'])
            value_results[method]['mse_std'] = np.std(value_results[method]['mse'])
            value_results[method]['runtime_mean'] = np.mean(value_results[method]['runtime'])
            value_results[method]['runtime_std'] = np.std(value_results[method]['runtime'])
        
        # Print summary for this value
        print(f"\nSummary for {parameter}={value}:")
        for method in ['Naive MF', 'Good SMF']:
            mse_mean = value_results[method]['mse_mean']
            mse_std = value_results[method]['mse_std']
            print(f"  {method}: MSE = {mse_mean:.6f} ± {mse_std:.6f}")
        
        # Compute improvement
        naive_mean = value_results['Naive MF']['mse_mean']
        good_mean = value_results['Good SMF']['mse_mean']
        if naive_mean > 0:
            improvement = (1 - good_mean / naive_mean) * 100
            print(f"  Improvement: {improvement:+.1f}%")
        
        results['data'][value] = value_results
    
    # Generate summary plots
    if save_outputs:
        print("\n" + "="*70)
        print("GENERATING SUMMARY PLOTS")
        print("="*70)
        
        figures_dir = exp_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Extract data for plotting
        naive_mse_means = [results['data'][v]['Naive MF']['mse_mean'] for v in values]
        naive_mse_stds = [results['data'][v]['Naive MF']['mse_std'] for v in values]
        good_mse_means = [results['data'][v]['Good SMF']['mse_mean'] for v in values]
        good_mse_stds = [results['data'][v]['Good SMF']['mse_std'] for v in values]
        
        naive_runtime_means = [results['data'][v]['Naive MF']['runtime_mean'] for v in values]
        good_runtime_means = [results['data'][v]['Good SMF']['runtime_mean'] for v in values]
        
        # Plot 1: MSE comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(
            values,
            naive_mse_means,
            yerr=naive_mse_stds,
            marker='o',
            markersize=8,
            linewidth=2,
            capsize=5,
            label='Naive MF',
            color='#2E86AB'
        )
        ax.errorbar(
            values,
            good_mse_means,
            yerr=good_mse_stds,
            marker='s',
            markersize=8,
            linewidth=2,
            capsize=5,
            label='Good SMF',
            color='#A23B72'
        )
        
        ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Reconstruction MSE', fontsize=12)
        ax.set_title(f'Sensitivity Analysis: {parameter.replace("_", " ").title()}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"sensitivity_{parameter}_mse.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved MSE plot")
        
        # Plot 2: Runtime comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        width = (values[-1] - values[0]) / len(values) * 0.35
        x_naive = np.array(values) - width/2
        x_good = np.array(values) + width/2
        
        ax.bar(x_naive, naive_runtime_means, width, label='Naive MF', 
               color='#2E86AB', alpha=0.8, edgecolor='black')
        ax.bar(x_good, good_runtime_means, width, label='Good SMF',
               color='#A23B72', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Runtime (seconds)', fontsize=12)
        ax.set_title(f'Runtime Comparison: {parameter.replace("_", " ").title()}',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"sensitivity_{parameter}_runtime.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved runtime plot")
        
        # Plot 3: Improvement percentage
        improvements = []
        for v in values:
            naive_mean = results['data'][v]['Naive MF']['mse_mean']
            good_mean = results['data'][v]['Good SMF']['mse_mean']
            if naive_mean > 0:
                improvement = (1 - good_mean / naive_mean) * 100
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(values, improvements, color='#06A77D', alpha=0.8, edgecolor='black')
        ax.axhline(0, color='black', linewidth=0.8)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{imp:+.1f}%',
                ha='center',
                va='bottom' if imp >= 0 else 'top',
                fontsize=9
            )
        
        ax.set_xlabel(parameter.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title(f'Good SMF Improvement over Naive MF: {parameter.replace("_", " ").title()}',
                    fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"sensitivity_{parameter}_improvement.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved improvement plot")
        
        print(f"\n✓ All figures saved to {figures_dir}")
        
        # Save results
        save_results(results, exp_dir, f"sensitivity_{parameter}.pkl")
        
        # Generate summary report
        _generate_sensitivity_report(results, exp_dir, parameter)
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    
    if save_outputs:
        print(f"\n✓ All results saved to: {exp_dir}")
        print(f"✓ View report at: {exp_dir / 'report.md'}")
    
    return results, exp_dir


def _generate_sensitivity_report(results, exp_dir, parameter):
    """Generate markdown report for sensitivity analysis."""
    report_path = exp_dir / "report.md"
    
    from datetime import datetime
    
    with open(report_path, 'w') as f:
        f.write(f"# Sensitivity Analysis: {parameter.replace('_', ' ').title()}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- **Varied Parameter:** {parameter}\n")
        f.write(f"- **Values Tested:** {results['values']}\n")
        f.write(f"- **Base Parameters:**\n")
        for key, value in results['base_params'].items():
            if key != parameter:
                f.write(f"  - {key}: {value}\n")
        
        f.write("\n## Results Summary\n\n")
        f.write("| Value | Naive MF MSE | Good SMF MSE | Improvement (%) |\n")
        f.write("|-------|--------------|--------------|------------------|\n")
        
        for value in results['values']:
            naive_mean = results['data'][value]['Naive MF']['mse_mean']
            good_mean = results['data'][value]['Good SMF']['mse_mean']
            improvement = (1 - good_mean / naive_mean) * 100 if naive_mean > 0 else 0
            
            f.write(f"| {value} | {naive_mean:.6f} | {good_mean:.6f} | {improvement:+.1f}% |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Find best/worst improvement
        improvements = []
        for value in results['values']:
            naive_mean = results['data'][value]['Naive MF']['mse_mean']
            good_mean = results['data'][value]['Good SMF']['mse_mean']
            improvement = (1 - good_mean / naive_mean) * 100 if naive_mean > 0 else 0
            improvements.append((value, improvement))
        
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"- **Best improvement:** {improvements[0][1]:+.1f}% at {parameter}={improvements[0][0]}\n")
        f.write(f"- **Worst improvement:** {improvements[-1][1]:+.1f}% at {parameter}={improvements[-1][0]}\n")
        
        avg_improvement = np.mean([imp for _, imp in improvements])
        f.write(f"- **Average improvement:** {avg_improvement:+.1f}%\n")
        
        f.write("\n## Figures\n\n")
        f.write("- `sensitivity_{}_mse.png`: MSE comparison across parameter values\n".format(parameter))
        f.write("- `sensitivity_{}_runtime.png`: Runtime comparison\n".format(parameter))
        f.write("- `sensitivity_{}_improvement.png`: Improvement percentages\n".format(parameter))
        
        f.write("\n---\n\n*Report generated automatically*\n")
    
    print(f"✓ Sensitivity report saved to: {report_path}")


if __name__ == "__main__":
    # Run sensitivity analysis on network size
    results, exp_dir = run_sensitivity_analysis(
        parameter='n_nodes',
        values=[10, 15, 20, 25, 30],
        max_iter=150,
        n_replicates=3,
        seed=42,
        save_outputs=True
    )
    
    print("\n✓ Sensitivity analysis completed successfully!")