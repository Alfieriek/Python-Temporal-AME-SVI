"""
Utility functions for experiments.

This module provides helper functions for running experiments, saving results,
and managing output directories.

Functions
---------
setup_experiment_dir
    Create directory structure for experiment outputs.
save_results
    Save experiment results to disk.
load_results
    Load previously saved results.
run_method_with_timing
    Run inference method and track runtime.
generate_experiment_report
    Generate markdown report of experiment results.

Author: Sean Plummer
Date: October 2025
"""

import os
import json
import pickle
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

import torch
import numpy as np


def setup_experiment_dir(
    experiment_name: str,
    base_dir: str = "results"
) -> Path:
    """
    Create directory structure for experiment outputs.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    base_dir : str, default="results"
        Base directory for all results.
        
    Returns
    -------
    exp_dir : Path
        Path to the created experiment directory.
        
    Examples
    --------
    >>> exp_dir = setup_experiment_dir("three_way_comparison")
    >>> print(exp_dir)  # results/three_way_comparison_20251011_120000
    """
    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    
    # Create subdirectories
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    
    return exp_dir


def save_results(
    results: Dict[str, Any],
    exp_dir: Path,
    filename: str = "results.pkl"
) -> None:
    """
    Save experiment results to disk.
    
    Parameters
    ----------
    results : dict
        Dictionary containing experiment results.
    exp_dir : Path
        Experiment directory.
    filename : str, default="results.pkl"
        Name of the output file.
        
    Notes
    -----
    Saves both pickle format (for Python) and JSON format (for readability)
    when possible.
    
    Examples
    --------
    >>> results = {'method1': {...}, 'method2': {...}}
    >>> save_results(results, exp_dir, "comparison_results.pkl")
    """
    # Save as pickle (preserves torch tensors)
    pkl_path = exp_dir / "data" / filename
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: {pkl_path}")
    
    # Try to save JSON version for readability (exclude tensors)
    try:
        json_results = _convert_to_json_serializable(results)
        json_path = exp_dir / "data" / filename.replace('.pkl', '.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"JSON summary saved to: {json_path}")
    except Exception as e:
        print(f"Could not save JSON summary: {e}")


def load_results(
    exp_dir: Path,
    filename: str = "results.pkl"
) -> Dict[str, Any]:
    """
    Load previously saved results.
    
    Parameters
    ----------
    exp_dir : Path
        Experiment directory.
    filename : str, default="results.pkl"
        Name of the results file.
        
    Returns
    -------
    results : dict
        Loaded results dictionary.
        
    Examples
    --------
    >>> results = load_results(exp_dir, "comparison_results.pkl")
    """
    pkl_path = exp_dir / "data" / filename
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    return results


def run_method_with_timing(
    vi_class,
    model,
    method_name: str,
    max_iter: int = 100,
    verbose: bool = True,
    **vi_kwargs
) -> Dict[str, Any]:
    """
    Run inference method and track runtime.
    
    Parameters
    ----------
    vi_class : class
        Variational inference class to instantiate.
    model : BaseAMEModel
        Model instance with observed data.
    method_name : str
        Name of the method (for logging).
    max_iter : int, default=100
        Maximum number of iterations.
    verbose : bool, default=True
        Whether to print progress.
    **vi_kwargs
        Additional keyword arguments for VI initialization.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'vi': The fitted VI object
        - 'history': Optimization history
        - 'X_est': Estimated states
        - 'runtime': Total runtime in seconds
        - 'iterations': Number of iterations performed
        
    Examples
    --------
    >>> from src.inference import TemporalAMENaiveMFVI
    >>> result = run_method_with_timing(
    ...     TemporalAMENaiveMFVI,
    ...     model,
    ...     "Naive MF",
    ...     max_iter=100
    ... )
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running: {method_name}")
        print(f"{'='*70}")
    
    # Initialize VI
    vi = vi_class(model, **vi_kwargs)
    
    # Fit with timing
    start_time = time.time()
    history = vi.fit(max_iter=max_iter, verbose=verbose)
    runtime = time.time() - start_time
    
    # Get estimates
    if hasattr(vi, 'X_mean'):
        X_est = vi.X_mean
    elif hasattr(vi, 'get_variational_means'):
        X_est = vi.get_variational_means()
    else:
        X_est = None
    
    # Package results
    result = {
        'vi': vi,
        'history': history,
        'X_est': X_est,
        'runtime': runtime,
        'iterations': len(history['elbo']) if 'elbo' in history else max_iter,
        'method_name': method_name
    }
    
    if verbose:
        print(f"\nCompleted in {runtime:.2f} seconds")
        if 'reconstruction_error' in history:
            final_mse = history['reconstruction_error'][-1]
            print(f"Final MSE: {final_mse:.6f}")
    
    return result


def generate_experiment_report(
    results: Dict[str, Dict[str, Any]],
    exp_dir: Path,
    X_true: Optional[torch.Tensor] = None,
    experiment_name: str = "Experiment"
) -> None:
    """
    Generate markdown report of experiment results.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping method names to result dictionaries.
    exp_dir : Path
        Experiment directory.
    X_true : torch.Tensor, optional
        True parameters for computing additional metrics.
    experiment_name : str, default="Experiment"
        Name of the experiment.
        
    Examples
    --------
    >>> generate_experiment_report(
    ...     results,
    ...     exp_dir,
    ...     X_true=X_true,
    ...     experiment_name="Three-Way Comparison"
    ... )
    """
    report_path = exp_dir / "report.md"
    
    with open(report_path, 'w') as f:
        # Header
        f.write(f"# {experiment_name} Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Output Directory:** `{exp_dir}`\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Method | Final MSE | Runtime (s) | Iterations |\n")
        f.write("|--------|-----------|-------------|------------|\n")
        
        for method_name, result in results.items():
            history = result.get('history', {})
            runtime = result.get('runtime', 0)
            iterations = result.get('iterations', 0)
            
            final_mse = (history.get('reconstruction_error', [0])[-1] 
                        if 'reconstruction_error' in history else 0)
            
            f.write(f"| {method_name} | {final_mse:.6f} | "
                   f"{runtime:.2f} | {iterations} |\n")
        
        # Rankings
        f.write("\n## Rankings\n\n")
        
        # By reconstruction error
        method_mses = []
        for method_name, result in results.items():
            history = result.get('history', {})
            if 'reconstruction_error' in history:
                final_mse = history['reconstruction_error'][-1]
                method_mses.append((method_name, final_mse))
        
        if method_mses:
            method_mses.sort(key=lambda x: x[1])
            f.write("### By Reconstruction Error (Best to Worst)\n\n")
            for rank, (method_name, mse) in enumerate(method_mses, 1):
                f.write(f"{rank}. **{method_name}**: {mse:.6f}\n")
            
            # Compute improvements
            baseline_mse = method_mses[-1][1]
            f.write(f"\n### Improvement over Baseline ({method_mses[-1][0]})\n\n")
            for method_name, mse in method_mses[:-1]:
                improvement = (1 - mse / baseline_mse) * 100
                f.write(f"- **{method_name}**: {improvement:+.1f}%\n")
        
        # Parameter recovery (if X_true available)
        if X_true is not None:
            from src.utils import compute_alignment_error
            
            f.write("\n## Parameter Recovery\n\n")
            f.write("| Method | Alignment Error |\n")
            f.write("|--------|----------------|\n")
            
            for method_name, result in results.items():
                if 'X_est' in result and result['X_est'] is not None:
                    error, _ = compute_alignment_error(
                        result['X_est'],
                        X_true,
                        latent_dim=2,
                        align=True
                    )
                    f.write(f"| {method_name} | {error:.6f} |\n")
        
        # Figures
        f.write("\n## Figures\n\n")
        f.write("Generated figures can be found in the `figures/` subdirectory:\n\n")
        figures_dir = exp_dir / "figures"
        if figures_dir.exists():
            for fig_file in sorted(figures_dir.glob("*.png")):
                f.write(f"- `{fig_file.name}`\n")
        
        f.write("\n---\n")
        f.write(f"\n*Report generated automatically by experiments/utils.py*\n")
    
    print(f"\nExperiment report saved to: {report_path}")


def _convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    
    Handles torch tensors, numpy arrays, and nested structures.
    """
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (torch.Tensor, np.ndarray)):
        return None  # Skip tensors in JSON
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_experiment_header(
    experiment_name: str,
    params: Dict[str, Any]
) -> None:
    """
    Print formatted experiment header.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    params : dict
        Dictionary of experiment parameters.
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*70)
    print("\nParameters:")
    for key, value in params.items():
        print(f"  {key:20s}: {value}")
    print("="*70 + "\n")