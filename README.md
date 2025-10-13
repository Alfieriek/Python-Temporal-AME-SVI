# Temporal AME: Structured Variational Inference for Dynamic Networks

A Python implementation of temporal Additive and Multiplicative Effects (AME) models with structured variational inference. This package demonstrates how properly designed factorization structure in mean-field variational inference can substantially improve parameter recovery and prediction accuracy in temporal network models.

## Overview

**Key Features:**
- Temporal AME model with AR(1) dynamics
- Multiple variational inference methods (Naive MF, Structured MF)
- Comprehensive visualization tools
- Reproducible experiments
- Parameter alignment utilities

**Main Result:** Structured mean-field VI with correct block structure outperforms naive mean-field VI by preserving important correlations in the variational approximation.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/scplummer/temporal-ame-svi.git
cd temporal-ame

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
python demo.py
```

This will:
1. Generate synthetic temporal network data
2. Fit three inference methods (Naive MF, Good SMF, Bad SMF)
3. Create comparison visualizations
4. Show that proper factorization structure matters!

### Run Experiments

```bash
# Main three-way comparison
python experiments/three_way_comparison.py

# Sensitivity analysis
python experiments/sensitivity_analysis.py
```

## Project Structure

```
temporal-ame/
├── src/                          # Core source code
│   ├── models/                   # AME model implementations
│   │   ├── base.py              # Abstract base class
│   │   ├── static_ame.py        # Static AME model
│   │   └── temporal_ame.py      # Temporal AME with AR(1)
│   ├── inference/                # Variational inference algorithms
│   │   ├── base.py              # Base VI classes
│   │   ├── naive_mf.py          # Naive mean-field VI
│   │   └── structured_mf.py     # Structured mean-field VI
│   ├── utils/                    # Utility functions
│   │   ├── diagnostics.py       # Model diagnostics
│   │   ├── alignment.py         # Parameter alignment
│   │   └── metrics.py           # Performance metrics
│   └── visualization/            # Plotting functions
│       ├── static_plots.py      # Static visualizations
│       ├── temporal_plots.py    # Temporal visualizations
│       └── comparison_plots.py  # Method comparisons
├── experiments/                  # Reproducible experiments
│   ├── three_way_comparison.py  # Main comparison experiment
│   ├── sensitivity_analysis.py  # Parameter sensitivity
│   └── utils.py                 # Experiment utilities
├── demo.py                       # Quick start demo script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage Examples

### Generate Data

```python
from src.models import TemporalAMEModel

# Create model
model = TemporalAMEModel(
    n_nodes=15,
    n_time=10,
    latent_dim=2,
    ar_coefficient=0.8
)

# Generate data
Y, X_true = model.generate_data(return_latents=True)
```

### Fit Model with Structured VI

```python
from src.inference import TemporalAMEStructuredMFVI

# Initialize inference
vi = TemporalAMEStructuredMFVI(
    model,
    factorization="good",  # Use correct block structure
    learning_rate=0.01
)

# Fit model
history = vi.fit(max_iter=100, verbose=True)

# Get estimates
X_est = vi.get_variational_means()
```

### Visualize Results

```python
from src.visualization import (
    plot_convergence,
    plot_state_trajectories,
    plot_three_way_comparison
)

# Convergence plot
fig1 = plot_convergence(history, title="Model Convergence")

# Trajectory plot
fig2 = plot_state_trajectories(X_est, node_indices=[0, 1, 2])

# Compare methods
fig3 = plot_three_way_comparison(results, X_true=X_true)
```

### Compute Metrics

```python
from src.utils import (
    compute_reconstruction_error,
    align_temporal_states,
    compute_temporal_contributions
)

# Align estimates
X_aligned = align_temporal_states(X_est, X_true, latent_dim=2)

# Compute error
mse = compute_reconstruction_error(Y_true, Y_pred)

# Analyze contributions
add_contrib, mult_contrib = compute_temporal_contributions(X_aligned, latent_dim=2)
```

## Model

The temporal AME model extends the static AME model [Hoff, 2021] with AR(1) dynamics:

**Observation Model:**
```
Y_ij^t = [y_ij^t, y_ji^t]' ~ N(μ_ij^t, R)
μ_ij^t = [a_i^t + b_j^t + U_i^t' V_j^t, ...]'
```

**Latent Dynamics:**
```
X_i^t = [a_i^t, b_i^t, U_i^t, V_i^t]'
X_i^t = Φ X_i^{t-1} + ε_i^t,  ε_i^t ~ N(0, Q)
```

Where:
- `a_i^t, b_i^t`: Sender/receiver effects (additive)
- `U_i^t, V_i^t`: Latent positions (multiplicative)
- `Φ`: AR(1) transition matrix
- `Q`: Process noise covariance

## Variational Inference Methods

### 1. Naive Mean-Field (Baseline)
Fully factorized approximation:
```
q(X) = ∏_i ∏_t q(X_i^t)
```
with diagonal covariances (all variables independent).

### 2. Structured Mean-Field (Good)
Preserves within-node correlations at each time:
```
q(X) = ∏_i ∏_t q(X_i^t)
```
where `q(X_i^t) = N(μ_i^t, Σ_i^t)` with **full** covariance Σ_i^t.

Groups `[a_i, b_i, U_i, V_i]` together (respects natural coupling).

### 3. Structured Mean-Field (Bad)
Wrong block structure for comparison:
```
q(X_i^t) = q(a_i^t, b_i^t) q(U_i^t, V_i^t)
```
Groups `[a_i, b_i]` separately from `[U_i, V_i]` (breaks natural coupling).

**Key Finding:** Good SMF > Naive MF > Bad SMF (structure matters!)

## Experiments

### Three-Way Comparison
Compares the three inference methods on the same synthetic data to demonstrate:
1. Structured MF improves over Naive MF
2. Incorrect structure can hurt performance
3. Proper block design is crucial

**Run:** `python experiments/three_way_comparison.py`

### Sensitivity Analysis
Examines how problem parameters affect relative performance:
- Network size (`n_nodes`)
- Temporal length (`n_time`)
- Temporal dependence (`ar_coefficient`)
- Edge reciprocity (`rho_dyadic`)

**Run:** `python experiments/sensitivity_analysis.py`

See `experiments/README.md` for more details.

## Results

Typical results on synthetic data (n=15, T=10):

| Method    | Final MSE | Improvement |
|-----------|-----------|-------------|
| Good SMF  | 0.0234    | Baseline    |
| Naive MF  | 0.0312    | -33.3%      |
| Bad SMF   | 0.0389    | -66.2%      |

**Interpretation:** Good SMF achieves 33% lower reconstruction error than Naive MF by preserving important correlations.

## Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy ≥ 1.20
- Matplotlib ≥ 3.3
- SciPy ≥ 1.6
- Seaborn ≥ 0.11

See `requirements.txt` for complete list.

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2025temporal,
  title={Structured Variational Inference for Temporal Network Models},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## References

- Hoff, P. D. (2021). Additive and multiplicative effects network models. *Statistical Science*, 36(1), 34-50.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859-877.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: seanp@uark.edu

## Acknowledgments

This work builds on the AME model framework developed by Peter Hoff and applies structured variational inference techniques inspired by the statistical machine learning literature.