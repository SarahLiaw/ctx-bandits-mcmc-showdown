# Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown

This repository implements various MCMC-based contextual bandit algorithms.

## Features

- **Algorithms**:
  - Langevin Monte Carlo (LMC)
  - Underdamped Langevin Monte Carlo (ULMC)
  - Metropolis-Adjusted Langevin Algorithm (MALA)
  - Hamiltonian Monte Carlo (HMC)
  - Epsilon-Greedy
  - Upper-Confidence-Bound (UCB)
  - Neural Thompson Sampling (NTS)
  - Linear Thompson Sampling (LTS)
  - Neural Upper-Confidence-Bound (NUCB)
  - Neural Greedy (NG)
  - And numerous variants with Feel-Good and smoothed Feel-Good exploration terms

- **Environments**:
  - Linear bandits
  - Logistic bandits
  - Wheel bandit problem
  - Neural bandits

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install ctx-bandits-mcmc
```

With optional dependencies:
```bash
# For development tools
pip install ctx-bandits-mcmc[dev]

# For neural bandit experiments  
pip install ctx-bandits-mcmc[neural]

# All optional dependencies
pip install ctx-bandits-mcmc[dev,neural]
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown.git
cd ctx-bandits-mcmc-showdown

# Install the package
pip install .

# Or install in editable mode for development
pip install -e .[dev]
```

### Option 3: Install from GitHub

```bash
pip install git+https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown.git
```

For detailed installation instructions, platform-specific notes, and troubleshooting, see [INSTALL.md](INSTALL.md).

## Quick Start

### Running Linear Bandit Experiments

To run a linear bandit experiment with the LMC-TS agent:

```bash
python3 run.py --config_path config/linear/lmcts.json
```

### Running Wheel Bandit Experiments

To run the wheel bandit experiment with the ULMC agent:

```bash
python3 run_all_wheel_agents.py --agents ulmc --num_trials 1
```

### Batch Running Multiple Experiments

To run multiple experiments with different seeds:

```bash
python3 run_linear_batch.py --n_seeds 5
```

## Configuration

Configuration files are stored in the `config/` directory, organized by environment type (linear, logistic, wheel, neural). Each agent has its own configuration file with hyperparameters.

## Results

Results are saved in the `results/` directory by default. The directory structure is:

```
results/
  ├── linear/
  ├── logistic/
  └── wheel/
  └── neural/
```

## Posterior Distribution Quality Analysis

### Overview

The `posterior_analysis.py` script provides a controlled comparison of MCMC algorithm posterior approximations against the true analytical Bayesian posterior. This analysis isolates **approximation quality** from **exploration strategy** by running all algorithms on identical data.

**Key insight:** Thompson Sampling theory assumes sampling from the true posterior π*_t. This tool quantifies how well MCMC approximations π̃_t match π*_t.

### Quick Start

Run posterior analysis with default algorithms:

```bash
python posterior_analysis.py
```

This will:
- Generate fixed synthetic data (6 arms, 20 dimensions, 2000 timesteps)
- Run LinTS, LMCTS, FGLMCTS, MALATS, and PLMCTS on identical data
- Compute true analytical posteriors for each arm
- Create 2D scatter plots comparing true (green) vs. algorithm (red) posteriors
- Calculate Wasserstein distances quantifying approximation quality

### Customization

**Select specific algorithms:**
```bash
python posterior_analysis.py --algorithms LinTS LMCTS MALATS
```

**Change random seed:**
```bash
python posterior_analysis.py --seed 42
```

**Available algorithms:**
- `LinTS` - Analytical Thompson Sampling (baseline)
- `LMCTS` - Langevin Monte Carlo TS
- `FGLMCTS` - Feel-Good LMC-TS
- `SFGLMCTS` - Smoothed Feel-Good LMC-TS
- `MALATS` - Metropolis-Adjusted Langevin
- `FGMALATS` - Feel-Good MALA-TS
- `SFGMALATS` - Smoothed Feel-Good MALA-TS
- `PLMCTS` - Preconditioned LMC-TS
- `PFGLMCTS` - Preconditioned Feel-Good LMC-TS
- `PSFGLMCTS` - Preconditioned Smoothed Feel-Good LMC-TS
- `HMCTS` - Hamiltonian Monte Carlo TS
- `FGHMCTS`, `SFGHMCTS`, `PHMCTS`, `PFGHMCTS`, `PSFGHMCTS` - HMC variants

### Configuration

Edit parameters in `posterior_analysis.py` (lines 31-38):

```python
K_ARMS = 6                      # Number of arms
D_DIM = 20                      # Context dimension
LAMBDA_PRIOR = 1.0              # Prior precision
SIGMA_REWARD = 0.5              # Reward noise
T_HORIZON = 2000                # Time horizon
N_POSTERIOR_SAMPLES = 1500      # Samples for visualization
ETA = 1.0                       # Inverse temperature
CORRELATED_CONTEXTS = True      # True: elliptical, False: circular posteriors
```

### Output Structure

Results are saved in `posterior_analysis_YYYYMMDD_HHMMSS/`:

```
posterior_analysis_20251023_100530/
├── synthetic_data.pt                    # Fixed data used by all algorithms
├── results.json                         # Wasserstein distances and play counts
├── LinTS_posterior_comparison.png       # 1×6 grid: true (green) vs. alg (red)
├── LMCTS_posterior_comparison.png
├── MALATS_posterior_comparison.png
└── ...
```

### Interpreting Results

**Visualization (PNG files):**
- **1×6 grid**: One subplot per arm showing β₁ vs β₂ projection
- **Green scatter**: 1500 samples from true analytical posterior
- **Red scatter**: 1500 samples from algorithm's posterior
- **Good approximation**: Red and green overlap
- **Poor approximation**: Red shifted or wrong shape
- **Under-exploration**: Sparse or missing red samples

**Metrics (results.json):**
```json
{
  "LMCTS": {
    "wasserstein_distances": [0.23, 0.34, 0.18, 0.56, 0.29, 1.23],
    "mean_wasserstein": 0.468,
    "num_plays_per_arm": [423, 312, 589, 245, 389, 42]
  }
}
```

- **Wasserstein distance < 0.3**: Excellent approximation
- **0.3 < W < 0.7**: Good approximation
- **W > 1.0**: Poor approximation
- **W = NaN**: Arm never played

### Testing

Run unit tests to verify core functionality:

```bash
python -m pytest test_posterior_analysis.py -v
```

Or with unittest:

```bash
python test_posterior_analysis.py
```

Tests cover:
- Data generation (correlated/uncorrelated)
- True posterior computation
- Feature map correctness
- Sampling procedures
- Wasserstein distance calculation

### Use Cases

1. **Algorithm Development**: Verify new MCMC variants maintain accurate posteriors
2. **Hyperparameter Tuning**: Check if step sizes / burn-in periods affect approximation quality
3. **Failure Mode Diagnosis**: Distinguish under-exploration from poor MCMC convergence
4. **Computational Trade-offs**: Evaluate if preconditioned methods justify extra cost

### Mathematical Background

For linear bandits with reward model r_t = X_t^T β_i + ε_t, the posterior is:

```
Prior:      β_i ~ N(0, λ^-1 I)
Posterior:  β_i | D_t ~ N(μ_post, Σ_post)

Σ_post^-1 = λI + (η/σ²) Σ X_s X_s^T
μ_post = Σ_post · (η/σ²) · Σ X_s r_s
```

This closed-form solution serves as ground truth. MCMC algorithms approximate this through sampling procedures (Langevin, MALA, HMC).

## Weights & Biases Integration

The code is integrated with Weights & Biases for experiment tracking. To use it:

1. Install wandb: `pip install wandb`
2. Log in: `wandb login`
3. Run your experiments - results will be logged to your W&B account

## Testing

### Running Tests

We provide comprehensive unit tests for the posterior analysis functionality:

```bash
# Quick test run
make test

# Verbose output
make test-verbose

# Skip slow tests
make test-quick
```

Or directly with pytest:
```bash
pytest test_posterior_analysis.py -v
```

### Test Coverage

Tests validate:
- ✅ Data generation (correlated/uncorrelated contexts)
- ✅ True posterior computation (Bayesian linear regression)
- ✅ Block diagonal feature maps
- ✅ Posterior sampling procedures
- ✅ Wasserstein distance calculation
- ✅ Complete pipeline integration

See [TESTING.md](TESTING.md) for detailed testing documentation.

## Development Workflow

### Using Make Commands

```bash
make install              # Install dependencies
make test                 # Run tests
make posterior-analysis   # Run posterior analysis
make build                # Build distribution packages
make upload-test          # Upload to TestPyPI
make update-version       # Update package version (interactive)
make clean                # Clean generated files
```

### Updating Package Version

To release a new version:

```bash
# Automated (recommended)
make update-version

# Or manual: update version in setup.py, pyproject.toml, src/__init__.py
# Then: make clean-all && make build && make upload
```

See [VERSION_UPDATE_GUIDE.md](VERSION_UPDATE_GUIDE.md) for complete instructions.

### Adding New Agents

To add a new agent:

1. Create a new class in `src/MCMC.py` inheriting from the base agent class
2. Implement the required methods (`choose_arm`, `update`, etc.)
3. Add the agent to the `format_agent` function in `run.py`
4. Create a configuration file in the appropriate `config/` subdirectory
5. Add tests if implementing new sampling mechanisms

## Citation

If you use this code in your research, please consider citing our paper:

@article{anand2025feelgoodthompsonsamplingcontextual,
      title={Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown}, 
      author={Emile Anand and Sarah Liaw},
      year={2025},
      eprint={2507.15290},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2507.15290](https://arxiv.org/abs/2507.15290)}, 
}
