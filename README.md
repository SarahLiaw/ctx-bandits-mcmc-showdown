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

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MCMC_cb
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

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

## Weights & Biases Integration

The code is integrated with Weights & Biases for experiment tracking. To use it:

1. Install wandb: `pip install wandb`
2. Log in: `wandb login`
3. Run your experiments - results will be logged to your W&B account

## Adding New Agents

To add a new agent:

1. Create a new class in `src/MCMC.py` inheriting from the base agent class
2. Implement the required methods (`choose_arm`, `update`, etc.)
3. Add the agent to the `format_agent` function in `run.py`
4. Create a configuration file in the appropriate `config/` subdirectory

## Citation

If you use this code in your research, please consider citing our paper:

@TODO: add bibtex
