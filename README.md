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

### Option 1: Using pip (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ctx-bandits-mcmc-showdown.git
   cd ctx-bandits-mcmc-showdown
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using conda

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ctx-bandits-mcmc-showdown.git
   cd ctx-bandits-mcmc-showdown
   ```

2. Create a conda environment:
   ```bash
   conda create -n ctx-bandits python=3.10
   conda activate ctx-bandits
   ```

3. Install PyTorch (with CUDA support if you have a GPU):
   ```bash
   # For CPU only
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   
   # For GPU (CUDA 11.8)
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For GPU (CUDA 12.1)
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 3: Using the environment.yml file

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ctx-bandits-mcmc-showdown.git
   cd ctx-bandits-mcmc-showdown
   ```

2. Create environment from yml file:
   ```bash
   conda env create -f environment.yml
   conda activate ctx-bandits-mcmc-showdown
   ```

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB recommended for neural experiments)
- **GPU**: Optional but recommended for neural bandit experiments
- **CUDA**: 11.8 or 12.1 (if using GPU)

### Verification

To verify your installation, run:
```bash
python -c "import torch; import wandb; import pandas; print('Installation successful!')"
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
  |-- linear/
  |-- logistic/
  |-- wheel/
  |-- neural/
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

@article{anand2025feelgoodthompsonsamplingcontextual,
      title={Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown}, 
      author={Emile Anand and Sarah Liaw},
      year={2025},
      eprint={2507.15290},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2507.15290](https://arxiv.org/abs/2507.15290)}, 
}
