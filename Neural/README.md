# Neural Bandit Algorithms

This directory contains neural bandit implementations and experiments for contextual bandit problems.

## 🏗️ Structure

- `algo/` - Neural bandit algorithm implementations
- `models/` - Neural network model definitions  
- `train_utils/` - Training utilities and data adapters
- `configs/` - YAML configuration files for experiments
- `analyze_regret/` - Regret analysis and evaluation tools
- `data/` - Dataset storage (excluded from git)
- `sweep/` - Hyperparameter sweep configurations

## 🚀 Quick Start

### Restaurant Dataset
```bash
python run_restaurant.py --config_path configs/restaurant/restaurant-restaurant-lmcts.yaml --log
```

### CIFAR-10 Image Experiments
```bash
python run_cifar.py --config_path configs/image/cifar10-neuralts.yaml --log
```

### UCI Datasets
```bash
python run_classifier.py --config_path configs/uci/shuttle-lmcts.yaml --log
```

### Financial Dataset
```bash
python run_financial.py --config_path configs/uci/financial-lmcts.yaml --log
```

## 📊 Supported Algorithms

- **NeuralTS**: Neural Thompson Sampling
- **NeuralUCB**: Neural Upper Confidence Bound  
- **NeuralEpsGreedy**: Neural Epsilon-Greedy
- **LinTS**: Linear Thompson Sampling
- **LMCTS**: Langevin Monte Carlo Thompson Sampling
- **FGNeuralTS**: Feel-Good Neural Thompson Sampling
- **FGLMCTS**: Feel-Good Langevin Monte Carlo Thompson Sampling
- **SFGNeuralTS**: Smoothed Feel-Good Neural Thompson Sampling
- **SFGLMCTS**: Smoothed Feel-Good Langevin Monte Carlo Thompson Sampling

## 🔬 Analysis Tools

### Regret Analysis
```bash
python analyze_regret/restaurant_regret_analysis.py
python analyze_regret/fetch_regret.py
```

### Hyperparameter Sweeps
```bash
# Use sweep configurations in sweep/ directory
wandb sweep sweep/image/cifar10-neuralts.yaml
```

## 📈 Datasets

- **Image**: CIFAR-10, MNIST
- **UCI**: Adult, Covtype, Mushroom, Shuttle, Magic, Financial, Jester
- **Restaurant**: Real-world recommendation data
- **Simulation**: Linear, Logistic, Quadratic bandits

## ⚙️ Configuration Tips

### LMCTS Algorithm
- Use deeper architectures: `layers: [100, 50, 25]`
- Tune `beta_inv` for exploration/exploitation balance
- Set sufficient `num_iter` (100+) for thorough sampling

### Feel-Good Variants
- `fg_mode: "hard"` for hard feel-good
- `fg_mode: "smooth"` for smoothed feel-good
- Tune `lambda_fg` and `b_fg` parameters

## 📝 Common Parameters

- `--repeat [NUM]`: Number of experiment repetitions
- `--log`: Enable Weights & Biases logging
- `--config_path`: Path to YAML configuration file
- `--device`: Device selection (cpu/cuda)

## 🔗 Integration

This module integrates with:
- **Weights & Biases** for experiment tracking
- **PyTorch** for neural network implementations
- **Main repository** for classical MCMC algorithms