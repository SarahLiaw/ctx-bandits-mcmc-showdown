# Contextual Bandits MCMC Showdown

A comprehensive implementation of contextual bandit algorithms using Markov Chain Monte Carlo methods, featuring both classical MCMC approaches and modern neural bandit algorithms.

## 🏗️ Repository Structure

```
ctx-bandits-mcmc-showdown/
├── README.md                    # Main documentation
├── INSTALL.md                   # Detailed installation guide
├── CONTRIBUTING.md              # Contribution guidelines
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── src/                         # Core MCMC algorithms
│   ├── MCMC.py                  # Main MCMC implementations
│   ├── baseline.py              # Baseline algorithms
│   ├── dataset.py               # Dataset utilities
│   ├── game.py                  # Game environment
│   ├── toy_example.py           # Example usage
│   ├── run_experiments.py       # Experiment runner
│   ├── run_all_wheel_agents.py  # Wheel dataset experiments
│   ├── run_linear_batch.py      # Linear bandit batch experiments
│   └── config/                  # JSON configs for MCMC algorithms
│       ├── linear/              # Linear bandit configs
│       ├── logistic/             # Logistic bandit configs
│       └── wheel/               # Wheel bandit configs
│
├── Neural/                      # Neural bandit implementations
│   ├── algo/                    # Neural bandit algorithms
│   ├── models/                  # Neural network models
│   ├── train_utils/             # Training utilities
│   ├── configs/                 # YAML configs for neural experiments
│   │   ├── image/               # Image dataset configs (CIFAR-10, MNIST)
│   │   ├── uci/                 # UCI dataset configs
│   │   ├── restaurant/          # Restaurant dataset configs
│   │   ├── simulation/          # Simulation configs
│   │   └── profile/             # Profile dataset configs
│   ├── analyze_regret/          # Regret analysis tools
│   ├── data/                    # Dataset storage
│   ├── sweep/                   # Hyperparameter sweep configs
│   ├── run_*.py                 # Dataset-specific runners
│   └── README.md                # Neural-specific documentation
│
└── scripts/                     # Main execution scripts
    └── run.py                   # Main experiment runner
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/ctx-bandits-mcmc-showdown.git
cd ctx-bandits-mcmc-showdown
pip install -r requirements.txt
```

### Running Experiments

**Classical MCMC Algorithms:**
```bash
python scripts/run.py --config_path src/config/linear/lmcts.json
```

**Neural Bandit Algorithms:**
```bash
python Neural/run_restaurant.py --config_path Neural/configs/restaurant/restaurant-restaurant-lmcts.yaml
```

**Batch Experiments:**
```bash
python src/run_linear_batch.py --n_seeds 5
```

## 📊 Supported Algorithms

### Classical MCMC
- Langevin Monte Carlo (LMC)
- Underdamped Langevin Monte Carlo (ULMC)
- Metropolis-Adjusted Langevin Algorithm (MALA)
- Hamiltonian Monte Carlo (HMC)

### Neural Bandits
- Neural Thompson Sampling (NeuralTS)
- Neural Upper Confidence Bound (NeuralUCB)
- Neural Epsilon-Greedy
- Linear Thompson Sampling (LinTS)
- LMCTS (Langevin MCTS)
- Feel-Good variants (FGNeuralTS, FGLMCTS)
- Smoothed Feel-Good variants (SFGNeuralTS, SFGLMCTS)

## 📈 Datasets

- **Linear/Logistic Bandits**: Synthetic datasets
- **Wheel Bandit**: Classic contextual bandit problem
- **UCI Datasets**: Adult, Covtype, Mushroom, Shuttle, Magic, Financial, Jester
- **Image Datasets**: CIFAR-10, MNIST
- **Restaurant Dataset**: Real-world recommendation data

## 🔬 Analysis Tools

- **Regret Analysis**: `Neural/analyze_regret/restaurant_regret_analysis.py`
- **Experiment Tracking**: Weights & Biases integration
- **Hyperparameter Sweeps**: Organized sweep configurations

## 📚 Documentation

- [Installation Guide](INSTALL.md) - Detailed setup instructions
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute
- [Neural Module README](Neural/README.md) - Neural-specific documentation

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{anand2025feelgoodthompsonsamplingcontextual,
  title={Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown}, 
  author={Emile Anand and Sarah Liaw},
  year={2025},
  eprint={2507.15290},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2507.15290}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.