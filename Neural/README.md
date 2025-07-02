# Langevin Monte Carlo for Contextual Bandits

This folder contains the PyTorch implementation of **Langevin Monte Carlo Thompson Sampling (LMC-TS)**, as proposed in the paper *Langevin Monte Carlo for Contextual Bandits*.

---

## Experiment Guide

This guide explains how to run experiments with different datasets using the Langevin Monte Carlo Thompson Sampling (LMCTS) algorithm and other bandit algorithms.

### Common Parameters

- `--repeat [NUM]`: Number of times to repeat the experiment (for reliable results)
- `--log`: Enable logging to Weights & Biases (WandB)
- `--config_path`: Path to configuration YAML file

### Financial Dataset Experiments

The financial dataset uses Yahoo Finance data to simulate a portfolio allocation task.

```bash
python Neural/run_financial.py --config_path Neural/configs/uci/financial-lmcts.yaml --device cpu --repeat 1 --log
```

Key configuration parameters:
- `beta_inv`: Controls exploration (higher value = more exploration)
- `layers`: Neural network architecture (e.g., [100, 50, 25])
- `num_iter`: Number of MCMC iterations per step

### Jester Dataset Experiments

The Jester dataset contains joke ratings that can be used for recommendation tasks.

```bash
python Neural/run_jester.py --config_path Neural/configs/uci/jester-lmcts.yaml --device cpu --repeat 5 --log
```

Note: The first run may require installing `xlrd` package if working with Excel files:
```bash
pip install xlrd
```

### CIFAR-10 Image Experiments

The CIFAR-10 dataset is used for image classification tasks.

```bash
python Neural/run_cifar.py --config_path Neural/configs/image/cifar10-lmcts.yaml --repeat 5 --log
```

Note: The CIFAR-10 experiment uses a different configuration directory (`configs/image/` instead of `configs/uci/`).

### UCI Datasets

To run bandit algorithms on UCI datasets (like Adult Census, Shuttle, etc.), use:

```bash
python Neural/run_classifier.py --config_path Neural/configs/uci/shuttle-lmcts.yaml --repeat 5 --log
```

Popular UCI datasets with configurations:
- Adult (Census): `configs/uci/adult-lmcts.yaml`
- Shuttle: `configs/uci/shuttle-lmcts.yaml`
- Mushroom: `configs/uci/mushroom-lmcts.yaml`

## Analyzing Results

To generate plots and analyze experiment results:
1. Run experiments with the `--log` flag to save results to WandB
2. Update the `analyze_regret` folder with your WandB path
3. Follow the analysis scripts to generate comparison plots

## Configuration Tips

For best performance with LMCTS algorithm:
1. Use deeper neural network architectures ([100, 50, 25] often works well, but [100] works as well)
2. Tune `beta_inv` parameter to balance exploration/exploitation
3. Set sufficient `num_iter` (100+) for thorough posterior sampling
4. Enable logging to track cumulative regret and returns