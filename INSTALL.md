# Installation Guide

This guide provides detailed installation instructions for the Contextual Bandits MCMC Showdown repository.

## Prerequisites

Before installing, ensure you have:

- **Python 3.8+** installed on your system
- **Git** for cloning the repository
- **pip** or **conda** package manager
- **CUDA toolkit** (optional, for GPU acceleration)

## Installation Methods

### Method 1: pip Installation (Recommended)

This is the simplest method for most users.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ctx-bandits-mcmc-showdown.git
cd ctx-bandits-mcmc-showdown

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

### Method 2: conda Installation

Use this method if you prefer conda or need specific CUDA versions.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ctx-bandits-mcmc-showdown.git
cd ctx-bandits-mcmc-showdown

# 2. Create conda environment
conda create -n ctx-bandits python=3.10
conda activate ctx-bandits

# 3. Install PyTorch (choose appropriate version)
# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# For GPU with CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For GPU with CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 4. Install remaining dependencies
pip install -r requirements.txt
```

### Method 3: Using environment.yml

Use this method if you want to replicate the exact environment used for development.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ctx-bandits-mcmc-showdown.git
cd ctx-bandits-mcmc-showdown

# 2. Create environment from yml file
conda env create -f environment.yml
conda activate ctx-bandits-mcmc-showdown
```

## GPU Setup (Optional)

If you have an NVIDIA GPU and want to use CUDA acceleration:

### Check CUDA Installation
```bash
nvidia-smi
```

### Install CUDA Toolkit
- **Ubuntu/Debian**: Follow [NVIDIA's installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- **Windows**: Download from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
- **macOS**: CUDA is not supported on macOS

### Verify PyTorch CUDA Support
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Verification

After installation, verify everything works:

```bash
# Test basic imports
python -c "import torch; import wandb; import pandas; import numpy; print('✓ All packages imported successfully')"

# Test PyTorch
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')"

# Test CUDA (if available)
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# Test wandb
python -c "import wandb; print(f'✓ Wandb version: {wandb.__version__}')"
```

## Troubleshooting

### Common Issues

1. **PyTorch installation fails**
   - Try installing from conda instead of pip
   - Check your Python version compatibility
   - Ensure you have the correct CUDA version

2. **CUDA not detected**
   - Verify CUDA toolkit installation: `nvcc --version`
   - Check PyTorch CUDA version matches your CUDA version
   - Restart your terminal after CUDA installation

3. **Memory errors**
   - Reduce batch sizes in configuration files
   - Use CPU-only PyTorch if GPU memory is insufficient
   - Close other applications to free up RAM

4. **Permission errors**
   - Use virtual environments to avoid permission issues
   - Run `pip install --user` if needed
   - Check file permissions in the repository directory

### Getting Help

If you encounter issues:

1. Check the [Issues](https://github.com/YOUR_USERNAME/ctx-bandits-mcmc-showdown/issues) page
2. Create a new issue with:
   - Your operating system
   - Python version
   - Error messages
   - Steps to reproduce the problem

## Development Setup

For contributors:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .
```

## Next Steps

After successful installation:

1. **Set up Weights & Biases** (optional):
   ```bash
   wandb login
   ```

2. **Run a simple experiment**:
   ```bash
   python run.py --config_path config/linear/lmcts.json
   ```

3. **Explore the examples** in the `examples/` directory

4. **Read the documentation** in the `docs/` directory
