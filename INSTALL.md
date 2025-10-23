# Installation Guide

## Method 1: Install from PyPI (Recommended for Users)

Once published to PyPI, install with:

```bash
pip install ctx-bandits-mcmc
```

With optional dependencies:
```bash
# For development (testing, linting, type checking)
pip install ctx-bandits-mcmc[dev]

# For neural bandit experiments
pip install ctx-bandits-mcmc[neural]

# All optional dependencies
pip install ctx-bandits-mcmc[dev,neural]
```

---

## Method 2: Install from Source (For Development)

### Standard Installation

```bash
# Clone repository
git clone https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown.git
cd ctx-bandits-mcmc-showdown

# Install in standard mode
pip install .

# Or install with optional dependencies
pip install .[dev,neural]
```

### Editable Installation (Recommended for Development)

This allows you to modify the code without reinstalling:

```bash
# Install in editable mode
pip install -e .

# With dev tools
pip install -e .[dev]
```

---

## Method 3: Install from GitHub Directly

```bash
# Latest release
pip install git+https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown.git

# Specific branch
pip install git+https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown.git@main

# Specific tag/version
pip install git+https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown.git@v1.0.0
```

---

## Verify Installation

After installation, verify it works:

```python
# Python
import src
from src import LMCTS, LinTS

print(f"Package version: {src.__version__}")
```

Or run tests:
```bash
pytest test_posterior_analysis.py -v
```

Or use the command-line tools:
```bash
ctx-bandits-posterior --help
```

---

## Building the Package (For Maintainers)

### Build Distribution Archives

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates:
# dist/ctx-bandits-mcmc-1.0.0.tar.gz
# dist/ctx_bandits_mcmc-1.0.0-py3-none-any.whl
```

### Test the Built Package Locally

```bash
# Install from wheel
pip install dist/ctx_bandits_mcmc-1.0.0-py3-none-any.whl

# Or from source distribution
pip install dist/ctx-bandits-mcmc-1.0.0.tar.gz
```

### Upload to PyPI

```bash
# Test on TestPyPI first (recommended)
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ ctx-bandits-mcmc

# Upload to real PyPI (when ready)
python -m twine upload dist/*
```

---

## Dependencies

### Core Dependencies
- Python >= 3.8
- torch >= 2.0.0
- numpy >= 1.20.0
- matplotlib >= 3.0.0
- scipy >= 1.10.0
- pandas >= 1.0.0
- scikit-learn >= 1.0.0
- tqdm >= 4.60.0
- wandb >= 0.12.0
- pytest >= 7.0.0
- pydantic >= 2.0.0
- PyYAML >= 6.0.0

### Optional Dependencies

**Development tools** (`[dev]`):
- pytest-cov >= 3.0.0
- black >= 22.0.0
- flake8 >= 4.0.0
- mypy >= 0.950

**Neural experiments** (`[neural]`):
- yfinance
- xlrd >= 2.0.0

---

## Platform-Specific Notes

### macOS (Apple Silicon)

PyTorch installation may require specific handling:

```bash
# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# Then install the package
pip install ctx-bandits-mcmc
```

### Linux

Standard installation should work:

```bash
pip install ctx-bandits-mcmc
```

### Windows

```bash
# May need Visual C++ Build Tools for some dependencies
pip install ctx-bandits-mcmc
```

---

## Troubleshooting

### Issue: ImportError after installation

**Solution**: Ensure you're not in the source directory when importing:
```bash
cd ~  # Move out of source directory
python -c "import src; print(src.__version__)"
```

### Issue: Command-line tools not found

**Solution**: Ensure pip install location is in PATH:
```bash
# Check where pip installs scripts
python -m site --user-base

# Add to PATH if needed (in ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

### Issue: Version conflicts

**Solution**: Use a virtual environment:
```bash
python -m venv bandit_env
source bandit_env/bin/activate  # On Windows: bandit_env\Scripts\activate
pip install ctx-bandits-mcmc
```

### Issue: Build fails

**Solution**: Update build tools:
```bash
pip install --upgrade pip setuptools wheel build
```

---

## Uninstallation

```bash
pip uninstall ctx-bandits-mcmc
```

---

## Development Setup

Complete development environment setup:

```bash
# Clone and navigate
git clone https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown.git
cd ctx-bandits-mcmc-showdown

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in editable mode with dev tools
pip install -e .[dev]

# Run tests
make test

# Run linters
black src/ test_*.py
flake8 src/

# Type checking
mypy src/
```

---

## Next Steps

After installation:
1. See [QUICKSTART.md](QUICKSTART.md) for quick usage examples
2. Read [README.md](README.md) for complete documentation
3. Check [TESTING.md](TESTING.md) for testing instructions
4. See [POSTERIOR_ANALYSIS_SUMMARY.md](POSTERIOR_ANALYSIS_SUMMARY.md) for detailed analysis guide

---

## Getting Help

- **Documentation**: [README.md](README.md)
- **Issues**: https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown/issues
- **Paper**: https://arxiv.org/abs/2507.15290
