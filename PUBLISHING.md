# Publishing Guide: How to Release to PyPI

This guide walks you through publishing the `ctx-bandits-mcmc` package to PyPI.

---

## Prerequisites

### 1. Create PyPI Accounts

- **TestPyPI** (for testing): https://test.pypi.org/account/register/
- **PyPI** (production): https://pypi.org/account/register/

### 2. Install Required Tools

```bash
pip install --upgrade build twine
```

### 3. Verify Package Structure

Ensure these files exist:
- [ ] `setup.py` - Package configuration
- [ ] `pyproject.toml` - Modern build configuration
- [ ] `MANIFEST.in` - Include non-Python files
- [ ] `src/__init__.py` - Package initialization
- [ ] `README.md` - Documentation
- [ ] `LICENSE` - License file

---

## Pre-Release Checklist

### 1. Update Version Number

Edit version in both files:

**`setup.py`:**
```python
version="1.0.0",  # Update this
```

**`pyproject.toml`:**
```toml
version = "1.0.0"  # Update this
```

**`src/__init__.py`:**
```python
__version__ = "1.0.0"  # Update this
```

### 2. Run Tests

```bash
# Run all tests
make test

# Verify no failures
pytest test_posterior_analysis.py -v
```

### 3. Update Documentation

- [ ] Update `README.md` with any new features
- [ ] Update `QUICKSTART.md` with new commands
- [ ] Check all links work
- [ ] Verify examples are current

### 4. Check GitHub Repository URL

Update URLs in `setup.py` and `pyproject.toml`:
```python
url="https://github.com/YOUR-USERNAME/ctx-bandits-mcmc-showdown",
```

### 5. Clean Previous Builds

```bash
make clean
rm -rf dist/ build/ *.egg-info
```

---

## Building the Package

### 1. Build Distribution Archives

```bash
python -m build
```

This creates:
- `dist/ctx-bandits-mcmc-1.0.0.tar.gz` (source distribution)
- `dist/ctx_bandits_mcmc-1.0.0-py3-none-any.whl` (wheel)

### 2. Verify Build Contents

```bash
# List contents of wheel
unzip -l dist/ctx_bandits_mcmc-1.0.0-py3-none-any.whl

# List contents of tarball
tar -tzf dist/ctx-bandits-mcmc-1.0.0.tar.gz
```

Check that:
- [ ] All Python files are included
- [ ] Config files are included
- [ ] Documentation files are included
- [ ] No unnecessary files (e.g., `__pycache__`)

---

## Testing on TestPyPI (Recommended First Step)

### 1. Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username (or use `__token__` if using API token)
- Password (or paste your API token)

### 2. Install from TestPyPI

```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ctx-bandits-mcmc
```

**Note**: `--extra-index-url` is needed because dependencies are on PyPI, not TestPyPI.

### 3. Test the Installation

```python
# Test imports
import src
from src import LMCTS, LinTS
print(src.__version__)

# Run a quick test
from posterior_analysis import generate_synthetic_data
contexts, betas, _, _ = generate_synthetic_data(3, 10, 100, 1.0, 0.5, seed=42)
print("Test passed!")
```

### 4. Test Command-Line Tools

```bash
ctx-bandits-posterior --help
```

---

## Publishing to PyPI (Production)

### 1. Final Checks

- [ ] All tests pass on TestPyPI installation
- [ ] Documentation is accurate
- [ ] Version number is correct
- [ ] GitHub repo is public and up-to-date

### 2. Upload to PyPI

```bash
python -m twine upload dist/*
```

Enter your PyPI credentials when prompted.

### 3. Verify Upload

Visit: https://pypi.org/project/ctx-bandits-mcmc/

Check that:
- [ ] Version is correct
- [ ] README renders properly
- [ ] All classifiers are correct
- [ ] Links work

### 4. Test Installation from PyPI

```bash
# Fresh environment
python -m venv prod_test
source prod_test/bin/activate

# Install from PyPI
pip install ctx-bandits-mcmc

# Verify
python -c "import src; print(src.__version__)"
```

---

## Post-Release

### 1. Tag the Release on GitHub

```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### 2. Create GitHub Release

1. Go to: https://github.com/YOUR-USERNAME/ctx-bandits-mcmc-showdown/releases
2. Click "Create a new release"
3. Select the tag you just created
4. Add release notes (what's new, bug fixes, etc.)
5. Attach distribution files (optional): `dist/*`

### 3. Update README Badge (Optional)

Add PyPI badge to README:
```markdown
[![PyPI version](https://badge.fury.io/py/ctx-bandits-mcmc.svg)](https://pypi.org/project/ctx-bandits-mcmc/)
[![Python versions](https://img.shields.io/pypi/pyversions/ctx-bandits-mcmc.svg)](https://pypi.org/project/ctx-bandits-mcmc/)
```

### 4. Announce Release

- Update paper repository link
- Post on relevant forums/communities
- Tweet about it (if applicable)
- Update arXiv paper with installation instructions

---

## Using API Tokens (Recommended)

### Set Up PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: "ctx-bandits-mcmc-upload"
4. Scope: "Entire account" or specific project
5. Copy the token (starts with `pypi-`)

### Configure `.pypirc`

Create/edit `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE
repository = https://test.pypi.org/legacy/
```

Now you can upload without entering credentials:
```bash
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

---

## Continuous Integration (GitHub Actions)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: python -m twine upload dist/*
```

**Setup**: Add `PYPI_API_TOKEN` to GitHub repository secrets.

---

## Version Numbering Guide

Use [Semantic Versioning](https://semver.org/):

- **1.0.0** - Initial release
- **1.0.1** - Bug fix (patch)
- **1.1.0** - New feature (minor)
- **2.0.0** - Breaking change (major)

Examples:
- Bug fix: `1.0.0` → `1.0.1`
- New algorithm added: `1.0.0` → `1.1.0`
- API change: `1.0.0` → `2.0.0`

---

## Troubleshooting

### Error: Package already exists

**Solution**: Increment version number. PyPI doesn't allow re-uploading the same version.

### Error: Invalid distribution filename

**Solution**: Ensure version format is correct (e.g., `1.0.0`, not `v1.0.0`)

### Error: Missing files in distribution

**Solution**: Check `MANIFEST.in` and rebuild:
```bash
rm -rf dist/ build/
python -m build
```

### Error: README doesn't render

**Solution**: Verify `long_description_content_type="text/markdown"` in `setup.py`

### Error: Dependencies not found

**Solution**: Check `install_requires` in `setup.py` matches `requirements.txt`

---

## Quick Reference Commands

```bash
# Clean previous builds
make clean
rm -rf dist/ build/ *.egg-info

# Build package
python -m build

# Test on TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ctx-bandits-mcmc

# Upload to PyPI (production)
python -m twine upload dist/*

# Tag release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

---

## Additional Resources

- **PyPI**: https://pypi.org/
- **TestPyPI**: https://test.pypi.org/
- **Packaging Guide**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/
- **setuptools**: https://setuptools.pypa.io/

---

## Checklist Summary

Before publishing:
- [ ] All tests pass
- [ ] Version number updated in all files
- [ ] Documentation is current
- [ ] GitHub URLs are correct
- [ ] Package builds successfully
- [ ] Tested on TestPyPI
- [ ] LICENSE file exists
- [ ] README is complete

After publishing:
- [ ] Verified PyPI page looks correct
- [ ] Tested pip installation
- [ ] Tagged GitHub release
- [ ] Updated badges/links
- [ ] Announced release

---

**Ready to publish?** Follow the steps above and your package will be live on PyPI! 🚀
