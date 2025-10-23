#!/usr/bin/env python3
"""Setup script for ctx-bandits-mcmc-showdown package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ctx-bandits-mcmc",
    version="1.0.1",
    author="Emile Anand and Sarah Liaw",
    author_email="emiletimothy@outlook.com", 
    description="Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown", 
    packages=find_packages(exclude=["tests", "Neural", "config"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.0.0",
        "scipy>=1.10.0",
        "pandas>=1.0.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "wandb>=0.12.0",
        "pytest>=7.0.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "build>=0.10.0",
            "twine>=4.0.0",
        ],
        "neural": [
            "yfinance",
            "xlrd>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ctx-bandits-posterior=posterior_analysis:main",
            "ctx-bandits-batch=run_linear_batch:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="thompson-sampling contextual-bandits mcmc reinforcement-learning bayesian",
    project_urls={
        "Paper": "https://arxiv.org/abs/2507.15290",
        "Bug Reports": "https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown/issues",
        "Source": "https://github.com/SarahLiaw/ctx-bandits-mcmc-showdown",
    },
    include_package_data=True,
    zip_safe=False,
)
