"""
Contextual Bandits with MCMC-based Thompson Sampling
====================================================

This package implements various MCMC-based Thompson Sampling algorithms
for contextual bandits, including:

- Langevin Monte Carlo Thompson Sampling (LMCTS)
- Metropolis-Adjusted Langevin Algorithm Thompson Sampling (MALATS)
- Hamiltonian Monte Carlo Thompson Sampling (HMCTS)
- Preconditioned variants (PLMCTS, PHMCTS)
- Feel-Good variants (FGLMCTS, FGMALATS, FGHMCTS)
- Smoothed Feel-Good variants (SFGLMCTS, SFGMALATS, SFGHMCTS)

Baseline algorithms:
- Linear Thompson Sampling (LinTS)
- Linear UCB (LinUCB)
- Epsilon-Greedy

For detailed documentation, see: https://arxiv.org/abs/2507.15290
"""

__version__ = "1.0.1"
__author__ = "Emile Anand and Sarah Liaw"

# Import key classes for convenience
from .MCMC import (
    LMCTS,
    FGLMCTS,
    SFGLMCTS,
    MALATS,
    FGMALATS,
    SFGMALATS,
    PLMCTS,
    PFGLMCTS,
    PSFGLMCTS,
    HMCTS,
    FGHMCTS,
    SFGHMCTS,
    PHMCTS,
    PFGHMCTS,
    PSFGHMCTS,
)

from .baseline import LinTS, LinUCB, Random, EpsGreedy

from .game import GameToy

__all__ = [
    # MCMC variants
    "LMCTS",
    "FGLMCTS",
    "SFGLMCTS",
    "MALATS",
    "FGMALATS",
    "SFGMALATS",
    "PLMCTS",
    "PFGLMCTS",
    "PSFGLMCTS",
    "HMCTS",
    "FGHMCTS",
    "SFGHMCTS",
    "PHMCTS",
    "PFGHMCTS",
    "PSFGHMCTS",
    # Baselines
    "LinTS",
    "LinUCB",
    "Random",
    "EpsGreedy",
    # Environment
    "GameToy",
]
