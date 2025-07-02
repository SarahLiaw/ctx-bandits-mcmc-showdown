"""
algo/malats.py   – Metropolis Adjusted Langevin Thompson Sampling
===============================================================
A thin wrapper around LMCTS that replaces the unadjusted Langevin
inner-loop by the MALA kernel defined in algo/langevin.py
"""

from .lmcts import LMCTS
from .langevin import mala_sampling
import torch

class MALATS(LMCTS):
    """
    Feel-Good Thompson Sampling with Metropolis-Adjusted Langevin proposals.
    Only two hooks are overridden:
       _sample_theta  – draws one θ from the current posterior with MALA
       name           – pretty string for wandb / logging
    All hyper-parameters are copied from LMCTS, plus:
        mala_step_size   : γ   (float)
        mala_n_steps     : #   (int, default = num_iter)
        mala_lazy        : bool
    """

    def __init__(self, *args,
                 mala_step_size: float = 1e-3,
                 mala_n_steps: int   = None,
                 mala_lazy: bool     = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mala_step_size = mala_step_size
        self.mala_n_steps   = mala_n_steps or self.num_iter
        self.mala_lazy      = mala_lazy

    # ------------------------------------------------------------------
    def _sample_theta(self, init_theta: torch.Tensor):
        """
        One posterior draw via MALA (uses current replay buffer and model).
        """
        # log-posterior already defined in LMCTS:
        def logp_fn(theta):
            return self._log_posterior(theta)

        return mala_sampling(
            init_theta.detach(),  # warm-start (previous θ or SGD estimate)
            logp_fn = logp_fn,
            step_size = self.mala_step_size,
            beta_inv = self.beta_inv,
            n_steps = self.mala_n_steps,
            lazy = self.mala_lazy,
        )

    # ------------------------------------------------------------------
