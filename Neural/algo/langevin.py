import math
import torch
from torch.optim import Optimizer

from torch import Tensor
from typing import List

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

def lmc(params: List[Tensor],
        d_p_list: List[Tensor],
        weight_decay: float,
        lr: float):
    r"""Functional API that performs Langevine MC algorithm computation.
    """

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add_(param, alpha=weight_decay)

        param.add_(d_p, alpha=-lr)


class LangevinMC(Optimizer):
    def __init__(self,
                 params,              # parameters of the model
                 lr=0.01,             # learning rate
                 beta_inv=0.01,       # inverse temperature parameter
                 sigma=1.0,           # variance of the Gaussian noise
                 weight_decay=1.0,
                 device=None):   # l2 penalty
        if lr < 0:
            raise ValueError('lr must be positive')
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.beta_inv = beta_inv
        self.lr = lr
        self.sigma = sigma
        self.temp = - math.sqrt(2 * beta_inv / lr) * sigma
        self.curr_step = 0
        defaults = dict(weight_decay=weight_decay)
        super(LangevinMC, self).__init__(params, defaults)

    def init_map(self):
        self.mapping = dict()
        index = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    num_param = p.numel()
                    self.mapping[p] = [index, num_param]
                    index += num_param
        self.total_size = index

    @torch.no_grad()
    def step(self):
        self.curr_step += 1
        if self.curr_step == 1:
            self.init_map()

        lr = self.lr
        temp = self.temp
        noise = temp * torch.randn(self.total_size, device=self.device)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            params_with_grad = []
            d_p_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)

                    start, length = self.mapping[p]
                    add_noise = noise[start: start + length].reshape(p.shape)
                    delta_p = p.grad
                    delta_p = delta_p.add_(add_noise)
                    d_p_list.append(delta_p)
                    # p.add_(delta_p)
            lmc(params_with_grad, d_p_list, weight_decay, lr)



@torch.no_grad()
def mala_sampling(
    init_theta: torch.Tensor,
    logp_fn,                        # callable returning log-posterior, gradients must be enabled
    step_size: float,
    beta_inv: float,
    n_steps: int,
    lazy: bool = True,
):
    """
    Metropolis-Adjusted Langevin Algorithm (1/2-lazy version).
    Parameters
    ----------
    init_theta  : starting position      (torch.Tensor,   requires_grad = False)
    logp_fn     : lambda θ → log π(θ)    (callable)
    step_size   : γ in the paper         (float)
    beta_inv    : 1/β  (noise scale)     (float)
    n_steps     : number of MALA moves   (int)
    lazy        : prepend a 'stay-put'   (bool)
    Returns
    -------
    theta  : last accepted sample (torch.Tensor, detached, CPU stays CPU / GPU stays GPU)
    """

    theta = init_theta.clone().detach()
    # Convenient wrappers
    normal = MultivariateNormal(torch.zeros_like(theta), torch.eye(theta.numel(), device=theta.device))
    unif   = Uniform(torch.tensor(0., device=theta.device), torch.tensor(1., device=theta.device))

    for k in range(n_steps):
        # ---------- proposal (Euler–Maruyama) ----------
        theta.requires_grad_(True)
        logp = logp_fn(theta)
        grad = torch.autograd.grad(logp, theta)[0]
        theta.requires_grad_(False)

        noise = torch.sqrt(torch.tensor(2.*step_size*beta_inv, device=theta.device))*normal.sample()
        prop  = theta + step_size * grad + noise

        # ---------- accept / reject ----------
        # forward transition density q(θ→θ')
        def _log_q(x_from, x_to, grad_from):
            diff = x_to - (x_from + step_size*grad_from)
            return - diff.pow(2).sum() / (4.*step_size*beta_inv)

        prop.requires_grad_(True)
        logp_prop = logp_fn(prop)
        grad_prop = torch.autograd.grad(logp_prop, prop)[0]
        prop.requires_grad_(False)

        log_alpha = (
            logp_prop
            + _log_q(prop, theta, grad_prop)
            - logp
            - _log_q(theta, prop, grad)
        )
        if lazy:
            log_alpha = torch.logaddexp(torch.log(torch.tensor(0.5, device=theta.device)), log_alpha)  # 1/2-lazy

        if torch.log(unif.sample()) < log_alpha:
            theta = prop.detach()

    return theta