
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn

import torch.nn.functional as F

BETA_INV = 0.001 
# take recipricol: info['d'] * np.log(info['T'])

class LMCTS(object):
    '''
    Langevin Monte Carlo Thompson Sampling bandit
    - info['d']: parameter dimension
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['step_size']: step size used for the Langevin update
    - info['K']: number of gradient iterations
    - info['K_not_updated']: number of gradient iterations when the posterior has not been updated
    - info['nb_arms']: number of arms
    - info['phi']: function (context x number of arms) -> feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    '''
    def __init__(self, info):
        self.info = info
        self.v = torch.tensor([])
        self.r = torch.tensor([])
        self.theta = nn.Parameter(torch.normal(0, 1, size=(self.info['d'], 1)))
        self.V = torch.empty(0, 1, self.info['d'])
        self.idx = 1
        self.is_posterior_updated = True
        base = info.get("beta_inv", BETA_INV)
        self.beta_inv = base * info['d'] * np.log(info['T'])

    def loss_fct(self, theta):
        loss = self.info['eta'] * ((self.v @ theta - self.r)**2).sum()
        loss += self.info['std_prior'] * torch.norm(theta)**2
        return loss

    def train(self):
        if self.theta.grad is not None:
            self.theta.grad.zero_()
        loss = self.loss_fct(self.theta)
        loss.backward()
        noise = torch.randn_like(self.theta) * np.sqrt(2 * self.lr * self.beta_inv)
        self.theta.data += - self.lr * self.theta.grad + noise

    def sample_posterior(self, arm_idx):
        self.lr = self.info['step_size'] / self.idx
        nb_iter = self.info['K'] if self.is_posterior_updated else self.info['K_not_updated']
        for _ in range(nb_iter):
            if self.idx == 1:
                return self.theta
            self.train()
        self.is_posterior_updated = False
        return self.theta

    def choose_arm(self, feature, arm_idx):
        theta = self.sample_posterior(arm_idx)
        rewards = self.info['phi'](feature, self.info['nb_arms']) @ theta
        return rewards.argmax()

    def update(self, action, reward, features, arm_idx):
        v = self.info['phi'](features, self.info['nb_arms'])
        self.v = torch.cat((self.v, v[action, :].unsqueeze(0)))
        dif = v.size()[0] - self.V.size()[1]
        if dif > 0:
            fill = torch.zeros((self.V.size()[0], dif, self.V.size()[2]))
            self.V = torch.cat((self.V, fill), dim=1)
        elif dif < 0:
            fill = torch.zeros((-dif, v.size()[1]))
            v = torch.cat((v, fill), dim=0)
        self.V = torch.cat((self.V, v.unsqueeze(0)))
        self.r = torch.cat((self.r, torch.tensor([reward]).unsqueeze(0)))
        self.idx += 1
        self.is_posterior_updated = True


class FGLMCTS(LMCTS):
    '''
    Feel-Good Langevin Monte Carlo Thompson Sampling bandit
    - info['lambda']: Feed good exploration term
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['b']: bound for the feel good term (cf definetion of the fg term)
    '''
    def __init__(self, info):
        super(FGLMCTS, self).__init__(info)

    def get_g_star(self):
        return (self.V @ self.theta).max(1).values.squeeze()

    def loss_fct(self, theta):
        loss = self.info['eta'] * ((self.v @ theta - self.r)**2).sum()
        loss -= self.info['lambda'] * torch.minimum(self.get_g_star(), torch.tensor([self.info['b']])).sum()
        loss += self.info['std_prior'] * torch.norm(theta)**2
        return loss


class MALATS(object):
    '''
    Metropolis-Adjusted Langevin Algorithm Thompson Sampling bandit:
    - info['step_size']
    - info['K']: number of gradient iterations
    - info['K_not_updated']: number of gradient iterations when the posterior has not been updated
    - info['phi']: function (context x number of arms) -> feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    - info['nb_arms']: number of arms
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['accept_reject_step']: number of gradient descent steps before the MALA update
    '''
    def __init__(self, info):
        self.info = info
        self.theta = nn.Parameter(torch.normal(0, 1, size=(self.info['d'], 1)))
        self.v = torch.tensor([])
        self.r = torch.tensor([])
        self.V = torch.empty(0, 1, self.info['d'])
        self.is_posterior_updated = True
        self.idx = 1
        base = info.get("beta_inv", BETA_INV)
        self.beta_inv = base * info['d'] * np.log(info['T'])

    def get_potential_grad(self, theta):
        if theta.grad is not None:
            theta.grad.zero_()
        loss = self.info['eta'] * ((self.v @ theta - self.r)**2).sum()
        loss += self.info['std_prior'] * torch.norm(theta)**2
        loss.backward()
        return loss, theta.grad

    def logQ(self, x, y, grad):
        return -(torch.norm(y - x - self.lr * grad, p=2) ** 2) / (4 * self.lr)

    def gradient_descent(self):
        _, gradx = self.get_potential_grad(self.theta)
        self.theta.data = self.theta - self.lr * gradx

    def mala_step(self, theta, last_grad, last_potential):
        y = theta.detach() - self.lr * last_grad + np.sqrt(2 * self.lr * self.beta_inv) * torch.randn_like(theta)
        y.requires_grad = True
        new_potential, new_grad = self.get_potential_grad(y)
        log_ratio = - new_potential + last_potential + self.logQ(y, theta, new_grad) - self.logQ(theta, y, last_grad)
        if torch.rand(1) < torch.exp(log_ratio):
            theta = y
            last_potential = new_potential
            last_grad = new_grad
        return theta, last_potential, last_grad

    def train(self, k):
        if k < self.info['accept_reject_step']:
            self.gradient_descent()
        elif k == self.info['accept_reject_step']:
            self.last_potential, self.last_grad = self.get_potential_grad(self.theta)
            self.theta, self.last_potential, self.last_grad = self.mala_step(self.theta, self.last_grad, self.last_potential)
        else:
            self.theta, self.last_potential, self.last_grad = self.mala_step(self.theta, self.last_grad, self.last_potential)

    def sample_posterior(self, arm_idx):
        self.lr = self.info['step_size'] / self.idx
        nb_iter = self.info['K'] if self.is_posterior_updated else self.info['K_not_updated']
        for k in range(nb_iter):
            if self.idx == 1:
                return self.theta
            self.train(k)
        self.is_posterior_updated = False
        return self.theta

    def choose_arm(self, feature, arm_idx):
        theta = self.sample_posterior(arm_idx)
        rewards = self.info['phi'](feature, self.info['nb_arms']) @ theta
        return rewards.argmax()

    def update(self, action, reward, features, arm_idx):
        v = self.info['phi'](features, self.info['nb_arms'])
        self.v = torch.cat((self.v, v[action, :].unsqueeze(0)))
        dif = v.size()[0] - self.V.size()[1]
        if dif > 0:
            fill = torch.zeros((self.V.size()[0], dif, self.V.size()[2]))
            self.V = torch.cat((self.V, fill), dim=1)
        elif dif < 0:
            fill = torch.zeros((-dif, v.size()[1]))
            v = torch.cat((v, fill), dim=0)
        self.V = torch.cat((self.V, v.unsqueeze(0)))
        self.r = torch.cat((self.r, torch.tensor([reward]).unsqueeze(0)))
        self.idx += 1
        self.is_posterior_updated = True



class FGMALATS(MALATS):
    def __init__(self, info):
        '''
        Feel Good Metropolis Adjusted Langevin Algorithm Thompson Sampling
        - info['eta']: inverse of temperature, controls the variance of the posterior distribution
        - info['lambda']: Feed good exploration term
        - info['std_prior']: standard deviation of the gaussian prior distribution
        - info['b']: bound for the feel good term (cf definetion of the fg term)
        '''
        super(FGMALATS, self).__init__(info)

    def get_potential_grad(self, theta):
        if theta.grad is not None:
            theta.grad.zero_()
        loss = self.info['eta'] * ((self.v @ theta - self.r)**2).sum()
        loss -= self.info['lambda'] * torch.minimum(self.get_g_star(), torch.tensor([self.info['b']])).sum()
        loss += self.info['std_prior'] * torch.norm(theta)**2
        loss.backward()
        return loss, theta.grad

    def get_g_star(self):
        return (self.V @ self.theta).max(1).values.squeeze()


class SFGLMCTS(LMCTS):
    """
    Smoothed FG-LMC-TS (soft-plus phi_s smoothing)
    
    - lambda: feel-good weight (same as FG)
    - b: feel-good cap
    - smooth_s: smoothing scale s(float >0, default 10.0)
    """
    def __init__(self, info):
        info.setdefault("smooth_s", 10.0)
        super().__init__(info)

   # \phi_s(u) = log(1+e^{su}) / s 
    def phi_s(self, u):
        s = self.info["smooth_s"]
        return F.softplus(s * u) / s

    def get_g_star(self):
        return (self.V @ self.theta).max(dim=1).values.squeeze()

   #  negative-log posterior 
    def loss_fct(self, theta):
       # data-fit + Gaussian prior
        loss = self.info["eta"] * ((self.v @ theta - self.r) ** 2).sum()
        loss += self.info["std_prior"] * torch.norm(theta) ** 2

       # smoothed feel-good exploration term
        g_star = self.get_g_star()
        fg_term = self.info["b"] - self.phi_s(self.info["b"] - g_star)
        loss -= self.info["lambda"] * fg_term.sum()

        return loss

class SFGMALATS(MALATS):
    """
    Smoothed-Feel-Good MALA-TS
    
    - lambda: feel-good weight
    - b: feel-good cap
    - smooth_s: smoothing scale  s  (float >0, default 10.0)
    """

    def __init__(self, info):
        info.setdefault("smooth_s", 10.0)
        super().__init__(info)

    def phi_s(self, u):
        s = self.info["smooth_s"]
        return F.softplus(s * u) / s

    def get_g_star(self):
        return (self.V @ self.theta).max(dim=1).values.squeeze()

    def get_potential_grad(self, theta):
        if theta.grad is not None:
            theta.grad.zero_()

        loss = self.info["eta"] * ((self.v @ theta - self.r) ** 2).sum()
        loss += self.info["std_prior"] * torch.norm(theta) ** 2

        g_star = self.get_g_star()
        fg_term = self.info["b"] - self.phi_s(self.info["b"] - g_star)
        loss -= self.info["lambda"] * fg_term.sum()

        loss.backward()
        return loss, theta.grad


class PLMCTS(object):
    """
    Pre-conditioned LMC-TS
    --
    Extra hyper-parameters
    - lambda_reg: λ in  V_t = λ I + Σ x_s x_sᵀ   (default 1.0)
    All other keys identical to the original LMCTS.
    """
   #  init 
    def __init__(self, info):
        self.info = info
        d = info['d']
        self.lmbda = info.get('lambda_reg', 1.0)

        self.X = torch.empty(0, d)
        self.r = torch.empty(0, 1)
        self.theta = nn.Parameter(torch.zeros(d, 1))

        self.idx, self.updated = 1, True
        base_beta = info.get("beta_inv", 0.01)
        self.beta_inv = base_beta * d * np.log(info['T'])

    def _loss(self):
        eta = self.info['eta']
        return eta * ((self.X @ self.theta - self.r) ** 2).sum() \
               + self.info['std_prior'] * torch.norm(self.theta) ** 2

    def _gradU(self):
        if self.theta.grad is not None:
            self.theta.grad.zero_()
        self._loss().backward()
        return self.theta.grad.detach()

   #  pre-conditioner  V_t⁻¹
    def _V_inv(self):
        d = self.info['d']
        if self.X.shape[0] == 0: 
            return torch.eye(d) / self.lmbda
        XtX = self.X.t() @ self.X
        return torch.linalg.inv(XtX + self.lmbda * torch.eye(d))

    def sample_posterior(self, _):
        if self.idx == 1: 
            return self.theta

        lr = self.info['step_size'] / self.idx
        K = self.info['K'] if self.updated else self.info['K_not_updated']
        Vinv = self._V_inv()
        noise_scale = np.sqrt(2 * lr * self.beta_inv)

        Vinv_sqrt = torch.linalg.cholesky(Vinv)

        for _ in range(K):
            g = self._gradU()
            z = torch.randn_like(self.theta)
            self.theta.data += -lr * (Vinv @ g) + noise_scale * (Vinv_sqrt @ z)

        self.updated = False
        return self.theta

    def choose_arm(self, ctx, arm_idx):
        th = self.sample_posterior(arm_idx)
        rewards = self.info['phi'](ctx, self.info['nb_arms']) @ th
        return rewards.argmax()

    def update(self, a, reward, ctx, arm_idx):
        x = self.info['phi_a'](ctx, a, self.info['nb_arms'])
        self.X = torch.cat([self.X, x.unsqueeze(0)])
        self.r = torch.cat([self.r, torch.tensor([[reward]])])

        self.idx += 1
        self.updated = True


class PFGLMCTS(PLMCTS):
    """
    Pre-conditioned Feel-Good LMC-TS
    --
    Adds the (hard-clipped) Feel-Good exploration term of Zhang (2021) on top
    of the pre-conditioned Langevin core in `PLMCTS`.

    Extra hyper-parameters (inherited from FGLMCTS)
        - lambda: feel-good weight
        - b: clip cap, min(b , g*)
    """

    def __init__(self, info):
        super().__init__(info)
        nb_arms = info["nb_arms"]
        d = info["d"]
        self.V = torch.empty(0, nb_arms, d) 

    def _g_star(self):
        if self.V.shape[0] == 0:
            return torch.empty(0, device=self.theta.device)
        return (self.V @ self.theta).max(dim=1).values.squeeze()

    def _loss(self):
        eta = self.info['eta']
        theta = self.theta

        data_term = eta * ((self.X @ theta - self.r) ** 2).sum()
        prior_term = self.info['std_prior'] * torch.norm(theta) ** 2


        if self.V.shape[0]:
            gstar = self._g_star()
            fg = torch.minimum(gstar, torch.tensor([self.info['b']],
                                                      device=theta.device))
            fg_term = -self.info['lambda'] * fg.sum()
        else:
            fg_term = 0.0

        return data_term + prior_term + fg_term

    def update(self, a, reward, ctx, arm_idx):
        """
        - push played-arm feature into X, reward into r
        - push full arm-feature matrix into V  (size A×d)
        - update inverse pre-conditioner via parent method
        """
        x_played = self.info['phi_a'](ctx, a, self.info['nb_arms'])
        self.X = torch.cat([self.X, x_played.unsqueeze(0)])
        self.r = torch.cat([self.r, torch.tensor([[reward]],
                                                   dtype=self.X.dtype,
                                                   device=self.X.device)])

        V_t = self.info['phi'](ctx, self.info['nb_arms'])
        self.V = torch.cat([self.V, V_t.unsqueeze(0)])
 
        self.idx += 1
        self.updated = True
        
class PSFGLMCTS(PLMCTS):
    """
    Pre-conditioned *Smoothed* Feel-Good LMC-TS
    
    - inherits all pre-conditioning from `PLMCTS`
    - replaces hard min(b, g*) by   b: phi_s(b: g*),
      with phi_s(u)=log(1+e^{s u})/s

    extra keys in `info`
        - lambda   : FG weight
        - b : FG cap
        - smooth_s : smoothing scale  s  (default 10.0)
    """


    def __init__(self, info):
        info.setdefault("smooth_s", 10.0)
        super().__init__(info)

        A, d = info["nb_arms"], info["d"]
        self.V = torch.empty(0, A, d)
        

    def _phi_s(self, u):
        s = self.info["smooth_s"]
        return F.softplus(s * u) / s
        
    def _g_star(self):
        return (self.V @ self.theta).max(dim=1).values.squeeze() \
               if self.V.shape[0] else torch.empty(0,
                              device=self.theta.device)

    def _loss(self):
        eta = self.info["eta"]
        theta = self.theta

        loss = eta * ((self.X @ theta - self.r) ** 2).sum()
        loss += self.info["std_prior"] * torch.norm(theta) ** 2

        if self.V.shape[0]:
            gstar = self._g_star()
            fg_term = self.info["b"] - self._phi_s(self.info["b"] - gstar)
            loss -= self.info["lambda"] * fg_term.sum()

        return loss

   #  update (store full arm matrix) 
    def update(self, a, reward, ctx, arm_idx):
        x_played = self.info['phi_a'](ctx, a, self.info['nb_arms']) # (d,)
        self.X = torch.cat([self.X, x_played.unsqueeze(0)])
        self.r = torch.cat([self.r,
                              torch.tensor([[reward]],
                                           dtype=self.X.dtype,
                                           device=self.X.device)])

        V_t = self.info['phi'](ctx, self.info['nb_arms'])          # (A,d)
        self.V = torch.cat([self.V, V_t.unsqueeze(0)])

        self.idx += 1
        self.updated = True

#   PHMCFGCTS, PHMCsFGTS

class HMCTS(object):
    """
    Hamiltonian Monte-Carlo Thompson Sampling

    info keys  (all required unless ‘default’ given)
    --
      d       : parameter dimension
      std_prior : Gaussian prior std
      eta     : inverse temperature (data term weight)
      step_size : ε  (leap-frog step size)
      L_leap  : L  (# leap-frog steps per proposal)
      K       : inner proposals when posterior updated
      K_not_updated  : proposals when posterior unchanged
      nb_arms : # arms
      phi     : ctx→(A×d)   feature matrix
      phi_a   : ctx,a→d     played-arm feature
    """

    def __init__(self, info):
        self.info = info
        self.theta = nn.Parameter(torch.zeros(info['d'], 1))

        self.X = torch.empty(0, info['d']) # played features
        self.r = torch.empty(0, 1) # rewards
        self.V = torch.empty(0, 1, info['d'])# all-arm ctx for FG variants

        self.t = 1 # outer time-step
        self.updated    = True # triggers K vs K_not_updated

   #  potential + grad 
    def _U(self, th):
        eta = self.info['eta']
        return eta * ((self.X @ th - self.r) ** 2).sum() + \
               self.info['std_prior'] * torch.norm(th) ** 2

    def _gradU(self, th):
        th = th.clone().detach().requires_grad_(True)
        self._U(th).backward()
        return th.grad.detach()

    def _leapfrog(self, th, p):
        eps = self.info['step_size']
        L   = self.info['L_leap']

        g = self._gradU(th)
        p = p - 0.5 * eps * g # half-kick

        for _ in range(L):
            th = th + eps * p # drift
            g  = self._gradU(th)
            if _ != L - 1: # full-kick except last
                p = p - eps * g
        p = p - 0.5 * eps * g # closing half-kick
        return th, p


    def _hmc_step(self, th):
        p0   = torch.randn_like(th)
        th_p, p_p = self._leapfrog(th, p0)
        
        U0, K0 = self._U(th), 0.5 * (p0**2).sum()
        Up, Kp = self._U(th_p), 0.5 * (p_p**2).sum()
        if torch.rand(1) < torch.exp((U0+K0) - (Up+Kp)):
            return th_p.clone().detach()
        return th.clone().detach()


    def sample_posterior(self, _):
        if self.t == 1:
            return self.theta

        K = self.info['K'] if self.updated else self.info['K_not_updated']
        for _ in range(K):
            self.theta.data = self._hmc_step(self.theta.data)

        self.updated = False
        return self.theta

    def choose_arm(self, ctx, _arm_idx):
        th      = self.sample_posterior(_arm_idx)
        rewards = self.info['phi'](ctx, self.info['nb_arms']) @ th
        return rewards.argmax()

    def update(self, a, reward, ctx, _arm_idx):
        x = self.info['phi_a'](ctx, a, self.info['nb_arms'])
        self.X = torch.cat([self.X, x.unsqueeze(0)])
        self.r = torch.cat([self.r,
                            torch.tensor([[reward]],
                                        dtype=self.X.dtype,
                                        device=self.X.device)])


        A_mat = self.info['phi'](ctx, self.info['nb_arms'])

        diff = A_mat.size(0) - self.V.size(1)
        if diff > 0:
            pad = torch.zeros(self.V.size(0), diff, self.V.size(2),
                            dtype=self.V.dtype, device=self.V.device)
            self.V = torch.cat([self.V, pad], dim=1)
        elif diff < 0:
            pad = torch.zeros(-diff, A_mat.size(1),
                            dtype=A_mat.dtype, device=A_mat.device)
            A_mat = torch.cat([A_mat, pad], dim=0)

        self.V = torch.cat([self.V, A_mat.unsqueeze(0)])       # (t+1, A*, d)

        self.t += 1
        self.updated = True



class FGHMCTS(HMCTS):
    """Feel-Good extension of HMC-TS (hard min(b, g*))."""

    def __init__(self, info):
        super().__init__(info)

    def _g_star(self, th):
        if self.V.shape[0] == 0:
            return torch.zeros(0, device=th.device)
        return (self.V @ th).max(dim=1).values.squeeze()

    def _U(self, th):
        base = super()._U(th)
        if self.V.shape[0]:
            gstar = self._g_star(th)
            fg = torch.minimum(gstar,
                                  torch.tensor([self.info['b']],
                                               device=th.device))
            base -= self.info['lambda'] * fg.sum()
        return base


class SFGHMCTS(HMCTS):
    """
    Hamiltonian Monte-Carlo *Smoothed* Feel-Good TS
    --
    Adds the soft-plus smoothing phi_s(u)=log(1+e^{s u})/s
    on top of the HMC core implemented in `HMCTS`.

    extra keys in `info`
        - lambda   : FG weight
        - b : FG cap
        - smooth_s : smoothing scale  s  (default 10.0)
    """

    def __init__(self, info):
        info.setdefault("smooth_s", 10.0)
        super().__init__(info)


        A, d = info["nb_arms"], info["d"]
        self.V = torch.empty(0, A, d)
        
    def _phi_s(self, u):
        s = self.info["smooth_s"]
        return F.softplus(s * u) / s

    def _g_star(self, th):
        return (self.V @ th).max(dim=1).values.squeeze() \
               if self.V.shape[0] else torch.zeros(0,
                               device=th.device, dtype=th.dtype)


    def _U(self, th):
        base = super()._U(th)
        if self.V.shape[0]:
            gstar = self._g_star(th)
            fg_term = self.info["b"] - self._phi_s(self.info["b"] - gstar)
            base -= self.info["lambda"] * fg_term.sum()
        return base


    def update(self, a, reward, ctx, _arm_idx):
        x = self.info['phi_a'](ctx, a, self.info['nb_arms'])
        self.X = torch.cat([self.X, x.unsqueeze(0)])
        self.r = torch.cat([self.r,
                            torch.tensor([[reward]],
                                         dtype=self.X.dtype,
                                         device=self.X.device)])

        A_mat = self.info['phi'](ctx, self.info['nb_arms'])
        self.V = torch.cat([self.V, A_mat.unsqueeze(0)])

        self.t += 1
        self.updated = True
        

def _build_metric(X, lmbda):
    """Returns  V  and  Vinv  (d×d PSD)  given feature history X."""
    d = X.shape[1]
    V = X.t() @ X + lmbda * torch.eye(d, device=X.device, dtype=X.dtype)
    Vinv = torch.linalg.inv(V)
    return V, Vinv


class PHMCTS(object):
    """Pre-conditioned version of HMCTS (no feel-good term)."""
    def __init__(self, info):
        self.info = info
        self.lmbda = info.get('lambda_reg', 1.0)
        d = info['d']

        self.X = torch.empty(0, d)
        self.r = torch.empty(0, 1)

        self.theta = nn.Parameter(torch.zeros(d, 1))
        self.t = 1
        self.updated = True

    def _U(self, th):
        eta = self.info['eta']
        return eta * ((self.X @ th - self.r) ** 2).sum() + \
               self.info['std_prior'] * torch.norm(th) ** 2

    def _gradU(self, th):
        th = th.clone().detach().requires_grad_(True)
        self._U(th).backward()
        return th.grad.detach()

    def _leapfrog(self, th, p, Vinv):
        eps = self.info['step_size'];
        L = self.info['L_leap']
        g = self._gradU(th)
        p = p - 0.5 * eps * g

        for _ in range(L):
            th = th + eps * (Vinv @ p)
            g = self._gradU(th)
            if _ != L - 1:
                p = p - eps * g
        p = p - 0.5 * eps * g
        return th, p

    def _hmc_step(self, th, V, Vinv):
       # momentum  p0 ~ N(0, V)  via Cholesky
        p0 = torch.linalg.cholesky(V) @ torch.randn_like(th)
        th_p, p_p = self._leapfrog(th, p0, Vinv)

        U0, K0 = self._U(th), 0.5 * (p0.t() @ Vinv @ p0).item()
        Up, Kp = self._U(th_p), 0.5 * (p_p.t() @ Vinv @ p_p).item()
        if torch.rand(1) < torch.exp((U0 + K0) - (Up + Kp)):
            return th_p.detach()
        return th.detach()

    def sample_posterior(self, _):
        if self.t == 1:
            return self.theta

        V, Vinv = _build_metric(self.X, self.lmbda)
        K = self.info['K'] if self.updated else self.info['K_not_updated']
        for _ in range(K):
            self.theta.data = self._hmc_step(self.theta.data, V, Vinv)

        self.updated = False
        return self.theta

    def choose_arm(self, ctx, arm_idx):
        th = self.sample_posterior(arm_idx)
        rewards = self.info['phi'](ctx, self.info['nb_arms']) @ th
        return rewards.argmax()

    def update(self, a, reward, ctx, arm_idx):
        x = self.info['phi_a'](ctx, a, self.info['nb_arms'])
        self.X = torch.cat([self.X, x.unsqueeze(0)])
        self.r = torch.cat([self.r,
                            torch.tensor([[reward]],
                                         dtype=self.X.dtype,
                                         device=self.X.device)])
        self.t += 1
        self.updated = True
        
class PFGHMCTS(PHMCTS):
    """Pre-conditioned HMC with (hard) Feel-Good exploration."""
    def _g_star(self, th):
        if not hasattr(self, 'Vctx') or self.Vctx.shape[0] == 0:
            return torch.zeros(0, device=th.device)
        return (self.Vctx @ th).max(dim=1).values.squeeze()

    def _U(self, th):
        base = super()._U(th)
        if hasattr(self, 'Vctx') and self.Vctx.shape[0]:
            g = self._g_star(th)
            fg = torch.minimum(g,
                               torch.tensor([self.info['b']],
                                            device=th.device))
            base -= self.info['lambda'] * fg.sum()
        return base

    def update(self, a, reward, ctx, arm_idx):
        super().update(a, reward, ctx, arm_idx)
        A = self.info['phi'](ctx, self.info['nb_arms'])
        if not hasattr(self, 'Vctx'):
            self.Vctx = torch.empty(0, *A.shape, device=A.device)
        self.Vctx = torch.cat([self.Vctx, A.unsqueeze(0)])



class PSFGHMCTS(PFGHMCTS):
    """Pre-conditioned HMC with *Smoothed* Feel-Good."""
    def __init__(self, info):
        info.setdefault("smooth_s", 10.0)
        super().__init__(info)

    def _phi_s(self, u):
        s = self.info['smooth_s']
        return F.softplus(s * u) / s

    def _U(self, th):
        base = super(PFGHMCTS, self)._U(th) 
        if hasattr(self, 'Vctx') and self.Vctx.shape[0]:
            g = self._g_star(th)
            fg = self.info['b'] - self._phi_s(self.info['b'] - g)
            base -= self.info['lambda'] * fg.sum()
        return base


class ULMCTS(object):
    '''
    Underdamped Langevin Monte Carlo Thompson Sampling bandit
    - info['gamma']: Friction coefficient (damping)
    - info['h']: Step size
    - info['K']: Number of leapfrog steps
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['phi']: function (context x number of arms) -> all feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm
    '''
    def __init__(self, info):
        self.info = info
        self.v = torch.tensor([]) # Feature vectors of played arms
        self.r = torch.tensor([]) # Rewards
        self.V = torch.empty(0, 1, self.info['d']) # History of feature matrices
        self.is_posterior_updated = True
        self.idx = 1
        self.gamma = info.get('gamma', 0.1) # Friction coefficient
        self.h = info.get('h', 0.01) # Step size
        self.theta_momentum = None
        self.theta = torch.randn((self.info['d'], 1)) # Initialize parameters
        self.K = info.get('K', 25) # Number of MCMC steps
        self.K_not_updated = info.get('K_not_updated', 1) # Steps when not updated

    def choose_arm(self, features, arm_idx):
        '''
        Choose an arm based on current posterior samples
        - features: context
        - arm_idx: index of the arm to choose
        '''
        theta = self.sample_posterior(arm_idx)
        return (features @ theta).argmax().item()

    def update(self, action, reward, features, arm_idx):
        '''
        Update the model with new observation
        - action: chosen arm (index of the chosen arm)
        - reward: observed reward
        - features: context tensor of shape (num_arms, feature_dim)
        - arm_idx: index of the chosen arm (redundant with action, but kept for compatibility)
        '''
       # Ensure features is at least 2D
        if features.dim() == 1:
            features = features.unsqueeze(0) # Add batch dimension if needed
            
       # Get the feature vector for the chosen arm
        if isinstance(action, torch.Tensor):
            action = action.item()
            
       # Ensure we have a valid action index
        if action < 0 or action >= features.size(0):
            raise ValueError(f"Invalid action index: {action} for features with shape {features.shape}")
            
       # Get the feature vector for the chosen arm and ensure it's 2D (1, feature_dim)
        chosen_feature = features[action].unsqueeze(0)
        
       # Store the feature vector and reward
        if self.v.numel() == 0: # First update
            self.v = chosen_feature
        else:
            self.v = torch.cat([self.v, chosen_feature], dim=0)
            
       # Store the reward
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32)
        if self.r.numel() == 0: # First update
            self.r = reward_tensor
        else:
            self.r = torch.cat([self.r, reward_tensor], dim=0)
        
       # Store the full context for all arms (add batch dimension if needed)
        if features.dim() == 2:
            features = features.unsqueeze(0) # Add batch dimension
            
        if self.V.numel() == 0: # First update
            self.V = features
        else:
            self.V = torch.cat([self.V, features], dim=0)
            
        self.idx += 1
        self.is_posterior_updated = True

    def sample_posterior(self, arm_idx):
        '''Sample from the posterior using underdamped Langevin dynamics'''
        if not self.is_posterior_updated:
            return self.theta.detach()

       # Number of steps to run the MCMC
        num_steps = self.K if self.is_posterior_updated else self.K_not_updated
        
       # Initialize momentum if needed
        if self.theta_momentum is None:
            self.theta_momentum = torch.randn_like(self.theta)
        
       # Run underdamped Langevin dynamics
        for _ in range(num_steps):
           # Half step for momentum
            grad_U = self.grad_U(self.theta)
            self.theta_momentum = self.theta_momentum - 0.5 * self.h * grad_U
            
           # Full step for position
            self.theta = self.theta + 0.5 * self.h * self.theta_momentum
            
           # Update momentum with friction
            noise = torch.randn_like(self.theta_momentum)
            self.theta_momentum = (1 - self.gamma * self.h) * self.theta_momentum \
                                - self.h * self.grad_U(self.theta) \
                                + np.sqrt(2 * self.h) * noise
            
           # Final half step for position
            self.theta = self.theta + 0.5 * self.h * self.theta_momentum
        
        self.is_posterior_updated = False
        return self.theta.detach()

    def grad_U(self, theta):
        '''Compute the gradient of the potential energy function U(theta)
        
        U(theta) = -log p(theta|D) = -log p(D|theta) - log p(theta)
        where:
        - p(D|theta) is the likelihood
        - p(theta) is the prior (Gaussian)
        '''
       # Compute the gradient of the negative log-likelihood
        if len(self.r) == 0: # If no data, just return the prior
            return theta / (self.info['std_prior'] ** 2)
            
       # Reshape theta if needed
        theta_flat = theta.view(-1, 1)
        
       # Compute the gradient of the negative log-likelihood
       # For Gaussian likelihood: -log p(D|theta) = 1/(2*sigma^2) * sum((r - phi^T theta)^2)
       # So grad = -1/sigma^2 * sum((r - phi^T theta) * phi)
       # For simplicity, we'll assume sigma = 1 here
        pred = self.v @ theta_flat
        error = pred - self.r
        grad_likelihood = (self.v.t() @ error) / len(self.r)
        
       # Add the gradient of the negative log-prior (Gaussian prior)
        grad_prior = theta_flat / (self.info['std_prior'] ** 2)
        
       # Total gradient (scaled by eta)
        grad = self.info.get('eta', 1.0) * grad_likelihood + grad_prior
        
        return grad

    def loss_fct(self, theta):
        '''Compute the potential energy function U(theta) = -log p(theta|D)'''
        if len(self.r) == 0: # If no data, just return the prior
            return 0.5 * (theta ** 2).sum() / (self.info['std_prior'] ** 2)
            
       # Reshape theta if needed
        theta_flat = theta.view(-1, 1)
        
       # Compute negative log-likelihood (Gaussian)
        pred = self.v @ theta_flat
        nll = 0.5 * ((pred - self.r) ** 2).sum() / len(self.r)
        
       # Add negative log-prior (Gaussian)
        nlp = 0.5 * (theta_flat ** 2).sum() / (self.info['std_prior'] ** 2)
        
       # Total potential energy (scaled by eta)
        return self.info.get('eta', 1.0) * nll + nlp


class ULMCFGTS(ULMCTS):
    '''
    Feel-Good Underdamped Langevin Monte Carlo Thompson Sampling bandit
    - info['lambda']: Feed good exploration term
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['b']: bound for the feel good term (cf definition of the fg term)
    '''
    def __init__(self, info):
        super(ULMCFGTS, self).__init__(info)
       # Store b as a regular tensor
        self.b_tensor = torch.tensor([info['b']])
    
   # Compute max_a (V @ theta) for each context in history
    def get_g_star(self):
        if not hasattr(self, 'V') or self.V.shape[0] == 0:
            return 0.0
        
       # Compute V @ theta for all arms in all contexts
        all_arm_values = (self.V @ self.theta).squeeze(-1) # [T, A]
       # Get max value for each context
        g_star = all_arm_values.max(dim=1)[0] # [T]
        return g_star
    
    def loss_fct(self, theta):
       # Standard linear regression loss
        pred = self.v @ theta
        mse_loss = 0.5 * ((pred - self.r) ** 2).mean()
        
       # Prior regularization
        prior_loss = 0.5 * (theta ** 2).sum() / (self.info['std_prior'] ** 2)
        
       # Feel-good exploration term
        if hasattr(self, 'V') and self.V.shape[0] > 0:
           # Compute max_a (V @ theta) for each context in history
            all_arm_values = (self.V @ theta).squeeze(-1) # [T, A]
            max_values = all_arm_values.max(dim=1)[0] # [T]
            
           # Clip the max values at b
            clipped_max = torch.min(max_values, self.b_tensor.to(theta.device))
            fg_term = -self.info['lambda'] * clipped_max.sum()
        else:
            fg_term = 0.0
        
       # Total loss
        total_loss = self.info['eta'] * (mse_loss + prior_loss) + fg_term
        return total_loss


class ULMCSFGTS(ULMCTS):
    '''
    Smoothed Feel-Good Underdamped Langevin Monte Carlo Thompson Sampling bandit
    - info['lambda']: Feel-good exploration term
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['std_prior']: standard deviation of the gaussian prior distribution
    - info['b']: bound for the feel good term
    - info['smooth_s']: smoothing scale parameter (default: 10.0)
    '''
    def __init__(self, info):
        info.setdefault("smooth_s", 10.0) # Default smoothing parameter
        super(ULMCSFGTS, self).__init__(info)
       # Store b as a regular tensor
        self.b_tensor = torch.tensor([info['b']])
    
    def phi_s(self, u):
        """Smooth plus function: log(1 + e^(s*u)) / s"""
        s = self.info['smooth_s']
        return F.softplus(s * u) / s
    
   # Compute max_a (V @ theta) for each context in history
    def get_g_star(self, theta=None):
        if not hasattr(self, 'V') or self.V.shape[0] == 0:
            return 0.0
        
        if theta is None:
            theta = self.theta
            
       # Compute V @ theta for all arms in all contexts
        all_arm_values = (self.V @ theta).squeeze(-1) # [T, A]
       # Get max value for each context
        g_star = all_arm_values.max(dim=1)[0] # [T]
        return g_star
    
    def loss_fct(self, theta):
       # Standard linear regression loss
        pred = self.v @ theta
        mse_loss = 0.5 * ((pred - self.r) ** 2).mean()
        
       # Prior regularization
        prior_loss = 0.5 * (theta ** 2).sum() / (self.info['std_prior'] ** 2)
        
       # Smoothed Feel-good exploration term
        if hasattr(self, 'V') and self.V.shape[0] > 0:
            g_star = self.get_g_star(theta)
            b = self.b_tensor.to(theta.device)
            
           # Apply smoothed min(b, g*)
            fg_term = b - self.phi_s(b - g_star)
            fg_term = -self.info['lambda'] * fg_term.sum()
        else:
            fg_term = 0.0
        
       # Total loss
        total_loss = self.info['eta'] * (mse_loss + prior_loss) + fg_term
        return total_loss


class SVRGLMCTS(LMCTS):
    """
    Langevin TS with Stochastic Variance-Reduced Gradient (SVRG).
    extra keys in `info`:  batch_size , m_inner
    """
    def __init__(self, info):
        super().__init__(info)
        self.theta_ref = self.theta.detach().clone()
        self.full_grad = torch.zeros_like(self.theta)
        self.k_inner = 0
        base = info.get("beta_inv", BETA_INV)
        self.beta_inv = base * info['d'] * np.log(info['T'])

        
    def _compute_full_grad(self):
        theta = self.theta_ref.clone().detach().requires_grad_(True)
        loss = self.loss_fct(theta)
        loss.backward()
        self.full_grad = theta.grad.detach()
        
    def _svrg_grad(self):
        B = min(self.info['batch_size'], self.v.shape[0])
        idx = torch.randint(0, self.v.shape[0], (B,))

        vB = self.v[idx]; rB = self.r[idx]

        th_cur = self.theta.clone().detach().requires_grad_(True)
        loss_c = self.info['eta'] * ((vB @ th_cur - rB) ** 2).mean()
        loss_c.backward()
        g_cur  = th_cur.grad.detach()

        th_ref = self.theta_ref.clone().detach().requires_grad_(True)
        loss_r = self.info['eta'] * ((vB @ th_ref - rB) ** 2).mean()
        loss_r.backward()
        g_ref  = th_ref.grad.detach()
        return g_cur - g_ref + self.full_grad

    def train(self):
        if self.k_inner == 0:
            self.theta_ref = self.theta.detach().clone()
            self._compute_full_grad()
        g = self._svrg_grad()
        noise = torch.randn_like(self.theta) * np.sqrt(2 * self.lr * self.beta_inv)
        self.theta.data += -self.lr * g + noise
        self.k_inner = (self.k_inner + 1) % self.info['m_inner']

