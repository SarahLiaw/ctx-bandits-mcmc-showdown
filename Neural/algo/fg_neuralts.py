import torch
from .baselines import NeuralTS
from train_utils.dataset import sample_data


class FGNeuralTS(NeuralTS):
    """
    Feel‑Good / Smoothed‑Feel‑Good Neural Thompson Sampling
    -------------------------------------------------------
    Extra JSON/YAML hyper‑parameters
    -------------------------------------------------------
    feel_good : bool    # turn the exploration bonus on/off (default False)
    fg_mode : str       # "hard" or "smooth" (default "hard")
    lambda_fg : float   # weight λ
    b_fg : float        # cap b
    smooth_s : float    # smoothing scale s (only used when fg_mode=="smooth")
    """
    def __init__(self,
                 *args,
                 feel_good=False,
                 fg_mode="hard",
                 lambda_fg=0.0,
                 b_fg=1.0,
                 smooth_s=10.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.feel_good = feel_good
        self.fg_mode = fg_mode
        self.lambda_fg = lambda_fg
        self.b_fg = b_fg
        self.smooth_s = smooth_s

    def _phi_s(self, u: torch.Tensor) -> torch.Tensor:
        """Φ_s(u) = log(1 + e^{su}) / s  (element-wise soft-plus)"""
        return torch.nn.functional.softplus(self.smooth_s * u) / self.smooth_s

    def _fg_bonus(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred : (N, K) predicted rewards on stored contexts
        returns scalar −λ Σ_i FG_i
        """
        if not self.feel_good:
            return 0.0

        g_star = pred.max(dim=1).values
        if self.fg_mode == "smooth":
            fg = self.b_fg - self._phi_s(self.b_fg - g_star)
        else:
            fg = torch.minimum(g_star, torch.tensor(self.b_fg, device=pred.device))

        return -self.lambda_fg * fg.sum()

    def update_model(self, num_iter):
        self.step += 1
        if self.reduce:
            if self.step % self.reduce != 0:
                return

        for p in self.optimizer.param_groups:
            p['weight_decay'] = self.reg / self.step

        # update using minibatch
        if self.batchsize and self.batchsize < self.step:
            ploader = sample_data(self.loader)
            for i in range(num_iter):
                contexts, rewards = next(ploader)
                contexts = contexts.to(self.device)
                rewards = rewards.to(dtype=torch.float32, device=self.device)
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss += self._fg_bonus(pred.unsqueeze(1))  # Add feel-good bonus
                loss.backward()
                self.optimizer.step()
            assert not torch.isnan(loss), 'Loss is Nan!'
        else:
            # update using full batch
            contexts, rewards = self.collector.fetch_batch()
            contexts = torch.stack(contexts, dim=0).to(self.device)
            rewards = torch.tensor(
                rewards, dtype=torch.float32, device=self.device)
            self.model.train()
            for i in range(num_iter):
                self.model.zero_grad()
                pred = self.model(contexts).squeeze(dim=1)
                loss = self.criterion(pred, rewards)
                loss += self._fg_bonus(pred.unsqueeze(1))  # Add feel-good bonus
                loss.backward()
                self.optimizer.step()
                if loss.item() < 1e-3:
                    break
            assert not torch.isnan(loss), 'Loss is Nan!'

        # update the design matrix
        self.model.zero_grad()
        if self.image:
            re = self.model(self.last_cxt.unsqueeze(0))
        else:
            re = self.model(self.last_cxt)
        re.backward()
        grad = torch.cat([p.grad.contiguous().view(-1).detach()
                          for p in self.model.parameters()])
        self.Design += grad * grad 