import torch
from .lmcts import LMCTS
from train_utils.dataset import sample_data


class FGLMCTS(LMCTS):
    """
    Feel‑Good / Smoothed‑Feel‑Good LMCTS
    ------------------------------------
    Extra JSON/YAML hyper‑parameters
    ------------------------------------
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

    def update_model(self, num_iter=5):
        self.step += 1
        if self.reduce and self.step % self.reduce != 0:
            return

        self.model.train()

        if self.batchsize and self.batchsize < self.step:
            if self.step % self.decay_step == 0:
                self.optimizer.lr = 10 * self.base_lr / self.step

            ploader = sample_data(self.loader)
            for _ in range(num_iter):
                ctx, rew = next(ploader)
                ctx = ctx.to(self.device)
                rew = rew.to(dtype=torch.float32, device=self.device)

                self.model.zero_grad()
                pred = self.model(ctx)
                loss = self.criterion(pred.squeeze(1), rew)
                loss += self._fg_bonus(pred)
                loss.backward()
                self.optimizer.step()
        else:
            if self.step % self.decay_step == 0:
                self.optimizer.lr = self.base_lr / self.step

            ctx, rew = self.collector.fetch_batch()
            ctx = torch.stack(ctx, 0).to(self.device)
            rew = torch.tensor(rew, dtype=torch.float32, device=self.device)

            for _ in range(num_iter):
                self.model.zero_grad()
                pred = self.model(ctx)
                loss = self.criterion(pred.squeeze(1), rew)
                loss += self._fg_bonus(pred)
                loss.backward()
                self.optimizer.step()

        assert not torch.isnan(loss), "Loss became NaN!"
