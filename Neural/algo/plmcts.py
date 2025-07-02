import torch
from torch.utils.data import DataLoader
from .lmcts import LMCTS
from .precond_langevin import PrecondLangevinMC
from train_utils.dataset import sample_data


class PrecondLMCTS(LMCTS):
    def __init__(self, *args, lambda_reg: float = 1.0, **kwargs):
        optimizer = PrecondLangevinMC(
            params=args[0].parameters(),
            lr=kwargs.pop("lr"),
            beta_inv=kwargs.pop("beta_inv"),
            weight_decay=0.,
            device=kwargs.get("device", 'cpu')
        )
        super().__init__(*args, optimizer=optimizer, **kwargs)
        self.lambda_reg = lambda_reg

    def _compute_preconditioner(self):
        d = self.model.weight.numel()
        V = torch.eye(d, device=self.device) * self.lambda_reg
        if len(self.collector.context):
            X = torch.stack(self.collector.context).to(self.device)
            V += X.t() @ X
        P = torch.linalg.inv(V)
        return P

    def update_model(self, num_iter: int = 1):
        self.step += 1
        if self.reduce and self.step % self.reduce != 0:
            return
        self.model.train()

        if self.batchsize and self.batchsize < self.step:
            data_iter = sample_data(self.loader)
            ctx, rew = next(data_iter)
            ctx, rew = ctx.to(self.device), rew.to(dtype=torch.float32, device=self.device)
        else:
            ctx, rew = self.collector.fetch_batch()
            ctx = torch.stack(ctx).to(self.device)
            rew = torch.tensor(rew, dtype=torch.float32, device=self.device)

        self.model.zero_grad()
        pred = self.model(ctx).squeeze(1)
        loss = self.criterion(pred, rew)
        loss.backward()

        P = self._compute_preconditioner()
        self.optimizer.set_preconditioner(P)
        self.optimizer.step()

        assert not torch.isnan(loss), "NaN loss!"

    @property
    def name(self):
        return "Precond-LMCTS"
