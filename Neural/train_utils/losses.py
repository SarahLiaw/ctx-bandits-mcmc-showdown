import torch
import torch.nn as nn
from functools import partial


def NLL(pred, target, reduction='sum'):
    '''
    Negative log likelihood loss given x^\top theta

    Pred: x^\top \theta, (N, 1)
    '''
    if reduction == 'sum':
        loss = torch.sum(torch.log(1 + torch.exp(pred)) - target * pred)
    else:
        loss = torch.mean(torch.log(1 + torch.exp(pred)) - target * pred)
    return loss



# Simple factory wrappers to keep backward compatibility with code that expects
# callable classes/functions named `BCELoss` and `MSELoss` (similar to PyTorch).
# They simply return the appropriate loss instances.

def BCELoss(reduction: str = 'mean'):
    """Return a BCEWithLogitsLoss instance (mirrors torch.nn.BCEWithLogitsLoss)."""
    return nn.BCEWithLogitsLoss(reduction=reduction)


def MSELoss(reduction: str = 'mean'):
    """Return an MSELoss instance (mirrors torch.nn.MSELoss)."""
    return nn.MSELoss(reduction=reduction)


def construct_loss(name, reduction='mean'):
    if name == 'L1':
        func = nn.L1Loss(reduction=reduction)
    elif name == 'L2':
        func = nn.MSELoss(reduction=reduction)
    elif name == 'BCE':
        func = nn.BCEWithLogitsLoss(reduction=reduction)
    elif name == 'Logistic':
        func = partial(NLL, reduction=reduction)
    else:
        func = nn.MSELoss(reduction=reduction)
        print('Invalid loss function name. Setting loss function to MSELoss by default...')
    return func