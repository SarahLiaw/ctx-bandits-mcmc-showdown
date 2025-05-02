import argparse
from src.MCMC import  MALATS, FGMALATS, FGLMCTS, LMCTS
from src.baseline import LinTS, LinUCB, Random
from src.game import GameToy, GameYahoo
import torch
from tqdm import tqdm
import pickle as pkl
import json
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config_path')

def format_agent(info):
    if info['agent'] == 'LMCTS':
        return LMCTS
    elif info['agent'] == 'FGLMCTS':
        return FGLMCTS
    elif info['agent'] == 'MALATS':
        return MALATS
    elif info['agent'] == 'FGMALATS':
        return FGMALATS
    elif info['agent'] == 'LinUCB':
        return LinUCB
    elif info['agent'] == 'LinTS':
        return LinTS
    elif info['agent'] == 'Random':
        return Random
    elif info['agent'] == 'SFGLMCTS':
        from src.MCMC import SFGLMCTS
        return SFGLMCTS
    elif info['agent'] == 'SFGMALATS':
        from src.MCMC import SFGMALATS
        return SFGMALATS
    elif info['agent'] == 'PLMCTS':
        from src.MCMC import PLMCTS
        return PLMCTS
    elif info['agent'] == 'PFGLMCTS':
        from src.MCMC import PFGLMCTS
        return PFGLMCTS
    elif info['agent'] == 'PSFGLMCTS':
        from src.MCMC import PSFGLMCTS
        return PSFGLMCTS
    elif info['agent'] == 'EpsGreedy':
        from src.baseline import EpsGreedy
        return EpsGreedy
    elif info['agent'] == 'HMCTS':
        from src.MCMC import HMCTS
        return HMCTS
    elif info['agent'] == 'FGHMCTS':
        from src.MCMC import FGHMCTS
        return FGHMCTS
    elif info['agent'] == 'SFGHMCTS':
        from src.MCMC import SFGHMCTS
        return SFGHMCTS
    elif info['agent'] == 'PHMCTS':
        from src.MCMC import PHMCTS
        return PHMCTS
    elif info['agent'] == 'PFGHMCTS':
        from src.MCMC import PFGHMCTS
        return PFGHMCTS
    elif info['agent'] == 'PSFGHMCTS':
        from src.MCMC import PSFGHMCTS
        return PSFGHMCTS
    elif info['agent'] == 'SVRGLMCTS':
        from src.MCMC import SVRGLMCTS
        return SVRGLMCTS
    elif info['agent'] == 'SVRGMALATS':
        from src.MCMC import SVRGMALATS
        return SVRGMALATS
    else:
        raise ValueError(info['agent'])

def load_config_file(config_path):
    f = open(config_path)
    info = json.load(f)
    info['agent'] = format_agent(info)
    if info['task_type'] == 'yahoo':
        info['phi'] = lambda x, y: x
        info['phi_a'] = lambda x, y, z: x[y, :]
        info['game'] = GameYahoo
    elif info['task_type'] == 'linear':
        info['phi'] = lambda x, nb_arms: torch.block_diag(*[x]*nb_arms)
        info['phi_a'] = lambda x, a, nb_arms: torch.block_diag(*[x]*nb_arms)[a, :]
        info['game'] = GameToy
    elif info['task_type'] == 'logistic':
        info['phi'] = lambda x, y: x
        info['phi_a'] = lambda x, a, nb_arms: x[a, :]
        info['game'] = GameToy
    else:
        raise ValueError(info['task_type'])
    return info

def run(config_path):
    info = load_config_file(config_path)
    info['out_dir'] = os.environ.get("LINEAR_OUT_DIR", "linear_results")

   # wandb.init(name=info['project_name'], config=info)
    wandb.init(
        name=f"lin_20d_{info['agent'].__name__}_seed{os.getenv('PYTHONHASHSEED', '0')}",
        group=info['agent'].__name__,
        config=info,
        reinit=True
        )
    info['game'](info).run()
    wandb.finish()

if __name__ == '__main__':
    args = parser.parse_args()
    run(args.config_path)