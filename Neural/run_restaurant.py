import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import random
import numpy as np

import torch
import multiprocessing as mp
from algo.lmcts import LMCTS
from algo.langevin import LangevinMC
from algo.baselines import NeuralTS, LinTS, NeuralUCB, NeuralEpsGreedy, NeuralLinUCB
from algo.fg_neuralts import FGNeuralTS
from algo.fg_lmcts import FGLMCTS

from train_utils.restaurant_adapter import load_restaurant_for_bandit
from train_utils.helper import get_model
from train_utils.losses import BCELoss, MSELoss
from train_utils.dataset import Collector


try:
    import wandb
except ImportError:
    wandb = None


def one_hot(img, num_arm):
    cxt = torch.zeros((num_arm, 3 * num_arm, 32, 32), device=img.device)
    for i in range(num_arm):
        cxt[i, 3 * i: 3 * i + 3, :, :] = img[0]
    return cxt


def run(config, args):
    seed = random.randint(1, 10000)
    print(f'Random seed: {seed}')
    torch.manual_seed(seed)
    if args.log and wandb:
        group = config['group'] if 'group' in config else None
        run = wandb.init(
            project=config['project'],
            group=group,
            config=config)
        config = wandb.config
    print('Starting restaurant bandit experiment...')
    
    # Parse configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    T = config['T']
    dim_context = config['dim_context']
    num_arm = config['num_arm']
    
    print("Loading restaurant dataset...")
    dataset, all_contexts, all_rewards = load_restaurant_for_bandit(feature_dim=dim_context)
    print(f"Dataset loaded: {len(all_contexts)} users, {dataset.num_arms} restaurants")
    
    model_config = {
        'model': config['model'],
        'dim_context': dim_context,
        'output_dim': 1,
        'layers': config.get('layers', [50, 50, 50]),
        'act': config.get('act', 'LeakyReLU')
    }
    model = get_model(model_config, device)
    
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=config.get('lr', 0.01))
    
    if config.get('loss', 'L2') == 'BCE':
        criterion = BCELoss()
    else:
        criterion = MSELoss()
    
    collector = Collector()
    
    if config['algo'] == 'FGNeuralTS':
        agent = FGNeuralTS(
            num_arm=num_arm,
            dim_context=dim_context,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            collector=collector,
            nu=config.get('nu', 0.00001),
            batch_size=32,
            device=device,
            name='FGNeuralTS',
            feel_good=config.get('feel_good', True),
            fg_mode=config.get('fg_mode', 'hard'),
            lambda_fg=config.get('lambda_fg', 0.01),
            b_fg=config.get('b_fg', 1.0),
            smooth_s=config.get('smooth_s', 10.0)
        )
    elif config['algo'] == 'SFGNeuralTS':
        agent = FGNeuralTS(
            num_arm=num_arm,
            dim_context=dim_context,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            collector=collector,
            nu=config.get('nu', 0.00001),
            batch_size=32,
            device=device,
            name='SFGNeuralTS',
            feel_good=config.get('feel_good', True),
            fg_mode=config.get('fg_mode', 'smooth'),
            lambda_fg=config.get('lambda_fg', 0.01),
            b_fg=config.get('b_fg', 1.0),
            smooth_s=config.get('smooth_s', 10.0)
        )
    elif config['algo'] == 'FGLMCTS':
        beta_inv = config.get('beta_inv', 0.01)
        langevin_optimizer = LangevinMC(
            model.parameters(),
            lr=config.get('lr', 0.01),
            beta_inv=beta_inv,
            weight_decay=config.get('reg', 1.0),
            device=device
        )
        agent = FGLMCTS(
            model=model,
            optimizer=langevin_optimizer,
            criterion=criterion,
            collector=collector,
            batch_size=32,
            device=device,
            name='FGLMCTS',
            feel_good=config.get('feel_good', True),
            fg_mode=config.get('fg_mode', 'hard'),
            lambda_fg=config.get('lambda_fg', 0.01),
            b_fg=config.get('b_fg', 1.0),
            smooth_s=config.get('smooth_s', 10.0)
        )
    elif config['algo'] == 'SFGLMCTS':
        beta_inv = config.get('beta_inv', 0.01)
        langevin_optimizer = LangevinMC(
            model.parameters(),
            lr=config.get('lr', 0.01),
            beta_inv=beta_inv,
            weight_decay=config.get('reg', 1.0),
            device=device
        )
        agent = FGLMCTS(
            model=model,
            optimizer=langevin_optimizer,
            criterion=criterion,
            collector=collector,
            batch_size=32,
            device=device,
            name='SFGLMCTS',
            feel_good=config.get('feel_good', True),
            fg_mode=config.get('fg_mode', 'smooth'),
            lambda_fg=config.get('lambda_fg', 0.01),
            b_fg=config.get('b_fg', 1.0),
            smooth_s=config.get('smooth_s', 10.0)
        )
    elif config['algo'] == 'NeuralTS':
        agent = NeuralTS(
            num_arm=num_arm,
            dim_context=dim_context,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            collector=collector,
            nu=config.get('nu', 0.00001),
            batch_size=32,
            device=device,
            name='NeuralTS'
        )
    elif config['algo'] == 'NeuralUCB':
        agent = NeuralUCB(
            num_arm=num_arm,
            dim_context=dim_context,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            collector=collector,
            nu=config.get('nu', 0.00001),
            batch_size=32,
            device=device,
            name='NeuralUCB'
        )
    elif config['algo'] == 'NeuralEpsGreedy':
        agent = NeuralEpsGreedy(
            num_arm=num_arm,
            dim_context=dim_context,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            collector=collector,
            eps=config.get('eps', 0.01),
            batch_size=32,
            device=device,
            name='NeuralEpsGreedy'
        )
    elif config['algo'] == 'LinTS':
        agent = LinTS(
            num_arm=num_arm,
            dim_context=dim_context,
            nu=config.get('nu', 0.00001),
            reg=config.get('reg', 1.0),
            device=device,
            name='LinTS'
        )
    elif config['algo'] == 'LMCTS':
        beta_inv = config.get('beta_inv', 0.01)
        langevin_optimizer = LangevinMC(
            model.parameters(),
            lr=config.get('lr', 0.01),
            beta_inv=beta_inv,
            weight_decay=config.get('reg', 1.0),
            device=device
        )
        agent = LMCTS(
            model=model,
            optimizer=langevin_optimizer,
            criterion=criterion,
            collector=collector,
            batch_size=32,
            device=device,
            name='LMCTS'
        )
    else:
        raise ValueError(f"Algorithm {config['algo']} not supported")
    
    pbar = tqdm(range(T), dynamic_ncols=True, smoothing=0.1)
    reward_history = []
    accum_regret = 0
    
    for e in pbar:
        if e < len(all_contexts):
            contexts = all_contexts[e]
            true_rewards = all_rewards[e]
        else:
            idx = np.random.randint(0, len(all_contexts))
            contexts = all_contexts[idx]
            true_rewards = all_rewards[idx]
            
        contexts = contexts.to(device)
        
        formatted_contexts = torch.zeros((num_arm, dim_context), device=device)
        for i in range(min(num_arm, len(contexts))):
            formatted_contexts[i] = contexts[i]
            
        arm = agent.choose_arm(formatted_contexts)
        
        reward = true_rewards[arm].item()
        
        best_reward = true_rewards.max().item()
        regret = best_reward - reward
        
        agent.receive_reward(arm, formatted_contexts[arm], reward)
        agent.update_model(num_iter=config.get('num_iter', 70))
        
        reward_history.append(reward)
        accum_regret += regret
        
        pbar.set_description(f'Accumulated regret: {accum_regret:.4f}')
        
        if wandb and args.log:
            wandb.log({
                'Regret': accum_regret,
                'Reward': reward,
                'Step': e
            })
    
    if wandb and args.log:
        run.finish()
    print('Restaurant bandit experiment completed successfully!')


if __name__ == '__main__':
    parser = ArgumentParser(description="Restaurant contextual bandit experiment runner")
    parser.add_argument('--config_path', type=str,
                        default='configs/restaurant/restaurant-restaurant-lmcts.yaml')
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    if args.repeat == 1:
        run(config, args)
    else:
        for i in range(args.repeat):
            p = mp.Process(target=run, args=(config, args))
            p.start()

