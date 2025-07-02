"""
Run LMCTS on the Jester dataset.
"""
import os
import argparse
import yaml
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from train_utils.jester_adapter import load_jester_for_bandit
from train_utils.losses import BCELoss, MSELoss
from train_utils.helper import get_model
from algo.langevin import LangevinMC
from algo.lmcts import LMCTS
from algo.baselines import NeuralTS, LinTS, GLMTSL
from train_utils.dataset import Collector

def parse_args():
    parser = argparse.ArgumentParser(description='Run LMCTS on the Jester dataset')
    parser.add_argument('--config_path', type=str, default='configs/uci/jester-lmcts.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--log', action='store_true',
                        help='Whether to log to wandb')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of times to repeat the experiment')
    return parser.parse_args()

def run_experiment(config, device, use_wandb=False):
    # Set up logging
    if use_wandb:
        import wandb
        wandb.init(project=config.get('project', 'ContextualBandit-Jester'))
        wandb.config.update(config)
    
    # Load the Jester dataset
    num_jokes = config.get('num_arm', 100)
    num_users = config.get('num_data', 10000)
    dataset, contexts, rewards = load_jester_for_bandit(num_jokes, num_users)
    
    # Set up the model
    dim_context = dataset.dim_context
    num_arms = dataset.num_arms
    
    # Create the model
    # Prepare model config that matches helper.py get_model() function
    model_config = {
        'model': config['model'],
        'dim_context': dim_context,
        'output_dim': 1,
        'layers': config.get('layers', [50, 50, 50]),
        'act': config.get('act', 'LeakyReLU')
    }
    model = get_model(model_config, device)
    
    # Set up the optimizer
    optimizer = Adam(model.parameters(), lr=config.get('lr', 0.01))
    
    # Set up the loss function
    if config.get('loss', 'MSE') == 'BCE':
        criterion = BCELoss()
    else:
        criterion = MSELoss()
    
    # Set up the collector
    collector = Collector()
    
    # Set up the algorithm
    if config['algo'] == 'LMCTS':
        # Set up the optimizer for LMCTS
        beta_inv = config.get('beta_inv', 0.01) 
        langevin_optimizer = LangevinMC(
            model.parameters(), 
            lr=config.get('lr', 0.01),
            beta_inv=beta_inv, 
            weight_decay=config.get('reg', 1.0),
            device=device
        )
        
        # Create LMCTS agent
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
        raise ValueError(f"Algorithm {config['algo']} not supported for Jester dataset")
    
    # Run the bandit algorithm
    T = config.get('T', 10000)
    regrets = []
    rewards_history = []
    
    # For each round
    for t in range(T):
        # Get a random user
        user_idx = np.random.randint(0, dataset.num_samples)
        user_rewards = rewards[user_idx]
        
        # Format contexts for the algorithm
        formatted_contexts = torch.zeros((num_arms, dim_context), device=device)
        for i in range(num_arms):
            formatted_contexts[i] = contexts[i]
        
        # Choose an arm (joke)
        arm = agent.choose_arm(formatted_contexts)
        
        # Get the reward
        reward = user_rewards[arm].item()
        
        # Calculate regret (difference between best possible reward and received reward)
        best_reward = user_rewards.max().item()
        regret = best_reward - reward
        
        # Update the agent
        agent.receive_reward(arm, formatted_contexts[arm], reward)
        agent.update_model(num_iter=config.get('num_iter', 70))
        
        # Record metrics
        regrets.append(regret)
        rewards_history.append(reward)
        
        # Log every 100 steps
        if t % 100 == 0:
            cum_regret = np.sum(regrets)
            avg_reward = np.mean(rewards_history[-100:]) if rewards_history else 0
            print(f"Step {t}/{T}: Cumulative Regret = {cum_regret:.4f}, Avg Reward = {avg_reward:.4f}")
            
            if use_wandb:
                wandb.log({
                    'step': t,
                    'cumulative_regret': cum_regret,
                    'average_reward': avg_reward,
                    'regret': regret,
                    'reward': reward
                })
    
    # Final results
    cum_regret = np.sum(regrets)
    avg_reward = np.mean(rewards_history)
    print(f"Final results: Cumulative Regret = {cum_regret:.4f}, Avg Reward = {avg_reward:.4f}")
    
    if use_wandb:
        wandb.finish()
    
    return cum_regret, avg_reward

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Run the experiment multiple times
    results = []
    for i in range(args.repeat):
        print(f"Run {i+1}/{args.repeat}")
        regret, reward = run_experiment(config, device, args.log)
        results.append((regret, reward))
    
    # Print average results
    avg_regret = np.mean([r[0] for r in results])
    avg_reward = np.mean([r[1] for r in results])
    print(f"Average over {args.repeat} runs: Cumulative Regret = {avg_regret:.4f}, Avg Reward = {avg_reward:.4f}")

if __name__ == "__main__":
    main()
