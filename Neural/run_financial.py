"""
Run LMCTS on financial datasets.
"""
import os
import argparse
import yaml
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from train_utils.financial_adapter import load_financial_for_bandit
from train_utils.losses import BCELoss, MSELoss
from train_utils.helper import get_model
from algo.langevin import LangevinMC
from algo.lmcts import LMCTS
from algo.baselines import NeuralTS, LinTS, GLMTSL
from train_utils.dataset import Collector

def parse_args():
    parser = argparse.ArgumentParser(description='Run LMCTS on financial datasets')
    parser.add_argument('--config_path', type=str, default='configs/uci/financial-lmcts.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    parser.add_argument('--log', action='store_true',
                        help='Whether to log to wandb')
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of times to repeat the experiment')
    parser.add_argument('--num_assets', type=int, default=10,
                        help='Number of financial assets to use')
    parser.add_argument('--lookback_days', type=int, default=30,
                        help='Number of days to look back for features')
    return parser.parse_args()

def run_experiment(config, device, num_assets, lookback_days, use_wandb=False):
    # Set up logging
    if use_wandb:
        import wandb
        # Create descriptive run name including algorithm, model type, and asset count
        run_name = f"LMCTS-{config['model']}-{num_assets}assets-{lookback_days}days"
        wandb.init(
            project=config.get('project', 'ContextualBandit-Financial'),
            name=run_name,
            config=config
        )
    
    # Load the financial dataset
    dataset, all_contexts, all_rewards = load_financial_for_bandit(
        num_assets=num_assets,
        lookback_days=lookback_days,
        feature_dim=config.get('dim_context', 20)
    )
    
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
        raise ValueError(f"Algorithm {config['algo']} not supported for financial dataset")
    
    # Run the bandit algorithm
    T = min(config.get('T', 10000), len(dataset))
    regrets = []
    rewards_history = []
    portfolio_value = 1.0  # Start with $1
    portfolio_history = [portfolio_value]
    
    # For each round
    for t in range(T):
        # Get context and rewards for this day
        if t < len(all_contexts):
            contexts = all_contexts[t]
            true_rewards = all_rewards[t]
        else:
            # If we've gone through all data, sample randomly
            idx = np.random.randint(0, len(all_contexts))
            contexts = all_contexts[idx]
            true_rewards = all_rewards[idx]
            
        contexts = contexts.to(device)
        
        # Format contexts for the algorithm
        formatted_contexts = torch.zeros((num_arms, dim_context), device=device)
        for i in range(num_arms):
            formatted_contexts[i] = contexts[i]
            
        # Choose an asset
        arm = agent.choose_arm(formatted_contexts)
        
        # Get the reward (return)
        reward = true_rewards[arm].item()
        
        # Calculate regret (difference between best possible return and received return)
        best_reward = true_rewards.max().item()
        regret = best_reward - reward
        
        # Update portfolio value
        portfolio_value *= (1 + reward/100)
        portfolio_history.append(portfolio_value)
        
        # Update the agent
        agent.receive_reward(arm, formatted_contexts[arm], reward)
        agent.update_model(num_iter=config.get('num_iter', 70))
        
        # Record metrics
        regrets.append(regret)
        rewards_history.append(reward)
        
        # Log every step
        cum_regret = np.sum(regrets)
        avg_reward = np.mean(rewards_history[-10:]) if rewards_history else 0
        print(f"Step {t}/{T}: Cumulative Regret = {cum_regret:.4f}, Avg Return = {avg_reward:.4f}, Portfolio = ${portfolio_value:.2f}")
            
        if use_wandb:
            # Log metrics focusing on cumulative regret
            wandb.log({
                'step': t,
                'cumulative_regret': cum_regret,
                'average_return': avg_reward,
                'portfolio_value': portfolio_value
            })
    
    # Final results
    cum_regret = np.sum(regrets)
    avg_reward = np.mean(rewards_history)
    print(f"Final results: Cumulative Regret = {cum_regret:.4f}, Avg Return = {avg_reward:.4f}, Final Portfolio = ${portfolio_value:.2f}")
    
    if use_wandb:
        # Enhanced final metrics for better experiment tracking
        wandb.log({
            'final_cumulative_regret': cum_regret,
            'final_portfolio_value': portfolio_value,
            'average_daily_return': avg_reward
        })
        # Create summary metrics that appear on the run overview
        wandb.run.summary['final_cumulative_regret'] = cum_regret
        wandb.run.summary['final_portfolio_value'] = portfolio_value
        wandb.run.summary['average_daily_return'] = avg_reward
        wandb.finish()
    
    return cum_regret, avg_reward, portfolio_value

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
        regret, reward, portfolio = run_experiment(config, device, args.num_assets, args.lookback_days, args.log)
        results.append((regret, reward, portfolio))
    
    # Print average results
    avg_regret = np.mean([r[0] for r in results])
    avg_reward = np.mean([r[1] for r in results])
    avg_portfolio = np.mean([r[2] for r in results])
    print(f"Average over {args.repeat} runs: Cumulative Regret = {avg_regret:.4f}, Avg Return = {avg_reward:.4f}, Final Portfolio = ${avg_portfolio:.2f}")

if __name__ == "__main__":
    main()
