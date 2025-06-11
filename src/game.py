import torch
from tqdm import tqdm
from src.dataset import Dataset
import numpy as np
from src.toy_example import LinearEnv, LogisticEnv
import wandb
import os
from pathlib import Path


class GameToy(object):
    '''
    - info['env']: environnement class
    - info['agent']: agent class
    - info['T']: time horizon
    - info['d']: parameter dimension
    '''
    
    def __init__(self, info):
        self.info = info
        if self.info['task_type'] == 'linear':
            self.env = LinearEnv(info)
        elif self.info['task_type'] == 'logistic':
            self.env = LogisticEnv(info)
        elif self.info['task_type'] == 'wheel':
            # For wheel bandit, we'll implement the environment methods directly in this class
            self.context_size = info['context_size']
            self.nb_arms = info['nb_arms']
            self.d = info['d']
            self.delta = info.get('delta', 0.99)
            self.mean_v = torch.tensor(info.get('mean_v', [0.1] * (self.nb_arms-1) + [0.15]))
            self.std_v = torch.tensor(info.get('std_v', [0.01] * self.nb_arms))
            self.mu_large = info.get('mu_large', 1.0)
            self.std_large = info.get('std_large', 0.01)
            # Generate the optimal arm for each context
            self.theta_star = torch.randn(self.d, 1)
        else:
            raise ValueError(f"Unsupported task_type: {info['task_type']}")
            
        self.agent = self.info['agent'](info)
        self.arm_idx = torch.arange(self.info['nb_arms'] if 'nb_arms' in info else info['d'])

    def context(self):
        """Generate a random context for the wheel bandit problem."""
        if self.info['task_type'] == 'wheel':
            # For wheel bandit, generate a random context vector
            return torch.randn(1, self.context_size)
        else:
            return self.env.context()
            
    def reward(self, feature, action):
        """Generate reward for the given action."""
        if self.info['task_type'] == 'wheel':
            # Implement wheel bandit reward function
            if isinstance(action, torch.Tensor):
                action = action.item()
                
            # Get the mean reward for this action
            mu = self.mean_v[action] if action < self.nb_arms - 1 else self.mu_large
            std = self.std_v[action] if action < self.nb_arms - 1 else self.std_large
            
            # Add noise to the reward
            return mu + torch.randn(1).item() * std
        else:
            return self.env.reward(feature, action)
    
    def get_reward_star(self, feature):
        """Get the optimal reward for the given context."""
        if self.info['task_type'] == 'wheel':
            # For wheel bandit, the optimal action is always the last one
            return self.mu_large
        else:
            return self.env.get_reward_star(feature)
            
    def get_mean_reward(self, feature, action):
        """Get the mean reward for the given action."""
        if self.info['task_type'] == 'wheel':
            if isinstance(action, torch.Tensor):
                action = action.item()
            return self.mean_v[action] if action < self.nb_arms - 1 else self.mu_large
        else:
            return self.env.get_mean_reward(feature, action)
    
    def play(self, t, cum_regret):
        feature = self.context()
        action = self.agent.choose_arm(feature, self.arm_idx)
        reward = self.reward(feature, action)
        mean_best_reward = self.get_reward_star(feature)
        mean_reward = self.get_mean_reward(feature, action)
        if t > 0:
            cum_regret[t] = cum_regret[t-1] + mean_best_reward - mean_reward
        else:
            cum_regret[0] = mean_best_reward - mean_reward
        wandb.log({'cum_regret': cum_regret[t]})
        self.agent.update(action, reward, feature, self.arm_idx)

    def run(self):
        print(f"[GameToy] start run; T={self.info['T']}  task={self.info['task_type']}")

        cum_regret = torch.zeros(self.info['T'])
        for t in range(self.info['T']):
            self.play(t, cum_regret)
        seed = os.environ.get("PYTHONHASHSEED", "0")

        result_dir = Path(self.info.get("out_dir", "linear_results")) / self.info["agent"].__name__ / f"seed{seed}"

        result_dir.mkdir(parents=True, exist_ok=True)
        torch.save(cum_regret.cpu(), result_dir / "cum_regret.pt")
        return cum_regret

class GameYahoo(object):
    def __init__(self, info):
        self.info = info
        self.dataset = Dataset()
        files = ('dataset/R6/ydata-fp-td-clicks-v1_0.20090503')
        self.dataset.get_yahoo_events(files, 3000000)
        np.random.shuffle(self.dataset.events)
        self.dataset.events = self.dataset.events[: info['yahoo_size']]
        self.agent = self.info['agent'](info)

    def run(self):
        G_learn = 0  
        T_learn = 0
        learn = []
        for _, event in enumerate(tqdm(self.dataset.events)):
            displayed = event[0]
            reward = event[1]
            user_feature = torch.tensor(event[2])
            pool_idx = torch.tensor(event[3])
            article_feature = torch.tensor(self.dataset.features[pool_idx])
            features = torch.cat((user_feature.repeat((len(pool_idx), 1)), article_feature), dim=1).float()
            chosen = self.agent.choose_arm(features, pool_idx)
            if chosen == displayed:
                G_learn += event[1]
                T_learn += 1
                self.agent.update(displayed, reward, features, pool_idx)
                wandb.log({'ctr': G_learn / T_learn})
                learn.append(G_learn / T_learn)
        return learn
            
            
            