import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
import random

class Random(object):
    '''
    Random bandit: Choose a random arm at each iteration
    '''
    def __init__(self, info):
        pass

    def choose_arm(self, features, arm_idx):
        return np.random.randint(0, len(arm_idx))

    def update(self, action, reward, features, arm_idx):
        pass

class LinUCB(object):
    '''
    Linear UCB bandit:
    - info['d']: parameter dimension
    - info['nb_arms']: number of arms
    - info['std_prior']: standard deviation of the gaussian prior
    - info['phi']: function (context x number of arms) -> all feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    - info['alpha']: controls ther size of the confidence interval
    '''
    def __init__(self, info): 
        self.info = info
        self.Vt_inv = torch.eye(self.info['d']) / self.info['std_prior']
        self.bt = torch.zeros((self.info['d'], 1))
        self.idx = 1

    def choose_arm(self, features, arm_idx):
        v = self.info['phi'](features, self.info['nb_arms'])
        norm = torch.sqrt((v @ self.Vt_inv @ v.T).diag())
        beta = self.info['alpha']
        p = v @ (self.Vt_inv @ self.bt).squeeze() + beta * norm
        return p.argmax()

    def update(self, action, reward, features, arm_idx):
        v = self.info['phi_a'](features, action, self.info['nb_arms']).unsqueeze(1)
        omega = self.Vt_inv @ v
        self.Vt_inv -= omega @ omega.T / (1 + torch.dot(omega.squeeze(), v.squeeze()))
        self.bt +=  reward * v
        self.idx += 1

class LinTS(object):
    '''
    Linear Thompson Sampling bandit:
    - info['d']: parameter dimension
    - info['eta']: inverse of temperature, controls the variance of the posterior distribution
    - info['nb_arms']: number of arms
    - info['std_prior']: standard deviation of the gaussian prior
    - info['phi']: function (context x number of arms) -> all feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    '''
    def __init__(self, info):
        self.info = info
        self.Vt_inv = torch.eye(self.info['d']) * self.info['eta'] / self.info['std_prior']
        self.bt = torch.zeros((self.info['d'], 1))
        self.idx = 1

    def choose_arm(self, feature, arm_idx):
        theta = self.sample_posterior(arm_idx)
        rewards = self.info['phi'](feature, self.info['nb_arms']) @ theta
        return rewards.argmax()
        
    def sample_posterior(self, arm_idx):
        theta = MultivariateNormal((self.Vt_inv @ self.bt).squeeze(), (1 / self.info['eta']) * self.Vt_inv).sample((1,)).T
        return theta

    def update(self, action, reward, features, arm_idx):
        v = self.info['phi_a'](features, action, self.info['nb_arms']).unsqueeze(1)
        self.bt +=  reward * v
        omega = self.Vt_inv @ v
        self.Vt_inv -= omega @ omega.T / (1 + torch.dot(omega.squeeze(), v.squeeze()))
        self.idx += 1

class EpsGreedy(object):
    '''
    Epsilon-Greedy bandit algorithm (Non-Contextual).

    Selects the arm with the highest estimated average reward greedily,
    but explores a random arm with probability epsilon.
    Can optionally force exploration of each arm once initially.
    Can optionally decay epsilon over time.

    Required info parameters:
    - info['nb_arms']: Total number of arms.
    - info['epsilon']: The base probability for exploration (float between 0 and 1).

    Optional info parameters:
    - info['epsilon_decay']: Boolean, whether epsilon should decay as epsilon/sqrt(t). Defaults to True.
    - info['force_explore_first']: Boolean, whether to force trying each arm at least once before decaying epsilon. Defaults to True.
    '''
    def __init__(self, info):
        self.info = info
        if 'nb_arms' not in self.info or not isinstance(self.info['nb_arms'], int) or self.info['nb_arms'] <= 0:
            raise ValueError("info['nb_arms'] must be a positive integer.")
        self.nb_arms = self.info['nb_arms']

        if 'epsilon' not in self.info or not isinstance(self.info['epsilon'], (float, int)) or not (0 <= self.info['epsilon'] <= 1):
             raise ValueError("info['epsilon'] must be a float between 0 and 1.")
        self.epsilon_base = self.info['epsilon']

        self.decay = self.info.get('epsilon_decay', True)
        self.force_explore = self.info.get('force_explore_first', True) 
        
        self.num_draw = torch.zeros(self.nb_arms, dtype=torch.long)
        self.avg_rewards = torch.zeros(self.nb_arms, dtype=torch.float32)
        self.step = 0

        print(f"Initialized EpsGreedy: Arms={self.nb_arms}, Epsilon={self.epsilon_base}, Decay={self.decay}, ForceExplore={self.force_explore}")

    def choose_arm(self, features, arm_idx):
        """
        Chooses an arm based on the epsilon-greedy strategy.
        Ignores features and arm_idx (non-contextual).
        """
        self.step += 1
        if self.force_explore:
             untried_arms = torch.where(self.num_draw == 0)[0]
             if len(untried_arms) > 0:
                 chosen_arm = random.choice(untried_arms.tolist())
                 return chosen_arm
        current_epsilon = self.epsilon_base
        if self.decay:
            decay_step = max(1, self.step - (self.nb_arms if self.force_explore else 0))
            current_epsilon = self.epsilon_base / np.sqrt(decay_step) 
        current_epsilon = min(current_epsilon, 1.0)
        if random.uniform(0, 1) < current_epsilon:
            chosen_arm = random.randrange(self.nb_arms)
        else:
            max_reward = torch.max(self.avg_rewards)
            best_arms = torch.where(self.avg_rewards == max_reward)[0]
            chosen_arm = random.choice(best_arms.tolist())
        return chosen_arm

    def update(self, action, reward, features, arm_idx):
        """
        Updates the agent's estimates based on the received reward.
        Ignores features and arm_idx (non-contextual).
        'action' is the index of the arm that was pulled.
        """
        if action < 0 or action >= self.nb_arms:
             print(f"Warning: EpsGreedy received invalid action {action} for {self.nb_arms} arms.")
             return
        n = self.num_draw[action].item()
        current_avg = self.avg_rewards[action].item()
        new_avg = current_avg + (reward - current_avg) / (n + 1)
        self.avg_rewards[action] = new_avg
        self.num_draw[action] += 1
        
