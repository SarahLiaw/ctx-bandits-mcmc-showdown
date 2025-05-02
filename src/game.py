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
        else:
            raise NotImplementedError
        self.agent = self.info['agent'](info)
        self.arm_idx = torch.arange(self.info['d'])

    def play(self, t, cum_regret):
        feature = self.env.context()
        action = self.agent.choose_arm(feature, self.arm_idx)
        reward = self.env.reward(feature, action)
        mean_best_reward = self.env.get_reward_star(feature)
        mean_reward = self.env.get_mean_reward(feature, action)
        if t > 0:
            cum_regret[t] = cum_regret[t-1] + mean_best_reward - mean_reward
        else:
            cum_regret[0] = mean_best_reward - mean_reward
        wandb.log({'cum_regret': cum_regret[t]})
        self.agent.update(action, reward, feature, self.arm_idx)

    def run(self):
        cum_regret = torch.zeros(self.info['T'])
        for t in range(self.info['T']):
            self.play(t, cum_regret)
        seed = os.environ.get("PYTHONHASHSEED", "0")

        if self.info['task_type'] == 'linear':
            result_dir = Path(self.info.get("out_dir", "linear_results")) / self.info["agent"].__name__ / f"seed{seed}"
        elif self.info['task_type'] == 'logistic':
            result_dir = Path(self.info.get("out_dir", "logistic_results")) / self.info["agent"].__name__ / f"seed{seed}"
        else:
            raise NotImplementedError

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
            
            
            