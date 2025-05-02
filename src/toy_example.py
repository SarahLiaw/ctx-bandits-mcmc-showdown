import torch
from torch.nn.functional import one_hot
import pickle as pkl
import os

class LinearEnv(object):
    '''
    Environnement for the toy exmaple
    - info['context_size']: size of the contextual vector
    - info['phi']: function (context x number of arms) -> feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    - info['nb_arms']: number of arms
    - info['std_reward']: standard deviation of the reward distribution
    '''
    # def __init__(self, info):
    #     self.info = info
    #     self.theta_s = self.load_true_parameter()
    #     self.context_iterator = self.load_true_contextual_iterator()
    #     self.idx = 0

    def __init__(self, info):
        self.info = info

        data_file = os.environ.get("LINEAR_DATA_FILE")
        if data_file and os.path.exists(data_file):
            blob = torch.load(data_file)
            self.context_matrix = blob["ctx"]
            self.theta_s = blob["theta"]
        else:
            self.theta_s = self.load_true_parameter()
            self.context_matrix = torch.normal(0, 1, size=(self.info['T'], 1, self.info['context_size']))

        self.context_iterator = iter(self.context_matrix)
        self.idx = 0


    def load_true_contextual_iterator(self):
        context_matrix = torch.normal(0, 1, size=(self.info['T'], 1, self.info['context_size'] ))
        context_iterator = iter(context_matrix)
        return context_iterator

    def load_true_parameter(self):
        theta_s = torch.normal(0, 1, size=(self.info['d'], 1))
        return theta_s

    def context(self):
        return next(self.context_iterator)

    def reward(self, x, a):
        v = self.info['phi_a'](x, a, self.info['nb_arms'])
        mean = v @ self.theta_s
        return torch.normal(mean, self.info['std_reward'])

    def get_reward_star(self, x):
        r = self.info['phi'](x, self.info['nb_arms']) @ self.theta_s
        return r.max()

    def get_mean_reward(self, x, a):
        return self.info['phi_a'](x, a, self.info['nb_arms']) @ self.theta_s


class LogisticEnv(object):
    '''
    Environnement for the toy exmaple
    - info['context_size']: size of the contextual vector
    - info['phi']: function (context x number of arms) -> feature vectors
    - info['phi_a']: function (context x arm x number of arms) -> feature vector of the corresponding arm 
    - info['nb_arms']: number of arms
    - info['std_reward']: standard deviation of the reward distribution
    '''
    def __init__(self, info):
        self.info = info
        self.theta_s = self.load_true_parameter()
        self.context_iterator = self.load_true_contextual_iterator()
        self.idx = 0

    def load_true_contextual_iterator(self):
        context_matrix = torch.normal(0, 1, size=(self.info['T'], self.info['nb_arms'], self.info['context_size'] ))
        context_iterator = iter(context_matrix)
        return context_iterator

    def load_true_parameter(self):
        theta_s = torch.normal(0, 1, size=(self.info['d'], 1))
        return theta_s

    def context(self):
        contexts = next(self.context_iterator)
        contexts /= torch.linalg.norm(contexts, ord=2)
        return contexts

    def reward(self, x, a):
        v = self.info['phi_a'](x, a, self.info['nb_arms'])
        mean = 1 / (1 + torch.exp(-v @ self.theta_s))
        return torch.bernoulli(mean)

    def get_reward_star(self, x):
        scalar_product = self.info['phi'](x, self.info['nb_arms']) @ self.theta_s
        mean_reward = 1 / (1 + torch.exp(-scalar_product))
        return mean_reward.max()

    def get_mean_reward(self, x, a):
        scalar_product = self.info['phi'](x, self.info['nb_arms']) @ self.theta_s
        mean_reward = 1 / (1 + torch.exp(-scalar_product))
        return mean_reward[a]
