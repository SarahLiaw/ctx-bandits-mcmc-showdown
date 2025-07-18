"""
Adapter module to integrate the Jester dataset with the neural bandit framework.
"""
import torch
import numpy as np
from torch.utils.data import Dataset

class JesterBanditAdapter(Dataset):
    def __init__(self, num_jokes=8, num_users=14936):
        super().__init__()
        self.contexts = np.load("data/jester_contexts.npy")
        self.rewards = np.load("data/jester_rewards.npy")
        self.contexts = self.contexts[:num_users]
        self.rewards = self.rewards[:num_users]
        self.contexts = torch.tensor(self.contexts, dtype=torch.float32)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32)
        self.num_users = self.contexts.shape[0]
        self.num_arms = self.rewards.shape[1]
        self.dim_context = self.contexts.shape[1]

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return self.contexts[idx], self.rewards[idx]

def load_jester_for_bandit(num_jokes=8, num_users=14936):
    dataset = JesterBanditAdapter(num_jokes=num_jokes, num_users=num_users)
    return dataset, dataset.contexts, dataset.rewards

if __name__ == "__main__":
    dataset, contexts, rewards = load_jester_for_bandit()
    print(f"Contexts shape: {contexts.shape}")
    print(f"Rewards shape: {rewards.shape}")
    batch_contexts, batch_rewards = dataset.get_batch(16)
    print(f"Batch contexts shape: {batch_contexts.shape}")
    print(f"Batch rewards shape: {batch_rewards.shape}")
