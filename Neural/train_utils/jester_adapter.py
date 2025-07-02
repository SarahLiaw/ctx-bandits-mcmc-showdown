"""
Adapter module to integrate the Jester dataset with the neural bandit framework.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from prepare_jester import JesterDataset, prepare_jester_dataset

class JesterBanditAdapter(Dataset):
    """
    Adapter class to use the Jester dataset with the neural bandit algorithms.
    This class converts the Jester dataset into the format expected by the bandit algorithms.
    Implements the interface required by the Collector class used in LMCTS.
    """
    def __init__(self, num_jokes=100, num_users=10000):
        super(JesterBanditAdapter, self).__init__()
        
        # Load the Jester dataset
        jester_dataset = JesterDataset(num_jokes, num_users)
        self.contexts = jester_dataset.contexts
        self.rewards = jester_dataset.rewards
        
        self.num_arms = jester_dataset.num_jokes
        self.dim_context = jester_dataset.dim_context
        self.num_samples = jester_dataset.num_users
        
        # Normalize contexts if not already done
        norms = torch.norm(self.contexts, dim=1, keepdim=True)
        self.contexts = self.contexts / norms
        
        print(f"Jester dataset loaded with {self.num_arms} arms and {self.num_samples} samples")
        print(f"Context dimension: {self.dim_context}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Return all contexts for all arms and the rewards for a specific user.
        This format matches what the neural bandit algorithms expect.
        """
        # Format required by LMCTS: each arm's context and the corresponding reward
        return self.contexts[idx % self.num_arms], self.rewards[idx // self.num_arms][idx % self.num_arms]
    
    def get_batch(self, batch_size=32):
        """Get a random batch of samples."""
        indices = torch.randint(0, self.num_samples, (batch_size,))
        batch_contexts = self.contexts.repeat(batch_size, 1, 1)
        batch_rewards = torch.tensor(self.rewards[indices], dtype=torch.float32)
        return batch_contexts, batch_rewards
        
    def clear(self):
        """Clear the dataset - required by the Collector interface."""
        # This is a no-op for this adapter as we don't collect data incrementally
        pass

def load_jester_for_bandit(num_jokes=100, num_users=10000):
    """
    Helper function to load the Jester dataset for bandit algorithms.
    
    Returns:
        dataset: JesterBanditAdapter instance
        contexts: Tensor of joke features
        rewards: Tensor of user ratings
    """
    dataset = JesterBanditAdapter(num_jokes, num_users)
    return dataset, dataset.contexts, dataset.rewards

if __name__ == "__main__":
    # Test the adapter
    dataset, contexts, rewards = load_jester_for_bandit()
    print(f"Contexts shape: {contexts.shape}")
    print(f"Rewards shape: {rewards.shape}")
    
    # Test batch retrieval
    batch_contexts, batch_rewards = dataset.get_batch(16)
    print(f"Batch contexts shape: {batch_contexts.shape}")
    print(f"Batch rewards shape: {batch_rewards.shape}")
