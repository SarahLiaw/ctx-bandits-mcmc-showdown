"""
Adapter module to integrate financial datasets with the neural bandit framework.
This module supports Yahoo Finance data and other financial datasets.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FinancialBanditAdapter(Dataset):
    """
    Adapter class to use financial datasets with the neural bandit algorithms.
    This class converts financial data into the format expected by the bandit algorithms.
    Implements the interface required by the Collector class used in LMCTS.
    """
    def __init__(self, num_assets=10, lookback_days=30, feature_dim=20, data_source='yahoo'):
        super(FinancialBanditAdapter, self).__init__()
        
        self.num_assets = num_assets
        self.lookback_days = lookback_days
        self.feature_dim = feature_dim
        
        # Load or download the financial data
        self.contexts, self.rewards = self._load_financial_data(data_source)
        
        self.num_arms = self.num_assets
        self.dim_context = self.feature_dim
        self.num_samples = len(self.rewards)
        
        print(f"Financial dataset loaded with {self.num_arms} assets and {self.num_samples} time points")
        print(f"Context dimension: {self.dim_context}")
    
    def _load_financial_data(self, data_source):
        """Load financial data from the specified source."""
        data_dir = os.path.join("data", "financial")
        os.makedirs(data_dir, exist_ok=True)
        
        cache_file = os.path.join(data_dir, f"financial_data_{self.num_assets}_{self.lookback_days}.pt")
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            print("Loading cached financial data...")
            data = torch.load(cache_file)
            return data['contexts'], data['rewards']
        
        print(f"Downloading financial data from {data_source}...")
        
        if data_source == 'yahoo':
            # List of top assets to use
            assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG', 
                     'UNH', 'HD', 'BAC', 'MA', 'XOM', 'DIS', 'CSCO', 'VZ', 'ADBE', 'CRM'][:self.num_assets]
            
            # Download data
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=self.lookback_days * 10)  # Get more historical data
            
            print(f"Downloading data for {assets} from {start_date} to {end_date}")
            all_data = yf.download(assets, start=start_date, end=end_date)
            
            print(f"Downloaded data shape: {all_data.shape}")
            if len(all_data) == 0:
                raise ValueError("No data returned from Yahoo Finance API")
                
            print(f"Data columns: {all_data.columns}")
            if isinstance(all_data.columns, pd.MultiIndex):
                print(f"Column levels: {all_data.columns.levels}")
            
            # Extract features and rewards
            contexts = []
            rewards = []
            
            # For each day in our sample period
            # Stop one day before the end to ensure we can access next day's price for returns
            for i in range(self.lookback_days * 2, len(all_data) - 1):
                day_features = []
                day_returns = []
                
                for asset in assets:
                    try:
                        # Determine which price column to use (API structure can vary)
                        price_col = None
                        
                        # Handle MultiIndex columns (standard yfinance format)
                        if isinstance(all_data.columns, pd.MultiIndex):
                            avail_cols = list(all_data.columns.levels[0])
                            if 'Adj Close' in avail_cols:
                                price_col = 'Adj Close'
                            elif 'Close' in avail_cols:
                                price_col = 'Close'
                            
                            # Get the asset data if we found a valid column
                            if price_col:
                                if asset in all_data[price_col].columns:
                                    asset_data = all_data[price_col][asset].iloc[i-self.lookback_days*2:i].values
                                else:
                                    print(f"Asset {asset} not found in {price_col} data")
                                    continue
                            else:
                                print(f"No suitable price column found in {avail_cols}")
                                continue
                        # Handle flat columns (sometimes yfinance returns this format)
                        else:
                            # In flat format, try to find columns for this asset
                            asset_cols = [col for col in all_data.columns if asset in col]
                            if len(asset_cols) > 0:
                                # Prefer Adj Close or Close if available
                                for col in asset_cols:
                                    if 'Adj Close' in col or 'Close' in col:
                                        asset_data = all_data[col].iloc[i-self.lookback_days*2:i].values
                                        break
                                else:
                                    # If no preferred column, use the first available
                                    asset_data = all_data[asset_cols[0]].iloc[i-self.lookback_days*2:i].values
                            else:
                                print(f"No columns found for asset {asset}")
                                continue
                                
                    except Exception as e:
                        print(f"Error accessing data for {asset}: {e}")
                        print(f"Data structure: {type(all_data.columns)}")
                        if isinstance(all_data.columns, pd.MultiIndex):
                            print(f"Available column levels: {all_data.columns.levels}")
                            if len(all_data.columns.levels) > 1:
                                print(f"Available price columns: {all_data.columns.levels[0]}")
                                print(f"Available assets: {all_data.columns.levels[1]}")
                        else:
                            print(f"Available columns: {all_data.columns}")
                        continue  # Skip this asset instead of failing the entire process
                    
                    if len(asset_data) < self.lookback_days:
                        continue  # Skip if not enough data
                    
                    # Calculate features (returns, volatility, momentum, etc.)
                    returns = np.diff(asset_data) / asset_data[:-1]
                    volatility = np.std(returns[-self.lookback_days:])
                    momentum_1d = returns[-1]
                    momentum_5d = np.mean(returns[-5:])
                    momentum_20d = np.mean(returns[-20:])
                    
                    # More advanced features
                    rolling_max = np.max(asset_data[-self.lookback_days:])
                    rolling_min = np.min(asset_data[-self.lookback_days:])
                    price_range = (rolling_max - rolling_min) / rolling_min
                    
                    # Combine features
                    features = np.concatenate([
                        returns[-self.lookback_days:],
                        [volatility, momentum_1d, momentum_5d, momentum_20d, price_range]
                    ])
                    
                    # Next day return (reward)
                    # Use the same price column (Close or Adj Close) for consistency
                    price_col = 'Close' if 'Close' in all_data.columns.levels[0] and 'Adj Close' not in all_data.columns.levels[0] else 'Adj Close'
                    
                    next_return = (all_data[price_col][asset].iloc[i+1] - 
                                   all_data[price_col][asset].iloc[i]) / all_data[price_col][asset].iloc[i]
                    
                    day_features.append(features)
                    day_returns.append(next_return)
                
                # Debug info about data collection
                if len(day_features) > 0:
                    print(f"Day {i}: Found data for {len(day_features)}/{self.num_assets} assets")
                    
                # Accept days where we have at least 2 assets with data
                # This makes the adapter more robust to API changes and missing data
                if len(day_features) >= 2:
                    contexts.append(day_features)
                    rewards.append(day_returns)
            
            # Convert to numpy arrays
            contexts = np.array(contexts)
            rewards = np.array(rewards)
            
            # Check if we have any valid data
            if len(contexts) == 0:
                raise ValueError(f"No valid financial data could be collected. Check your assets and date range.")
                
            print(f"Collected data for {len(contexts)} trading days")
                
            # Reduce dimensionality if needed
            if len(contexts.shape) >= 3 and contexts.shape[2] > self.feature_dim:
                reshaped_contexts = contexts.reshape(-1, contexts.shape[2])
                scaler = StandardScaler()
                reshaped_contexts = scaler.fit_transform(reshaped_contexts)
                
                pca = PCA(n_components=self.feature_dim)
                reshaped_contexts = pca.fit_transform(reshaped_contexts)
                
                contexts = reshaped_contexts.reshape(contexts.shape[0], contexts.shape[1], self.feature_dim)
            
            # Normalize rewards to [0, 1] range for easier learning
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            rewards = (rewards - min_reward) / (max_reward - min_reward)
            
            # Convert to tensors
            contexts = torch.tensor(contexts, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            
            # Cache the data
            torch.save({'contexts': contexts, 'rewards': rewards}, cache_file)
            
            return contexts, rewards
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Return all contexts for all assets and the rewards for a specific day.
        This format matches what the neural bandit algorithms expect.
        """
        # Format required by LMCTS: each arm's context and the corresponding reward
        day_idx = idx // self.num_arms
        arm_idx = idx % self.num_arms
        if day_idx >= len(self.contexts):
            day_idx = day_idx % len(self.contexts)
        return self.contexts[day_idx][arm_idx], self.rewards[day_idx][arm_idx]
    
    def get_batch(self, batch_size=32):
        """Get a random batch of samples."""
        indices = torch.randint(0, self.num_samples, (batch_size,))
        batch_contexts = self.contexts[indices]
        batch_rewards = self.rewards[indices]
        return batch_contexts, batch_rewards
        
    def clear(self):
        """Clear the dataset - required by the Collector interface."""
        # This is a no-op for this adapter as we don't collect data incrementally
        pass

def load_financial_for_bandit(num_assets=10, lookback_days=30, feature_dim=20):
    """
    Helper function to load financial data for bandit algorithms.
    
    Returns:
        dataset: FinancialBanditAdapter instance
        contexts: Tensor of asset features
        rewards: Tensor of asset returns
    """
    dataset = FinancialBanditAdapter(num_assets, lookback_days, feature_dim)
    return dataset, dataset.contexts, dataset.rewards

if __name__ == "__main__":
    # Test the adapter
    dataset, contexts, rewards = load_financial_for_bandit()
    print(f"Contexts shape: {contexts.shape}")
    print(f"Rewards shape: {rewards.shape}")
    
    # Test batch retrieval
    batch_contexts, batch_rewards = dataset.get_batch(16)
    print(f"Batch contexts shape: {batch_contexts.shape}")
    print(f"Batch rewards shape: {batch_rewards.shape}")
