"""
Script to download and prepare the Jester dataset for use with the neural bandit algorithms.
The Jester dataset contains ratings of jokes from users.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import urllib.request
import zipfile
import io

def download_jester_dataset():
    """Download the Jester dataset."""
    print("Downloading Jester dataset...")
    
    # URLs for the Jester dataset
    url = "https://goldberg.berkeley.edu/jester-data/jester-data-1.zip"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Download and extract the dataset
    response = urllib.request.urlopen(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
    zip_file.extractall("data/jester")
    
    print("Dataset downloaded and extracted to data/jester/")

def prepare_jester_dataset(num_jokes=100, num_users=10000):
    """
    Prepare the Jester dataset for bandit algorithms.
    
    Args:
        num_jokes: Number of jokes to include
        num_users: Number of users to include
    
    Returns:
        contexts: Joke features
        rewards: User ratings (normalized to [0, 1])
    """
    # Check if the dataset exists, download if not
    if not os.path.exists("data/jester"):
        download_jester_dataset()
    
    # Load the dataset
    data_path = "data/jester/jester-data-1.xls"
    if not os.path.exists(data_path):
        data_path = "data/jester/jester-data-1.csv"  # Try alternative format
    
    try:
        df = pd.read_excel(data_path)
    except:
        # Try different encodings to handle the file
        for encoding in ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                print(f"Successfully loaded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                if encoding == 'cp1252':
                    raise
                print(f"Failed with encoding: {encoding}, trying next...")
    
    # Process the data
    # The first column is user info, the rest are joke ratings
    ratings = df.iloc[:num_users, 1:num_jokes+1].values
    
    # Replace 99 (missing values) with NaN
    ratings[ratings == 99] = np.nan
    
    # Fill missing values with mean of each joke
    joke_means = np.nanmean(ratings, axis=0)
    for j in range(ratings.shape[1]):
        mask = np.isnan(ratings[:, j])
        ratings[mask, j] = joke_means[j]
    
    # Normalize ratings to [0, 1]
    ratings = (ratings + 10) / 20.0  # Original scale is -10 to +10
    
    # Create joke features using PCA on the transpose of the ratings matrix
    # This gives us features for each joke based on user ratings
    scaler = StandardScaler()
    pca = PCA(n_components=min(32, num_jokes))
    
    # Transpose to get jokes as rows
    joke_data = ratings.T
    joke_data_scaled = scaler.fit_transform(joke_data)
    joke_features = pca.fit_transform(joke_data_scaled)
    
    # Save the processed data
    np.save("data/jester_contexts.npy", joke_features)
    np.save("data/jester_rewards.npy", ratings)
    
    print(f"Processed {num_jokes} jokes and {num_users} users")
    print(f"Joke features shape: {joke_features.shape}")
    print(f"User ratings shape: {ratings.shape}")
    
    return joke_features, ratings

class JesterDataset(torch.utils.data.Dataset):
    """Jester dataset for bandit algorithms."""
    def __init__(self, num_jokes=100, num_users=10000):
        super(JesterDataset, self).__init__()
        
        # Check if processed data exists
        if (not os.path.exists("data/jester_contexts.npy") or 
            not os.path.exists("data/jester_rewards.npy")):
            self.contexts, self.rewards = prepare_jester_dataset(num_jokes, num_users)
        else:
            self.contexts = np.load("data/jester_contexts.npy")
            self.rewards = np.load("data/jester_rewards.npy")
        
        self.num_jokes = self.contexts.shape[0]
        self.num_users = self.rewards.shape[0]
        self.dim_context = self.contexts.shape[1]
        
        # Convert to torch tensors
        self.contexts = torch.tensor(self.contexts, dtype=torch.float32)
    
    def __len__(self):
        return self.num_users
    
    def __getitem__(self, idx):
        # Return all joke contexts and the user's ratings
        return self.contexts, torch.tensor(self.rewards[idx], dtype=torch.float32)

if __name__ == "__main__":
    # Test the dataset preparation
    prepare_jester_dataset()
    dataset = JesterDataset()
    print(f"Dataset loaded with {dataset.num_jokes} jokes and {dataset.num_users} users")
    print(f"Context dimension: {dataset.dim_context}")
