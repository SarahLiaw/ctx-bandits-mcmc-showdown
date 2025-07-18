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

def prepare_jester_dataset_paper():
    """
    Prepare the Jester dataset to match the paper:
    - 19181 users who rated all 40 jokes
    - 8 arms (jokes), 32 context features (jokes)
    """
    # Download if needed
    if not os.path.exists("data/jester"):
        download_jester_dataset()
    data_path = "data/jester/jester-data-1.xls"
    if not os.path.exists(data_path):
        data_path = "data/jester/jester-data-1.csv"
    try:
        df = pd.read_excel(data_path)
    except:
        for encoding in ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                print(f"Successfully loaded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                if encoding == 'cp1252':
                    raise
                print(f"Failed with encoding: {encoding}, trying next...")

    # The first column is user info, the next 100 are joke ratings
    ratings = df.iloc[:, 1:101].values
    ratings[ratings == 99] = np.nan

    # Only keep users who rated all of the first 40 jokes
    mask = ~np.isnan(ratings[:, :40]).any(axis=1)
    filtered = ratings[mask]
    print(f"Users with all 40 jokes rated: {filtered.shape[0]}")
    filtered = filtered[:19181]  # match paper

    # Randomly select 8 arms and 32 context jokes from the first 40
    np.random.seed(42)
    perm = np.random.permutation(40)
    arm_idx = perm[:8]
    context_idx = perm[8:]

    # For each user, context is their ratings for the 32 context jokes
    contexts = filtered[:, context_idx]
    # For each user, rewards are their ratings for the 8 arms
    rewards = filtered[:, arm_idx]

    # Normalize ratings to [0, 1]
    contexts = (contexts + 10) / 20.0
    rewards = (rewards + 10) / 20.0

    np.save("data/jester_contexts.npy", contexts)
    np.save("data/jester_rewards.npy", rewards)
    print(f"Saved contexts shape: {contexts.shape}")
    print(f"Saved rewards shape: {rewards.shape}")
    print(f"Arms (joke indices): {arm_idx}")
    print(f"Context (joke indices): {context_idx}")

if __name__ == "__main__":
    prepare_jester_dataset_paper()
