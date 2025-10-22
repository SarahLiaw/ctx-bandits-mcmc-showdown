
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class RestaurantBanditAdapter(Dataset):
    def __init__(self, data_dir='data/restaurant', feature_dim=64):
        super(RestaurantBanditAdapter, self).__init__()
        self.data_dir = data_dir
        self.feature_dim = 64  # Fixed to 64 for each feature type
        os.makedirs(data_dir, exist_ok=True)
        
        # Load real UCI data with correct encoding
        print('Loading REAL UCI Restaurant & Consumer Data...')
        raw_data_dir = 'data/restaurant_raw'
        
        # Load CSV files with ISO-8859-1 encoding
        geoplaces = pd.read_csv(os.path.join(raw_data_dir, 'geoplaces2.csv'), encoding='iso-8859-1')
        userprofile = pd.read_csv(os.path.join(raw_data_dir, 'userprofile.csv'), encoding='iso-8859-1')
        rating_final = pd.read_csv(os.path.join(raw_data_dir, 'rating_final.csv'), encoding='iso-8859-1')
        
        print(f'Loaded: {len(geoplaces)} restaurants, {len(userprofile)} users, {len(rating_final)} ratings')
        
        # Process data
        self.restaurant_ids = [str(rid) for rid in geoplaces['placeID'].unique()]
        self.user_ids = [str(uid) for uid in userprofile['userID'].unique()]
        
        # Create features
        restaurant_features = self._create_restaurant_features(geoplaces)
        user_features = self._create_user_features(userprofile)
        
        # Create interaction matrix
        interaction_matrix = self._create_interaction_matrix(rating_final)
        
        # Create contexts and rewards
        contexts, rewards = self._create_bandit_data(restaurant_features, user_features, interaction_matrix)
        
        self.contexts = torch.tensor(contexts, dtype=torch.float32)
        self.rewards = torch.tensor(rewards, dtype=torch.float32)
        
        self.num_arms = len(self.restaurant_ids)
        self.dim_context = self.contexts.shape[2]  # Actual context dimension (128)
        self.num_samples = len(self.rewards)
        
        print(f'Dataset: {self.num_arms} restaurants, {self.num_samples} user interactions, {self.dim_context}D contexts')
    
    def _create_restaurant_features(self, geoplaces):
        features = []
        for _, restaurant in geoplaces.iterrows():
            feature_vector = [
                restaurant['latitude'],
                restaurant['longitude'],
                hash(str(restaurant['alcohol'])) % 10,
                hash(str(restaurant['smoking_area'])) % 10,
                hash(str(restaurant['dress_code'])) % 10,
                hash(str(restaurant['accessibility'])) % 10,
                hash(str(restaurant['price'])) % 10,
                hash(str(restaurant['Rambience'])) % 10,
                hash(str(restaurant['franchise'])) % 10,
                hash(str(restaurant['area'])) % 10,
                hash(str(restaurant['other_services'])) % 10
            ]
            
            # Pad to 64
            if len(feature_vector) < 64:
                feature_vector.extend([0] * (64 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:64]
            
            features.append(feature_vector)
        return np.array(features)
    
    def _create_user_features(self, userprofile):
        features = []
        for _, user in userprofile.iterrows():
            feature_vector = [
                user['latitude'],
                user['longitude'],
                hash(str(user['smoker'])) % 10,
                hash(str(user['drink_level'])) % 10,
                hash(str(user['dress_preference'])) % 10,
                hash(str(user['ambience'])) % 10,
                hash(str(user['transport'])) % 10,
                hash(str(user['marital_status'])) % 10,
                hash(str(user['hijos'])) % 10,
                hash(str(user['interest'])) % 10,
                hash(str(user['personality'])) % 10,
                hash(str(user['religion'])) % 10,
                hash(str(user['activity'])) % 10,
                hash(str(user['color'])) % 10,
                hash(str(user['budget'])) % 10
            ]
            
            # Pad to 64
            if len(feature_vector) < 64:
                feature_vector.extend([0] * (64 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:64]
            
            features.append(feature_vector)
        return np.array(features)
    
    def _create_interaction_matrix(self, rating_final):
        interaction_matrix = np.zeros((len(self.user_ids), len(self.restaurant_ids)))
        
        user_id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        restaurant_id_to_idx = {rid: i for i, rid in enumerate(self.restaurant_ids)}
        
        for _, rating in rating_final.iterrows():
            user_idx = user_id_to_idx.get(str(rating['userID']))
            restaurant_idx = restaurant_id_to_idx.get(str(rating['placeID']))
            
            if user_idx is not None and restaurant_idx is not None:
                # Convert rating (0,1,2) to binary (0,1) for bandit rewards
                interaction_matrix[user_idx, restaurant_idx] = 1 if rating['rating'] >= 1 else 0
        
        return interaction_matrix
    
    def _create_bandit_data(self, restaurant_features, user_features, interaction_matrix):
        contexts = []
        rewards = []
        
        for user_idx in range(len(self.user_ids)):
            user_context = user_features[user_idx]
            user_rewards = interaction_matrix[user_idx]
            
            user_contexts = []
            for restaurant_idx in range(len(self.restaurant_ids)):
                combined_context = np.concatenate([user_context, restaurant_features[restaurant_idx]])
                user_contexts.append(combined_context)
            
            contexts.append(user_contexts)
            rewards.append(user_rewards)
        
        return np.array(contexts), np.array(rewards)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.contexts[idx], self.rewards[idx]

def load_restaurant_for_bandit(feature_dim=64, min_ratings_per_user=3, min_ratings_per_restaurant=5):
    dataset = RestaurantBanditAdapter(feature_dim=feature_dim)
    return dataset, dataset.contexts, dataset.rewards
