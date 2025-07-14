import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas

import torch
from torch.utils.data import Dataset, DataLoader

# Import adapters for financial and jester datasets
try:
    from .financial_adapter import load_financial_for_bandit
    from .jester_adapter import load_jester_for_bandit
except ImportError:
    # If relative import fails, try absolute import
    try:
        from train_utils.financial_adapter import load_financial_for_bandit
        from train_utils.jester_adapter import load_jester_for_bandit
    except ImportError:
        # If both fail, we'll handle it in the loaddata method
        load_financial_for_bandit = None
        load_jester_for_bandit = None


continuous_dataset = ['shuttle', 'covertype']


def sample_data(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def remove_nan(arr):
    '''
    Drop the rows that contain Nan
    '''
    df = pd.DataFrame(arr)
    df = df.dropna()
    return df.to_numpy()


class SimData(Dataset):
    def __init__(self, datapath, num_data=None, index=0):
        data = torch.load(datapath)
        self.context = data['context']
        if num_data:
            self.context = self.context[index:index + num_data]
        # self.context = self.context / torch.norm(context, dim=1, keepdim=True)

    def __getitem__(self, idx):
        return self.context[idx]

    def __len__(self):
        return self.context.shape[0]


class UCI(Dataset):
    def __init__(self, datapath, dim_context, num_data=None, num_arms=2):
        super(UCI, self).__init__()
        self.dim_context = dim_context
        self.num_arms = num_arms
        self.loaddata(datapath, dim_context, num_data)

    def __getitem__(self, idx):
        x = self.context[idx]
        cxt = torch.zeros((self.num_arms, self.dim_context * self.num_arms))
        for i in range(self.num_arms):
            cxt[i, i * self.dim_context: (i + 1) * self.dim_context] = x
        return cxt, self.label[idx]

    def __len__(self):
        return self.label.shape[0]

    def loaddata(self, datapath, dim_context, num_data=None):
        data = np.loadtxt(datapath)
        self.label = (data[:, -1] - 1).astype(int)
        # data preprocessing
        context = data[:, 0:dim_context].astype(np.float32)
        if num_data:
            context = context[0:num_data]
            self.label = self.label[0:num_data]
        # context = context - context.mean(axis=0, keepdims=True)
        self.context = context / np.linalg.norm(context, axis=1, keepdims=True)
        self.context = torch.tensor(self.context)


class AutoUCI(Dataset):
    def __init__(self, name, dim_context, num_arms, num_data=None, version='active'):
        super(AutoUCI, self).__init__()
        self.dim_context = dim_context
        self.num_arms = num_arms
        self.loaddata(name, version, num_data)

    def __getitem__(self, idx):
        # For financial and jester datasets, context is already in the right format
        # (num_samples, num_arms, dim_context * num_arms)
        if len(self.context.shape) == 3:
            return self.context[idx], self.label[idx]
        else:
            # For UCI datasets, reshape the context
            x = self.context[idx]
            cxt = torch.zeros((self.num_arms, self.dim_context * self.num_arms))
            for i in range(self.num_arms):
                cxt[i, i * self.dim_context: (i + 1) * self.dim_context] = x
            return cxt, self.label[idx]

    def __len__(self):
        return self.label.shape[0]

    def loaddata(self, name, version, num_data):
        # Handle financial dataset
        if name == 'financial':
            if load_financial_for_bandit is None:
                raise ImportError("Financial adapter not available. Please ensure financial_adapter.py is in the train_utils directory.")
            
            # Load financial data using the adapter
            dataset, contexts, rewards = load_financial_for_bandit(
                num_assets=self.num_arms,
                lookback_days=30,
                feature_dim=self.dim_context
            )
            
            # Convert to the format expected by AutoUCI
            # For financial data, contexts are (num_days, num_assets, feature_dim)
            # We need to create a dataset where each sample represents a day with all asset contexts
            if len(contexts.shape) == 3:  # (num_days, num_assets, feature_dim)
                num_days = contexts.shape[0]
                # Create context tensor: (num_days, num_arms, dim_context * num_arms)
                self.context = torch.zeros(num_days, self.num_arms, self.dim_context * self.num_arms)
                for day in range(num_days):
                    for arm in range(self.num_arms):
                        if arm < contexts.shape[1]:  # Make sure we don't exceed available assets
                            self.context[day, arm, arm * self.dim_context:(arm + 1) * self.dim_context] = contexts[day, arm]
                
                # Create labels (optimal arm for each day based on rewards)
                self.label = torch.argmax(rewards, dim=1)  # Shape: (num_days,)
            else:
                # Fallback for different data format
                self.context = contexts
                self.label = torch.zeros(len(contexts), dtype=torch.long)
            
            # Limit data if specified
            if num_data and len(self.context) > num_data:
                self.context = self.context[:num_data]
                self.label = self.label[:num_data]
            
            return
        
        # Handle jester dataset
        if name == 'jester':
            if load_jester_for_bandit is None:
                raise ImportError("Jester adapter not available. Please ensure jester_adapter.py is in the train_utils directory.")
            
            # Load jester data using the adapter
            dataset, contexts, rewards = load_jester_for_bandit(
                num_jokes=self.num_arms,
                num_users=num_data if num_data else 10000
            )
            
            # Convert to the format expected by AutoUCI
            # For jester data, contexts are (num_users, num_jokes, feature_dim)
            # We need to create a dataset where each sample represents a user with all joke contexts
            if len(contexts.shape) == 3:  # (num_users, num_jokes, feature_dim)
                num_users = contexts.shape[0]
                # Create context tensor: (num_users, num_arms, dim_context * num_arms)
                self.context = torch.zeros(num_users, self.num_arms, self.dim_context * self.num_arms)
                for user in range(num_users):
                    for arm in range(self.num_arms):
                        if arm < contexts.shape[1]:  # Make sure we don't exceed available jokes
                            self.context[user, arm, arm * self.dim_context:(arm + 1) * self.dim_context] = contexts[user, arm]
                
                # Create labels (optimal joke for each user based on rewards)
                self.label = torch.argmax(rewards, dim=1)  # Shape: (num_users,)
            else:
                # Fallback for different data format
                self.context = contexts
                self.label = torch.zeros(len(contexts), dtype=torch.long)
            
            # Limit data if specified
            if num_data and len(self.context) > num_data:
                self.context = self.context[:num_data]
                self.label = self.label[:num_data]
            
            return
        
        # Handle UCI datasets (existing logic)
        if name == 'adult':
            cxt_df, label_ser = fetch_openml(
                name=name,
                version=version,
                data_home='data',
                return_X_y=True
            )
            cxt_df = cxt_df.replace('?', pd.NA).dropna()
            label_ser = label_ser[cxt_df.index]

            feat_enc = OrdinalEncoder(dtype=np.float32)
            context = feat_enc.fit_transform(cxt_df)

            lab_enc = OrdinalEncoder(dtype=int)
            label = lab_enc.fit_transform(
                label_ser.to_numpy().reshape(-1, 1)
            ).ravel().astype(int)

            if num_data:
                context = context[:num_data]
                label = label[:num_data]

            context = context / np.linalg.norm(context, axis=1, keepdims=True)
            self.context = torch.from_numpy(context)
            self.label = label
            return
        
        # Handle other UCI datasets
        cxt_df, label_ser = fetch_openml(
            name=name,
            version=version,
            data_home='data',
            as_frame=True,
            return_X_y=True
        )
        cxt_df = cxt_df.replace('?', pd.NA).dropna()
        label_ser = label_ser[cxt_df.index]

        if any(cxt_df.dtypes.apply(lambda dt: dt.kind not in "fi")):
            feat_enc = OrdinalEncoder(
                dtype=np.float32,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            context = feat_enc.fit_transform(cxt_df)
        else:
            context = cxt_df.to_numpy(dtype=np.float32)

        lab_enc = OrdinalEncoder(dtype=int)
        label = lab_enc.fit_transform(
            label_ser.to_numpy().reshape(-1, 1)
        ).ravel().astype(int)

        if num_data:
            context = context[:num_data]
            label = label[:num_data]

        context = context / np.linalg.norm(context, axis=1, keepdims=True)
        self.context = torch.from_numpy(context)
        self.label = label

    # def loaddata(self, name, version, num_data):
    #     cxt, label = fetch_openml(name=name, version=version, data_home='data', return_X_y=True)

    #     context = np.array(cxt).astype(np.float32)
    #     if num_data:
    #         label = label[0:num_data]
    #         context = context[0:num_data, :]
    #     # encode label
    #     if name not in continuous_dataset:
    #         encoder = OrdinalEncoder(dtype=int)
    #         label = encoder.fit_transform(label.reshape((-1, 1)))

    #         # Drop rows that contain Nan
    #         raw = np.concatenate([context, label], axis=1)
    #         raw = remove_nan(raw)
    #         self.label = raw[:, -1]
    #         context = raw[:, :-1]
    #     else:
    #         self.label = np.array(label).astype(int) - 1
    #     self.context = context / np.linalg.norm(context, axis=1, keepdims=True)
    #     self.context = torch.tensor(self.context)


class Collector(Dataset):
    '''
    Collect the context vectors that have appeared 
    '''

    def __init__(self):
        super(Collector, self).__init__()
        self.context = []
        self.rewards = []
        self.chosen_arms = []

    def __getitem__(self, key):
        return self.context[key], self.rewards[key]

    def __len__(self):
        return len(self.rewards)

    def collect_data(self, context, arm, reward):
        self.context.append(context.cpu())
        self.chosen_arms.append(arm)
        self.rewards.append(reward)

    def fetch_batch(self, batch_size=None):
        if batch_size is None or batch_size > len(self.rewards):
            return self.context, self.rewards
        else:
            offset = np.random.randint(0, len(self.rewards) - batch_size)
            return self.context[offset:offset + batch_size], self.rewards[offset: offset + batch_size]

    def clear(self):
        self.context = []
        self.rewards = []
        self.chosen_arms = []
