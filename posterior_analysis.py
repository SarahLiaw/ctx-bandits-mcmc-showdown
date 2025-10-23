#!/usr/bin/env python3
"""
Posterior Distribution Analysis for Linear Bandits
Compares MCMC algorithm posteriors against true analytical posterior
Similar to Figure 1 in Thompson Sampling paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import argparse
import scipy.linalg
from scipy import stats

# Import agents
from src.MCMC import (
    LMCTS, FGLMCTS, MALATS, FGMALATS,
    SFGLMCTS, SFGMALATS,
    PLMCTS, PFGLMCTS, PSFGLMCTS,
    HMCTS, FGHMCTS, SFGHMCTS,
    PHMCTS, PFGHMCTS, PSFGHMCTS
)
from src.baseline import LinTS

# ============================================================================
# Configuration
# ============================================================================
K_ARMS = 6          # Number of arms
D_DIM = 20          # Context dimension  
LAMBDA_PRIOR = 1.0  # Prior precision: β ~ N(0, λ^-1 I)
SIGMA_REWARD = 0.5  # Reward noise std
T_HORIZON = 2000    # Time horizon
N_POSTERIOR_SAMPLES = 1500  # Number of posterior samples to visualize
ETA = 1.0           # Inverse temperature for MCMC algorithms
CORRELATED_CONTEXTS = True  # True: elliptical posteriors, False: circular posteriors

# Algorithm configurations (subset for analysis)
ALGORITHMS = [
    'LinTS',
    'LMCTS',
    'FGLMCTS',
    'MALATS',
    'PLMCTS',  # Preconditioned LMC (not HMC)
]

# ============================================================================
# Helper Functions
# ============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_synthetic_data(k_arms, d_dim, t_horizon, lambda_prior, sigma_reward, seed=0, correlated=True):
    """
    Generate synthetic linear bandit data.
    
    Args:
        correlated: If True, generate contexts with correlations (elliptical posteriors)
                   If False, use i.i.d. N(0,I) contexts (circular posteriors)
    
    Returns:
        contexts: (T, d) context vectors
        true_betas: (k, d) true parameter vectors for each arm
        optimal_actions: (T,) optimal action at each timestep
        optimal_rewards: (T,) optimal reward at each timestep
    """
    set_seed(seed)
    
    # Generate true parameters for each arm: β_i ~ N(0, λ^-1 I)
    true_betas = torch.randn(k_arms, d_dim) / np.sqrt(lambda_prior)
    
    if correlated:
        # Generate contexts with strong correlations in first few dimensions
        # This ensures β₁ vs β₂ plots show clear elliptical structure
        
        # Start with i.i.d. standard normal
        contexts = torch.randn(t_horizon, d_dim)
        
        # Add strong correlation between first 2 dimensions
        # Make β₁ have variance 3.0 and β₂ have variance 0.3
        # Plus correlation of 0.6 between them
        contexts[:, 0] = contexts[:, 0] * 1.7  # std = 1.7, var ≈ 3.0
        contexts[:, 1] = contexts[:, 1] * 0.55 + 0.6 * contexts[:, 0]  # correlated with dim 0
        
        # Add moderate correlations to other dimensions
        for i in range(2, min(d_dim, 10)):
            scale = 1.0 + 0.5 * (i - 2) / 8  # Gradually varying scales
            contexts[:, i] = contexts[:, i] * scale
    else:
        # Generate contexts: X_t ~ N(0, I) - original behavior
        contexts = torch.randn(t_horizon, d_dim)
    
    # Compute optimal actions and rewards
    expected_rewards = contexts @ true_betas.T  # (T, k)
    optimal_actions = expected_rewards.argmax(dim=1)
    optimal_rewards = expected_rewards.max(dim=1).values
    
    return contexts, true_betas, optimal_actions, optimal_rewards

def phi_linear(x, nb_arms):
    """Feature map for linear bandits: block diagonal."""
    return torch.block_diag(*[x]*nb_arms)

def phi_a_linear(x, a, nb_arms):
    """Feature map for single arm."""
    return torch.block_diag(*[x]*nb_arms)[a, :]

def compute_true_posterior(played_features, rewards, lambda_prior, sigma_reward, eta=1.0):
    """
    Compute the true analytical posterior for linear Bayesian regression.
    
    Prior: β ~ N(0, λ^-1 I)
    Likelihood: r_i = x_i^T β + ε, ε ~ N(0, σ²)
    Posterior: β | D ~ N(μ_post, Σ_post)
    
    With precision parametrization:
        Σ_post^-1 = λI + (1/σ²) Σ x_i x_i^T
        μ_post = Σ_post (1/σ²) Σ x_i r_i
    
    Args:
        played_features: (n, d) feature vectors of played arms
        rewards: (n,) observed rewards
        lambda_prior: prior precision
        sigma_reward: reward noise std
        eta: inverse temperature (scales data term)
        
    Returns:
        mu_post: (d,) posterior mean
        Sigma_post: (d, d) posterior covariance
    """
    if played_features.shape[0] == 0:
        # No data: return prior
        d = played_features.shape[1] if played_features.ndim == 2 else len(lambda_prior)
        mu_post = torch.zeros(d)
        Sigma_post = torch.eye(d) / lambda_prior
        return mu_post, Sigma_post
    
    d = played_features.shape[1]
    X = played_features  # (n, d)
    r = rewards.reshape(-1, 1)  # (n, 1)
    
    # Posterior precision: Λ_post = λI + η/σ² X^T X
    precision_post = lambda_prior * torch.eye(d) + (eta / (sigma_reward ** 2)) * (X.T @ X)
    
    # Posterior covariance
    Sigma_post = torch.linalg.inv(precision_post)
    
    # Posterior mean: μ_post = Σ_post * η/σ² * X^T r
    mu_post = Sigma_post @ ((eta / (sigma_reward ** 2)) * X.T @ r)
    mu_post = mu_post.squeeze()
    
    return mu_post, Sigma_post

def sample_from_posterior(mu, Sigma, n_samples):
    """Sample from multivariate normal posterior."""
    d = len(mu)
    dist = torch.distributions.MultivariateNormal(mu, Sigma)
    samples = dist.sample((n_samples,))  # (n_samples, d)
    return samples

def extract_mcmc_samples(agent, n_samples, arm_idx=None):
    """
    Extract posterior samples from MCMC agent.
    
    For MCMC algorithms, we run the sampling procedure multiple times
    to collect samples.
    """
    samples = []
    
    # Save the current state
    original_updated = agent.is_posterior_updated if hasattr(agent, 'is_posterior_updated') else agent.updated
    
    for _ in range(n_samples):
        # Reset the update flag to trigger full sampling
        if hasattr(agent, 'is_posterior_updated'):
            agent.is_posterior_updated = True
        elif hasattr(agent, 'updated'):
            agent.updated = True
            
        # Sample from posterior
        theta_sample = agent.sample_posterior(arm_idx if arm_idx is not None else 0)
        samples.append(theta_sample.detach().squeeze().clone())
    
    # Restore state
    if hasattr(agent, 'is_posterior_updated'):
        agent.is_posterior_updated = original_updated
    else:
        agent.updated = original_updated
    
    samples = torch.stack(samples)  # (n_samples, d)
    return samples

def run_algorithm_with_data(algo_name, contexts, true_betas, sigma_reward, seed=0):
    """
    Run a single algorithm on fixed data and return the agent at final timestep.
    
    Returns:
        agent: trained agent
        played_features_per_arm: list of (n_i, d) tensors for each arm
        rewards_per_arm: list of (n_i,) tensors for each arm
    """
    set_seed(seed)
    
    T = contexts.shape[0]
    k_arms = true_betas.shape[0]
    d_dim = true_betas.shape[1]
    
    # Create info dict
    info = {
        'd': d_dim * k_arms,  # Full dimensionality for linear bandit
        'nb_arms': k_arms,
        'context_size': d_dim,
        'T': T,
        'std_prior': 1.0 / np.sqrt(LAMBDA_PRIOR),
        'eta': ETA,
        'std_reward': sigma_reward,
        'phi': phi_linear,
        'phi_a': phi_a_linear,
        'step_size': 0.01,
        'K': 100,
        'K_not_updated': 10,
        'lambda': 0.1,  # For feel-good variants
        'b': 10.0,  # For feel-good variants
        'L_leap': 10,  # For HMC
        'lambda_reg': 1.0,  # For preconditioned variants
        'accept_reject_step': 10,  # For MALA
    }
    
    # Initialize agent
    if algo_name == 'LinTS':
        agent = LinTS(info)
    elif algo_name == 'LMCTS':
        agent = LMCTS(info)
    elif algo_name == 'FGLMCTS':
        agent = FGLMCTS(info)
    elif algo_name == 'MALATS':
        agent = MALATS(info)
    elif algo_name == 'FGMALATS':
        agent = FGMALATS(info)
    elif algo_name == 'SFGLMCTS':
        agent = SFGLMCTS(info)
    elif algo_name == 'SFGMALATS':
        agent = SFGMALATS(info)
    elif algo_name == 'PLMCTS':
        agent = PLMCTS(info)
    elif algo_name == 'PFGLMCTS':
        agent = PFGLMCTS(info)
    elif algo_name == 'PSFGLMCTS':
        agent = PSFGLMCTS(info)
    elif algo_name == 'HMCTS':
        agent = HMCTS(info)
    elif algo_name == 'FGHMCTS':
        agent = FGHMCTS(info)
    elif algo_name == 'SFGHMCTS':
        agent = SFGHMCTS(info)
    elif algo_name == 'PHMCTS':
        agent = PHMCTS(info)
    elif algo_name == 'PFGHMCTS':
        agent = PFGHMCTS(info)
    elif algo_name == 'PSFGHMCTS':
        agent = PSFGHMCTS(info)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Track played features per arm
    played_features_per_arm = [[] for _ in range(k_arms)]
    rewards_per_arm = [[] for _ in range(k_arms)]
    
    arm_idx = torch.arange(k_arms)
    
    # Run algorithm
    for t in range(T):
        context = contexts[t]
        
        # Agent chooses arm
        action = agent.choose_arm(context, arm_idx)
        if isinstance(action, torch.Tensor):
            action = action.item()
        
        # Generate reward: r = X^T β_a + ε
        true_reward = context @ true_betas[action]
        noise = torch.randn(1) * sigma_reward
        reward = (true_reward + noise).item()
        
        # Store played data
        feature = phi_a_linear(context, action, k_arms)
        played_features_per_arm[action].append(feature)
        rewards_per_arm[action].append(reward)
        
        # Update agent
        agent.update(action, reward, context, arm_idx)
    
    # Convert to tensors
    for i in range(k_arms):
        if len(played_features_per_arm[i]) > 0:
            played_features_per_arm[i] = torch.stack(played_features_per_arm[i])
            rewards_per_arm[i] = torch.tensor(rewards_per_arm[i])
        else:
            played_features_per_arm[i] = torch.empty(0, d_dim * k_arms)
            rewards_per_arm[i] = torch.empty(0)
    
    return agent, played_features_per_arm, rewards_per_arm

def visualize_posteriors(true_samples_per_arm, algo_samples_per_arm, algo_name, out_dir):
    """
    Create Figure 1 style visualization: 1x6 grid of scatter plots.
    
    Each subplot shows first two dimensions of posterior for one arm.
    Green: true analytical posterior
    Red: algorithm's posterior
    """
    fig, axes = plt.subplots(1, K_ARMS, figsize=(18, 3))
    
    for arm_idx in range(K_ARMS):
        ax = axes[arm_idx]
        
        true_samples = true_samples_per_arm[arm_idx]  # (n_samples, d)
        algo_samples = algo_samples_per_arm[arm_idx]  # (n_samples, d)
        
        if true_samples.shape[0] > 0:
            # Plot first two dimensions
            ax.scatter(true_samples[:, 0], true_samples[:, 1], 
                      c='green', alpha=0.3, s=1, label='True Posterior')
        
        if algo_samples.shape[0] > 0:
            ax.scatter(algo_samples[:, 0], algo_samples[:, 1],
                      c='red', alpha=0.3, s=1, label=algo_name)
        
        ax.set_title(f'Arm {arm_idx + 1}')
        ax.set_xlabel(r'$\beta_1$')
        if arm_idx == 0:
            ax.set_ylabel(r'$\beta_2$')
        ax.grid(True, alpha=0.3)
        if arm_idx == K_ARMS - 1:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    fig_path = out_dir / f'{algo_name}_posterior_comparison.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {fig_path}")
    return fig_path

def compute_wasserstein_distance(samples1, samples2):
    """
    Compute 2-Wasserstein distance between two sample sets (first 2 dims only).
    Using scipy's implementation for 2D case.
    """
    if samples1.shape[0] == 0 or samples2.shape[0] == 0:
        return float('nan')
    
    # Use only first 2 dimensions for visualization consistency
    s1 = samples1[:, :2].numpy()
    s2 = samples2[:, :2].numpy()
    
    # Compute pairwise distances and approximate Wasserstein
    # For simplicity, use mean and covariance comparison
    mu1, mu2 = s1.mean(axis=0), s2.mean(axis=0)
    cov1, cov2 = np.cov(s1.T), np.cov(s2.T)
    
    # 2-Wasserstein for Gaussians (approximate)
    mean_dist = np.linalg.norm(mu1 - mu2) ** 2
    cov_dist = np.trace(cov1 + cov2 - 2 * np.real(scipy.linalg.sqrtm(cov1 @ cov2)))
    
    return np.sqrt(mean_dist + cov_dist)

# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--algorithms', nargs='+', default=ALGORITHMS,
                       help='Algorithms to analyze')
    args = parser.parse_args()
    
    # Create output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"posterior_analysis_{ts}")
    out_dir.mkdir(exist_ok=True)
    print(f"Output directory: {out_dir}")
    
    # Generate synthetic data
    print("\n=== Generating Synthetic Data ===")
    contexts, true_betas, optimal_actions, optimal_rewards = generate_synthetic_data(
        K_ARMS, D_DIM, T_HORIZON, LAMBDA_PRIOR, SIGMA_REWARD, seed=args.seed, correlated=CORRELATED_CONTEXTS
    )
    context_type = "correlated (elliptical posteriors)" if CORRELATED_CONTEXTS else "i.i.d. (circular posteriors)"
    print(f"Generated {T_HORIZON} timesteps, {K_ARMS} arms, {D_DIM} dimensions")
    print(f"Context distribution: {context_type}")
    
    # Save data
    torch.save({
        'contexts': contexts,
        'true_betas': true_betas,
        'optimal_actions': optimal_actions,
        'optimal_rewards': optimal_rewards,
        'config': {
            'k_arms': K_ARMS,
            'd_dim': D_DIM,
            't_horizon': T_HORIZON,
            'lambda_prior': LAMBDA_PRIOR,
            'sigma_reward': SIGMA_REWARD,
            'seed': args.seed,
        }
    }, out_dir / 'synthetic_data.pt')
    
    # Results storage
    results = {}
    
    # Analyze each algorithm
    for algo_name in args.algorithms:
        print(f"\n=== Analyzing {algo_name} ===")
        
        # Run algorithm
        print(f"  Running algorithm...")
        agent, played_features_per_arm, rewards_per_arm = run_algorithm_with_data(
            algo_name, contexts, true_betas, SIGMA_REWARD, seed=args.seed
        )
        
        # Compute true posteriors for each arm
        print(f"  Computing true posteriors...")
        true_samples_per_arm = []
        algo_samples_per_arm = []
        wasserstein_distances = []
        
        for arm_idx in range(K_ARMS):
            played_features = played_features_per_arm[arm_idx]
            rewards = rewards_per_arm[arm_idx]
            
            n_plays = played_features.shape[0]
            print(f"    Arm {arm_idx}: {n_plays} plays")
            
            if n_plays > 0:
                # Extract features for this arm (block diagonal structure)
                # For arm i, features are at indices [i*D_DIM : (i+1)*D_DIM]
                arm_features = played_features[:, arm_idx*D_DIM:(arm_idx+1)*D_DIM]
                
                # Compute true posterior
                mu_post, Sigma_post = compute_true_posterior(
                    arm_features, rewards, LAMBDA_PRIOR, SIGMA_REWARD, eta=ETA
                )
                
                # Sample from true posterior
                true_samples = sample_from_posterior(mu_post, Sigma_post, N_POSTERIOR_SAMPLES)
                true_samples_per_arm.append(true_samples)
                
                # Extract samples from algorithm posterior
                # For MCMC algorithms, we sample from their internal state
                # For LinTS, we sample from their analytical posterior
                if algo_name == 'LinTS':
                    # LinTS maintains Vt_inv and bt
                    # We need to extract arm-specific posterior
                    # The full posterior is over concatenated parameters
                    # Extract arm block
                    arm_start = arm_idx * D_DIM
                    arm_end = (arm_idx + 1) * D_DIM
                    
                    # Get full posterior
                    full_mu = (agent.Vt_inv @ agent.bt).squeeze()
                    full_Sigma = (1 / agent.info['eta']) * agent.Vt_inv
                    
                    # Extract arm block
                    arm_mu = full_mu[arm_start:arm_end]
                    arm_Sigma = full_Sigma[arm_start:arm_end, arm_start:arm_end]
                    
                    algo_samples = sample_from_posterior(arm_mu, arm_Sigma, N_POSTERIOR_SAMPLES)
                else:
                    # For MCMC algorithms, run sampling procedure
                    # Note: This samples the full concatenated parameter
                    # We need to extract the arm-specific block
                    full_samples = extract_mcmc_samples(agent, N_POSTERIOR_SAMPLES, arm_idx=arm_idx)
                    arm_start = arm_idx * D_DIM
                    arm_end = (arm_idx + 1) * D_DIM
                    algo_samples = full_samples[:, arm_start:arm_end]
                
                algo_samples_per_arm.append(algo_samples)
                
                # Compute Wasserstein distance
                w_dist = compute_wasserstein_distance(true_samples, algo_samples)
                wasserstein_distances.append(w_dist)
            else:
                true_samples_per_arm.append(torch.empty(0, D_DIM))
                algo_samples_per_arm.append(torch.empty(0, D_DIM))
                wasserstein_distances.append(float('nan'))
        
        # Visualize
        print(f"  Creating visualization...")
        fig_path = visualize_posteriors(
            true_samples_per_arm, algo_samples_per_arm, algo_name, out_dir
        )
        
        # Store results
        results[algo_name] = {
            'wasserstein_distances': wasserstein_distances,
            'mean_wasserstein': np.nanmean(wasserstein_distances),
            'num_plays_per_arm': [pf.shape[0] for pf in played_features_per_arm],
        }
        
        print(f"  Mean Wasserstein distance: {results[algo_name]['mean_wasserstein']:.4f}")
        print(f"  Plays per arm: {results[algo_name]['num_plays_per_arm']}")
    
    # Save results
    results_path = out_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, float)) else x)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {out_dir}")
    print(f"\nSummary:")
    for algo_name, res in results.items():
        print(f"  {algo_name}: Mean W-distance = {res['mean_wasserstein']:.4f}")

if __name__ == '__main__':
    main()
