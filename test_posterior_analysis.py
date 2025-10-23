#!/usr/bin/env python3
"""
Unit tests for posterior_analysis.py

Tests core functionality including:
- Data generation with and without correlations
- True posterior computation
- Posterior sampling
- Block diagonal feature extraction
- Wasserstein distance computation
"""

import unittest
import torch
import numpy as np
from posterior_analysis import (
    generate_synthetic_data,
    compute_true_posterior,
    sample_from_posterior,
    phi_linear,
    phi_a_linear,
    compute_wasserstein_distance,
    set_seed
)


class TestDataGeneration(unittest.TestCase):
    """Test synthetic data generation."""
    
    def test_uncorrelated_contexts(self):
        """Test i.i.d. N(0,I) context generation."""
        k_arms, d_dim, t_horizon = 6, 20, 1000
        contexts, true_betas, opt_actions, opt_rewards = generate_synthetic_data(
            k_arms, d_dim, t_horizon, lambda_prior=1.0, sigma_reward=0.5,
            seed=42, correlated=False
        )
        
        # Check shapes
        self.assertEqual(contexts.shape, (t_horizon, d_dim))
        self.assertEqual(true_betas.shape, (k_arms, d_dim))
        self.assertEqual(opt_actions.shape, (t_horizon,))
        self.assertEqual(opt_rewards.shape, (t_horizon,))
        
        # Check contexts are approximately N(0,1) for uncorrelated case
        mean = contexts.mean(dim=0)
        std = contexts.std(dim=0)
        self.assertTrue(torch.allclose(mean, torch.zeros(d_dim), atol=0.1))
        # First few dims should be close to 1.0 (uncorrelated)
        self.assertTrue(torch.allclose(std[5:], torch.ones(d_dim-5), atol=0.2))
    
    def test_correlated_contexts(self):
        """Test correlated context generation."""
        k_arms, d_dim, t_horizon = 6, 20, 2000
        contexts, true_betas, opt_actions, opt_rewards = generate_synthetic_data(
            k_arms, d_dim, t_horizon, lambda_prior=1.0, sigma_reward=0.5,
            seed=42, correlated=True
        )
        
        # Check shapes
        self.assertEqual(contexts.shape, (t_horizon, d_dim))
        
        # Check first dimension has higher variance
        var_dim0 = contexts[:, 0].var().item()
        var_dim5 = contexts[:, 5].var().item()
        self.assertGreater(var_dim0, 2.0, "First dimension should have variance ~3.0")
        self.assertLess(var_dim5, 2.0, "Later dimensions should have variance ~1.0")
        
        # Check correlation between first two dimensions
        corr_matrix = torch.corrcoef(contexts[:, :2].T)
        corr_01 = corr_matrix[0, 1].item()
        self.assertGreater(abs(corr_01), 0.4, "Dimensions 0 and 1 should be correlated")
    
    def test_reproducibility(self):
        """Test that same seed produces same data."""
        k_arms, d_dim, t_horizon = 6, 20, 100
        
        contexts1, betas1, _, _ = generate_synthetic_data(
            k_arms, d_dim, t_horizon, 1.0, 0.5, seed=123
        )
        contexts2, betas2, _, _ = generate_synthetic_data(
            k_arms, d_dim, t_horizon, 1.0, 0.5, seed=123
        )
        
        self.assertTrue(torch.allclose(contexts1, contexts2))
        self.assertTrue(torch.allclose(betas1, betas2))


class TestPosteriorComputation(unittest.TestCase):
    """Test analytical posterior computation."""
    
    def test_prior_with_no_data(self):
        """Test that posterior equals prior with no observations."""
        d = 20
        lambda_prior = 1.0
        sigma_reward = 0.5
        
        # Empty data
        played_features = torch.empty(0, d)
        rewards = torch.empty(0)
        
        mu_post, Sigma_post = compute_true_posterior(
            played_features, rewards, lambda_prior, sigma_reward
        )
        
        # Posterior should be prior: N(0, λ^-1 I)
        self.assertTrue(torch.allclose(mu_post, torch.zeros(d), atol=1e-5))
        expected_cov = torch.eye(d) / lambda_prior
        self.assertTrue(torch.allclose(Sigma_post, expected_cov, atol=1e-5))
    
    def test_posterior_with_single_observation(self):
        """Test posterior update with one observation."""
        d = 5  # Use smaller dimension for clarity
        lambda_prior = 1.0
        sigma_reward = 0.5
        eta = 1.0
        
        # Single observation: X = [1, 0, 0, 0, 0], r = 2.0
        X = torch.zeros(1, d)
        X[0, 0] = 1.0
        r = torch.tensor([2.0])
        
        mu_post, Sigma_post = compute_true_posterior(X, r, lambda_prior, sigma_reward, eta)
        
        # Check posterior mean is non-zero in first dimension
        self.assertGreater(mu_post[0].item(), 0.0)
        # Other dimensions should be near zero
        self.assertTrue(torch.allclose(mu_post[1:], torch.zeros(d-1), atol=0.1))
        
        # Check posterior covariance is positive definite
        eigenvalues = torch.linalg.eigvalsh(Sigma_post)
        self.assertTrue((eigenvalues > 0).all(), "Covariance must be positive definite")
    
    def test_posterior_reduces_uncertainty(self):
        """Test that posterior variance decreases with more data."""
        d = 10
        lambda_prior = 1.0
        sigma_reward = 0.5
        
        # Generate random features and rewards
        torch.manual_seed(42)
        X_small = torch.randn(10, d)
        r_small = torch.randn(10)
        
        X_large = torch.randn(100, d)
        r_large = torch.randn(100)
        
        _, Sigma_small = compute_true_posterior(X_small, r_small, lambda_prior, sigma_reward)
        _, Sigma_large = compute_true_posterior(X_large, r_large, lambda_prior, sigma_reward)
        
        # More data should reduce posterior variance
        var_small = Sigma_small.diag().mean().item()
        var_large = Sigma_large.diag().mean().item()
        self.assertLess(var_large, var_small, "More data should reduce variance")


class TestFeatureMaps(unittest.TestCase):
    """Test block diagonal feature maps."""
    
    def test_phi_linear_shape(self):
        """Test that phi_linear creates correct block diagonal structure."""
        d = 5
        nb_arms = 3
        x = torch.randn(d)
        
        phi_x = phi_linear(x, nb_arms)
        
        # Should be (nb_arms, d * nb_arms)
        self.assertEqual(phi_x.shape, (nb_arms, d * nb_arms))
        
        # Each row should be block diagonal
        for i in range(nb_arms):
            # Check that only block i is non-zero
            for j in range(nb_arms):
                block = phi_x[i, j*d:(j+1)*d]
                if i == j:
                    self.assertTrue(torch.allclose(block, x))
                else:
                    self.assertTrue(torch.allclose(block, torch.zeros(d)))
    
    def test_phi_a_linear_extracts_correct_block(self):
        """Test that phi_a extracts the correct arm's block."""
        d = 5
        nb_arms = 4
        x = torch.randn(d)
        
        for arm in range(nb_arms):
            phi_a = phi_a_linear(x, arm, nb_arms)
            
            # Should be (d * nb_arms,)
            self.assertEqual(phi_a.shape, (d * nb_arms,))
            
            # Only the arm's block should be non-zero
            for i in range(nb_arms):
                block = phi_a[i*d:(i+1)*d]
                if i == arm:
                    self.assertTrue(torch.allclose(block, x))
                else:
                    self.assertTrue(torch.allclose(block, torch.zeros(d)))


class TestPosteriorSampling(unittest.TestCase):
    """Test sampling from posterior distributions."""
    
    def test_sample_from_posterior_shape(self):
        """Test that sampling produces correct shape."""
        d = 10
        n_samples = 100
        
        mu = torch.randn(d)
        Sigma = torch.eye(d)
        
        samples = sample_from_posterior(mu, Sigma, n_samples)
        
        self.assertEqual(samples.shape, (n_samples, d))
    
    def test_sample_statistics(self):
        """Test that samples have correct mean and covariance."""
        d = 5
        n_samples = 5000  # Large number for statistical accuracy
        
        mu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        Sigma = torch.eye(d) * 0.5
        
        torch.manual_seed(42)
        samples = sample_from_posterior(mu, Sigma, n_samples)
        
        # Check mean
        sample_mean = samples.mean(dim=0)
        self.assertTrue(torch.allclose(sample_mean, mu, atol=0.1))
        
        # Check covariance (roughly)
        sample_cov = torch.cov(samples.T)
        self.assertTrue(torch.allclose(sample_cov, Sigma, atol=0.1))


class TestWassersteinDistance(unittest.TestCase):
    """Test Wasserstein distance computation."""
    
    def test_identical_distributions(self):
        """Test that W2 distance is zero for identical distributions."""
        n_samples = 100
        d = 10
        
        # Same samples
        samples1 = torch.randn(n_samples, d)
        samples2 = samples1.clone()
        
        dist = compute_wasserstein_distance(samples1, samples2)
        
        self.assertAlmostEqual(dist, 0.0, places=5)
    
    def test_empty_samples(self):
        """Test handling of empty sample sets."""
        samples1 = torch.empty(0, 10)
        samples2 = torch.randn(100, 10)
        
        dist = compute_wasserstein_distance(samples1, samples2)
        self.assertTrue(np.isnan(dist))
    
    def test_mean_shift(self):
        """Test that mean shift increases distance."""
        n_samples = 1000
        d = 10
        
        torch.manual_seed(42)
        samples1 = torch.randn(n_samples, d)
        samples2 = torch.randn(n_samples, d) + 2.0  # Shift mean by 2
        
        dist = compute_wasserstein_distance(samples1, samples2)
        
        # Distance should be roughly 2.0 (mean shift in 2D)
        self.assertGreater(dist, 1.5)
        self.assertLess(dist, 3.0)


class TestSeedSetting(unittest.TestCase):
    """Test random seed functionality."""
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed ensures reproducibility."""
        set_seed(123)
        x1 = torch.randn(10)
        y1 = np.random.randn(10)
        
        set_seed(123)
        x2 = torch.randn(10)
        y2 = np.random.randn(10)
        
        self.assertTrue(torch.allclose(x1, x2))
        self.assertTrue(np.allclose(y1, y2))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def test_full_pipeline(self):
        """Test complete pipeline: data generation -> posterior computation -> sampling."""
        # Generate data
        k_arms, d_dim, t_horizon = 3, 10, 100
        contexts, true_betas, _, _ = generate_synthetic_data(
            k_arms, d_dim, t_horizon, 1.0, 0.5, seed=42, correlated=False
        )
        
        # Simulate playing arm 0 for 50 timesteps
        arm_idx = 0
        played_contexts = contexts[:50]
        played_features = torch.stack([phi_a_linear(x, arm_idx, k_arms) for x in played_contexts])
        
        # Generate rewards
        true_beta_arm0 = true_betas[arm_idx]
        rewards = torch.stack([x @ true_beta_arm0 + 0.1 * torch.randn(1) for x in played_contexts]).squeeze()
        
        # Extract arm-specific features
        arm_features = played_features[:, arm_idx*d_dim:(arm_idx+1)*d_dim]
        
        # Compute posterior
        mu_post, Sigma_post = compute_true_posterior(arm_features, rewards, 1.0, 0.5)
        
        # Sample from posterior
        samples = sample_from_posterior(mu_post, Sigma_post, 100)
        
        # Basic sanity checks
        self.assertEqual(samples.shape, (100, d_dim))
        self.assertFalse(torch.isnan(samples).any())
        self.assertFalse(torch.isinf(samples).any())


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
