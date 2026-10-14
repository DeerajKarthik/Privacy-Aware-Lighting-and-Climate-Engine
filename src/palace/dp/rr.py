"""k-ary randomised response (local DP). eps=inf => no noise."""
import numpy as np

def keep_prob(eps, K=3):
    return 1.0 if np.isinf(eps) else np.exp(eps) / (np.exp(eps) + K - 1)

def channel(eps, K=3):
    """Row-stochastic matrix C[true, reported]."""
    p = keep_prob(eps, K); q = (1 - p) / (K - 1); C = np.full((K, K), q); np.fill_diagonal(C, p); return C

def randomise(v, eps, rng, K=3):
    v = np.asarray(v); p = keep_prob(eps, K)
    flip = rng.random(v.shape) > p
    other = (v + rng.integers(1, K, v.shape)) % K          # uniform over the K-1 other categories
    return np.where(flip, other, v)

def debias_freq(obs_freq, eps, K=3):
    """Unbiased estimator of the true category frequency from RR'd reports."""
    p = keep_prob(eps, K); q = (1 - p) / (K - 1)
    return (np.asarray(obs_freq) - q) / (p - q)
