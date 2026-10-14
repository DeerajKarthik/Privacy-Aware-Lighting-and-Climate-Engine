"""SYNTHETIC occupants (hidden ground truth: delta_u, L*_u, c*_u are SYNTHETIC) + RR-privatised votes + grid-Bayes posterior.
Votes: 0=too cold/dim, 1=ok, 2=too warm/bright. Reports pass through randomised response with eps_fb; the learner's likelihood uses the known channel."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from palace.dp.rr import channel, randomise

T_NEUTRAL, W_T, S_T = 24.5, 1.5, 0.7      # ASSUMED neutral temp, half-width of 'ok' band, vote noise (degC) -- synthetic
W_L, S_L = 0.35, 0.15                     # ASSUMED half-width and noise on ln(lux) -- synthetic

def sample_users(n, rng, heavy=False):
    """Hidden thermal offset delta ~ N(0,1) (or Student-t(3) scaled if heavy), preferred lux ~ LogNormal(ln500, .25). SYNTHETIC."""
    delta = rng.standard_t(3, n) * 0.8 if heavy else rng.normal(0, 1.0, n)
    ln_l = np.log(500) + (rng.standard_t(3, n) * 0.18 if heavy else rng.normal(0, 0.25, n))
    return delta, ln_l

def _probs(x, centre, w, s):
    """Ordered-probit: P(cold/dim)=Phi((centre-w-x)/s), P(hot/bright)=Phi((x-centre-w)/s); x, centre broadcast."""
    lo = norm.cdf((centre - w - x) / s); hi = norm.cdf((x - centre - w) / s); return np.stack([lo, 1 - lo - hi, hi], -1)

def true_votes(kind, x, truth, rng):
    c = T_NEUTRAL + truth if kind == "thermal" else truth
    w, s = (W_T, S_T) if kind == "thermal" else (W_L, S_L)
    P = _probs(x, c, w, s); u = rng.random(np.shape(x)); return (u > P[..., 0]).astype(int) + (u > P[..., 0] + P[..., 1]).astype(int)

class Posterior:
    """Grid posterior over delta (thermal) or ln L* (lighting) for ONE occupant. Prior is the learner's ASSUMED distribution."""
    def __init__(self, kind, eps, prior_mean=0.0, prior_sd=1.0, grid=None):
        self.kind = kind; self.C = channel(eps)
        self.g = np.linspace(-4, 4, 801) * prior_sd + prior_mean if grid is None else grid
        self.logp = norm.logpdf(self.g, prior_mean, prior_sd)
    def update(self, x, report):
        c = T_NEUTRAL + self.g if self.kind == "thermal" else self.g; w, s = (W_T, S_T) if self.kind == "thermal" else (W_L, S_L)
        P = _probs(x, c, w, s)                       # (grid,3) true-vote probs
        lik = P @ self.C[:, report]                  # marginalise over the true vote through the known RR channel
        self.logp = self.logp + np.log(np.clip(lik, 1e-12, None))
    def mean(self):
        p = np.exp(self.logp - self.logp.max()); p /= p.sum(); return float((p * self.g).sum())

def simulate(kind, eps, n_votes, n_users, seed, heavy_truth=False, prior_sd=None):
    """Mean-squared error of posterior mean over users. heavy_truth=True => misspecification (truth heavy-tailed, learner assumes Gaussian)."""
    rng = np.random.default_rng(seed); delta, ln_l = sample_users(n_users, rng, heavy_truth)
    truth, pm, psd = (delta, 0.0, 1.0) if kind == "thermal" else (ln_l, np.log(500), 0.25)
    errs = np.empty(n_users)
    for u in range(n_users):
        x = rng.uniform(22, 30, n_votes) if kind == "thermal" else np.exp(rng.uniform(np.log(150), np.log(1000), n_votes))
        xx = x if kind == "thermal" else np.log(x)
        v = true_votes(kind, xx, truth[u], rng); r = randomise(v, eps, rng)
        post = Posterior(kind, eps, pm, psd if prior_sd is None else prior_sd)
        for xi, ri in zip(xx, r): post.update(xi, int(ri))
        errs[u] = post.mean() - truth[u]
    return float(np.sqrt(np.mean(errs ** 2))), float(np.mean(np.abs(errs)))
