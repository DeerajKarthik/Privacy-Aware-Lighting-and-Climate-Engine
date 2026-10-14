from __future__ import annotations
import numpy as np

def boot_ci(x, n=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed); m = rng.choice(x, (n, len(x))).mean(1); return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))

def paired(a, b, n=10000, seed=0):
    """a, b aligned arrays (paired by site/season/seed). Returns mean diff (a-b), CI, Cohen's d_z, two-sided bootstrap p (sign-flip on centred diffs)."""
    d = np.asarray(a, float) - np.asarray(b, float); d = d[~np.isnan(d)]; mean, lo, hi = boot_ci(d, n, seed)
    dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
    rng = np.random.default_rng(seed + 1); null = (rng.choice(d - d.mean(), (n, len(d))).mean(1)); p = float((np.abs(null) >= abs(d.mean())).mean())
    return dict(mean=mean, ci_low=lo, ci_high=hi, cohen_dz=dz, p=max(p, 1.0 / (n + 1)), n=int(len(d)))

def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()
