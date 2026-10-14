from __future__ import annotations
import numpy as np

def pareto_mask(P):
    """P: (n, k) objectives, all MINIMISED. True where non-dominated."""
    P = np.asarray(P, float); n = len(P); keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(P <= P[i], 1) & np.any(P < P[i], 1)
        if dom.any(): keep[i] = False
    return keep

def hypervolume(P, ref, n_mc=200000, seed=0):
    """Monte-Carlo hypervolume dominated by P within the box [min(P), ref] (all minimised). Deterministic given seed."""
    P = np.asarray(P, float); ref = np.asarray(ref, float); lo = np.minimum(P.min(0), ref); rng = np.random.default_rng(seed)
    s = rng.uniform(lo, ref, (n_mc, P.shape[1])); dom = np.zeros(n_mc, bool)
    for p in P[pareto_mask(P)]: dom |= np.all(s >= p, 1)
    return float(dom.mean() * np.prod(ref - lo))
