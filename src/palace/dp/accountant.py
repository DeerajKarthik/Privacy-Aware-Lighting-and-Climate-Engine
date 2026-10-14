"""Cumulative-epsilon accounting across FL rounds with Opacus' RDP accountant (one persistent accountant per client)."""
from __future__ import annotations
import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.analysis import rdp as R

def delta_for(n_train_days: int) -> float:
    return 1.0 / (10.0 * n_train_days)

def cumulative_epsilon(acc: RDPAccountant, delta: float) -> float:
    return float(acc.get_epsilon(delta=delta)) if len(acc.history) else 0.0

def hand_rdp_epsilon(noise_multiplier: float, steps: int, delta: float, orders=None):
    """No-subsampling (q=1) Gaussian mechanism: RDP(alpha)=steps*alpha/(2 sigma^2); convert with Balle et al. 2020 Thm 21 (same rule Opacus uses)."""
    orders = np.array(orders if orders is not None else R.DEFAULT_ALPHAS if hasattr(R, "DEFAULT_ALPHAS") else [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)), float)
    rdp = steps * orders / (2 * noise_multiplier ** 2)
    eps = rdp - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    return float(np.nanmin(eps))
