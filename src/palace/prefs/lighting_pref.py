"""SYNTHETIC circadian CCT preference c*_u(t) = a_u + b_u * circadian(t). ALL SYNTHETIC GROUND TRUTH (no public per-user dataset).
Warmer in the evening, cooler in the morning; ranges ASSUMED within 2700-6500 K (literature on circadian lighting to be verified before citing)."""
import numpy as np

def circadian(hour):
    """+1 around late morning/midday (cool/bright), -1 in early morning and evening (warm): cosine centred at 12:00."""
    return np.cos(2 * np.pi * (np.asarray(hour, float) - 12.0) / 24.0)

def sample_cct_params(n, rng):
    a = rng.normal(4400, 300, n); b = rng.uniform(300, 900, n); return a, b      # SYNTHETIC

def cct_pref(a, b, hour): return np.clip(a + b * circadian(hour), 2700, 6500)
