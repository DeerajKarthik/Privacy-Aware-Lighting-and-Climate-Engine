"""Number of people in a zone from CASAS-derived signals (documented function; tested)."""
from __future__ import annotations
import numpy as np

def n_people(presence, active_sensors, capacity=4, per_sensor=1.0):
    """n = 0 if no presence; else clip(ceil(per_sensor*active_sensors), 1, capacity). Single-sensor sites => 1 person when present."""
    n = np.ceil(per_sensor * np.asarray(active_sensors, float))
    n = np.clip(n, 1, capacity)
    return np.where(np.asarray(presence) > 0, n, 0).astype(float)
