"""B0 fixed schedule, B1 PIR-reactive rule. Both depend only on time / live presence / weather (not on zone state),
so action series are precomputed vectors for a whole year."""
from __future__ import annotations
import numpy as np, pandas as pd

def b0(index: pd.DatetimeIndex, cfg):
    """Lights 100%/4000 K and setpoint 24 C during working hours (Mon-Fri 08:00-18:00), lights off & setpoint 30 C (AC effectively off) otherwise."""
    h = index.hour + index.minute / 60; work = (index.dayofweek < 5) & (h >= 8) & (h < 18)
    dim = np.where(work, 1.0, 0.0); cct = np.full(len(index), 4000.0); sp = np.where(work, 24.0, cfg["ac"]["setpoint_range_C"][1])
    return dim, cct, sp.astype(float)

def b1(presence, ghi_lux_day, cfg, step_min=15, off_delay_min=10, setback_after_min=30, target_lux=500.0):
    """PIR-reactive: lights on at presence with 10-min off-delay and daylight-linked dimming (dim so that daylight+electric ~ target);
    setpoint 24 C occupied, 28 C after `setback_after_min` of absence. Presence is the live (hold-extended) PIR signal."""
    from palace.sim.lighting import e_max
    T = len(presence); em = e_max(cfg["lighting"])
    off_bins = int(np.ceil(off_delay_min / step_min)); sb_bins = int(np.ceil(setback_after_min / step_min))
    since = np.zeros(T, int); s = 10 ** 6
    for t in range(T):
        s = 0 if presence[t] > 0 else s + 1; since[t] = s
    lights_on = since < max(off_bins, 1) * (1) if off_bins > 0 else presence > 0
    lights_on = (since <= off_bins - 1) | (presence > 0)
    dim = np.where(lights_on, np.clip((target_lux - ghi_lux_day) / em, 0, 1), 0.0); dim = np.round(dim * 10) / 10
    sp = np.where(since >= sb_bins, 28.0, 24.0)
    return dim, np.full(T, 4000.0), sp
