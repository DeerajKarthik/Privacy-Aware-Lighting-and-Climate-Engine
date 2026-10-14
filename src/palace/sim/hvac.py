"""Cooling-only split AC. Capacity-limited, deadband, deadbeat-to-setpoint using the exact discrete model."""
from __future__ import annotations
import numpy as np

def cop(To, ref, t_ref, slope, floor):
    return np.maximum(ref - slope * (To - t_ref), floor)

def cooling_power(Qcool, To, c):
    """Electric power (W) for delivered cooling Qcool (W, >=0)."""
    plr = np.clip(Qcool / c["Qmax_W"], 0, 1)
    eff = cop(To, c["cop_ref"], c["cop_t_ref_C"], c["cop_slope_per_K"], c["cop_floor"]) * (1 + c["part_load_gain"] * (1 - plr))
    return np.where(Qcool > 0, Qcool / eff + c["fan_standby_W"], 0.0)

def required_cooling(Ta_free_next, gain_per_W, setpoint, c):
    """Qcool (>=0, capacity-limited) so that Ta_next hits the setpoint; no cooling inside the deadband.
    Ta_free_next: next-step air temperature with Qhvac=0; gain_per_W: dTa_next/dQhvac (K/W, >0 for heating sign)."""
    need = (Ta_free_next - setpoint) / gain_per_W        # W of cooling (positive)
    on = Ta_free_next > setpoint + c["deadband_K"]
    return np.clip(np.where(on, need, 0.0), 0.0, c["Qmax_W"])
