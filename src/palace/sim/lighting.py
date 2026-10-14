"""Dimmable tunable-white LED: delivered lux, electrical power."""
from __future__ import annotations
import numpy as np

def e_max(l):   # lux at full output on workplane
    return l["pmax_w_per_m2"] * l["led_lm_per_W"] * l["utilisation_factor"]

def power_w(dim, cct, area, l):
    d = np.asarray(dim, float)
    f = d * (1 + l["driver_loss_at_low_dim"] * (1 - d))                       # driver efficiency loss at low dim
    g = 1 + l["cct_penalty_per_2500K"] * np.abs(np.asarray(cct, float) - 4000.0) / 2500.0
    return area * l["pmax_w_per_m2"] * f * g + area * l["standby_w_per_m2"]

def electric_lux(dim, l):
    return np.asarray(dim, float) * e_max(l)

def total_lux(dim, e_day, l, occupied=None):
    e = electric_lux(dim, l) + e_day
    if occupied is not None: e = np.where(occupied, np.maximum(e, l["min_egress_lux_occupied"]), e)
    return e
