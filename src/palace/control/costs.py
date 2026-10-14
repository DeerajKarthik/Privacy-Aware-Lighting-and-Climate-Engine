"""Normalised cost terms in [0,1] (so epsilon-sensitivity of the APF is well defined) and shared constants."""
from __future__ import annotations
import numpy as np
from palace.sim import lighting
from palace.sim.comfort import visual_discomfort

DIMS = np.round(np.arange(0, 1.01, 0.1), 1); CCTS = np.linspace(2700, 6500, 11); SPS = np.arange(22, 31, 1, dtype=float)
ND, NC, NS = len(DIMS), len(CCTS), len(SPS)               # 11 x 11 x 9 = 1089

def decode(idx):
    idx = np.asarray(idx); d = idx // (NC * NS); c = (idx // NS) % NC; s = idx % NS; return DIMS[d], CCTS[c], SPS[s]
ALL_A = np.arange(ND * NC * NS)

class CostModel:
    def __init__(self, cfg, ctrl, pnc_table, e_step_max):
        self.cfg, self.w = cfg, ctrl["weights"]; self.pnc_grid, self.pnc_vals = pnc_table; self.pnc_max = float(self.pnc_vals.max()); self.e_step_max = e_step_max
        self.jmax_step = self.w["w_e"] + self.w["w_c"] + self.w["w_s"]
    def d_th(self, Ta, delta):
        """Relative thermal discomfort in [0,1]: 1 - p(no change | Ta - delta_u)/max_Ta p(no change). delta_u = personal neutral offset (positive = prefers warmer)."""
        p = np.interp(np.asarray(Ta, float) - delta, self.pnc_grid, self.pnc_vals); return np.clip(1 - p / self.pnc_max, 0, 1)
    def d_vis(self, lux, lux_pref, cct, cct_pref, occupied=True):
        return visual_discomfort(lux, lux_pref, cct, cct_pref, cap=np.log(3.0), task=np.asarray(occupied, bool), floor=self.cfg["lighting"]["task_lux_floor"])
    def energy_norm(self, P_light, P_hvac, P_equip_ignored=None):
        return np.clip((P_light + P_hvac) * self.cfg["dt_s"] / 3.6e6 / self.e_step_max, 0, 1)
    def smooth(self, dim, cct, sp, prev):
        return (np.abs(dim - prev[0]) / 1.0 + np.abs(cct - prev[1]) / 3800.0 + np.abs(sp - prev[2]) / 8.0) / 3.0
