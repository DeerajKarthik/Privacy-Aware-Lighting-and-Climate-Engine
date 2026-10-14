"""Vectorised zone simulator: batch of B candidate action sequences rolled out over H steps from a given state/time.
Perturbation hook: multiplicative factors on key parameters (Rao, Ram, Ca, Cm, COP_ref, window area, DF)."""
from __future__ import annotations
import copy, numpy as np
from palace.sim.thermal import Thermal
from palace.sim import hvac, lighting

PERT_KEYS = {"Rao": ("zone", "Rao_K_per_W"), "Ram": ("zone", "Ram_K_per_W"), "Ca": ("zone", "Ca_J_per_K"), "Cm": ("zone", "Cm_J_per_K"),
             "COP_ref": ("ac", "cop_ref"), "window_area": ("window", "area_frac_of_floor"), "DF": ("lighting", "daylight_factor")}

def perturb_cfg(cfg: dict, factors: dict | None):
    c = copy.deepcopy(cfg)
    for k, f in (factors or {}).items():
        sec, key = PERT_KEYS[k]; c[sec][key] = c[sec][key] * f
    return c

class Zone:
    def __init__(self, cfg: dict, inputs: dict):
        """inputs (arrays over the full series, length T): To (degC), solar_w (W through window, UNperturbed unit area-scaled by window_area),
        e_day_unit (lux per unit DF), n_people, met."""
        self.c = cfg; z = cfg["zone"]; self.area = z["floor_area_m2"]; self.vol = self.area * z["height_m"]
        self.th = Thermal(z["Ca_J_per_K"], z["Cm_J_per_K"], z["Rao_K_per_W"], z["Ram_K_per_W"], cfg["dt_s"])
        self.g = float(self.th.Bd[0, 1])                       # dTa_next / dQhvac
        self.inp = inputs
        self.win_scale = cfg["window"]["area_frac_of_floor"] / inputs.get("win_frac_ref", cfg["window"]["area_frac_of_floor"])
        self.ua_inf_per_ach = 1.2 * 1005.0 * self.vol / 3600.0

    def lux_day(self, t):
        return self.c["lighting"]["daylight_factor"] * self.inp["ghi"][t] * self.c["lighting"]["luminous_efficacy_lm_per_W_solar"]

    def step(self, x, dim, cct, sp, t):
        """x (B,2); dim,cct,sp (B,) or scalars; t int index. Returns x_next (B,2) and dict of (B,) outputs."""
        c = self.c; z = c["zone"]; To = self.inp["To"][t]; met = self.inp["met"][t]
        npl = self.inp["n_people"][t] if getattr(self, "_n_plan", None) is None else self._n_plan[t - self._t0]
        occ = npl > 0
        p_l = lighting.power_w(dim, cct, self.area, c["lighting"])
        p_e = self.area * (c["equipment"]["w_per_m2_occupied"] if occ else c["equipment"]["w_per_m2_standby"])
        q_p = npl * c["people"]["sensible_w_per_met"] * met
        q_int = q_p + p_l + p_e
        sol = self.inp["solar_w"][t] * self.win_scale
        Ta = x[:, 0]
        ach = z["infiltration_ach0"] + z["infiltration_k_ach_per_K"] * np.abs(To - Ta)
        q_inf = self.ua_inf_per_ach * ach * (To - Ta)
        q_air = z["solar_frac_air"] * sol + (1 - z["internal_frac_mass"]) * q_int + q_inf
        q_mass = (1 - z["solar_frac_air"]) * sol + z["internal_frac_mass"] * q_int
        zeros = np.zeros_like(Ta); u = np.stack([np.full_like(Ta, To), zeros, q_air + zeros, q_mass + zeros], axis=-1)
        x_free = self.th.step(x, u)
        sp = np.broadcast_to(np.asarray(sp, float), Ta.shape)
        qc = hvac.required_cooling(x_free[:, 0], self.g, sp, c["ac"])
        u[:, 1] = -qc
        xn = self.th.step(x, u)
        p_h = hvac.cooling_power(qc, To, c["ac"])
        lux = lighting.total_lux(np.broadcast_to(np.asarray(dim, float), Ta.shape), self.lux_day(t), c["lighting"], occ)
        return xn, dict(Ta=xn[:, 0], Tm=xn[:, 1], lux=lux, P_light=p_l, P_hvac=p_h, P_equip=np.full_like(Ta, p_e), Qcool=qc, Q_people=np.full_like(Ta, q_p))

    def rollout(self, x0, dim, cct, sp, t0, n_plan=None):
        """dim,cct,sp: (B,H). Returns dict of (B,H) arrays plus final state."""
        self._n_plan, self._t0 = n_plan, t0
        x = np.array(x0, float); B, H = np.shape(dim); x = np.broadcast_to(x, (B, 2)).copy(); out = {}
        for h in range(H):
            x, o = self.step(x, dim[:, h], cct[:, h], sp[:, h], t0 + h)
            for k, v in o.items(): out.setdefault(k, np.empty((B, H)))[:, h] = v
        out["x_final"] = x; self._n_plan = None
        return out
