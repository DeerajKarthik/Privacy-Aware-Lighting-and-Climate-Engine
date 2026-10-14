"""CEM-MPC planner + exhaustive first-action utility (used by APF). Forecast-only: never reads future true occupancy."""
from __future__ import annotations
import numpy as np
from palace.control.costs import DIMS, CCTS, SPS, ND, NC, NS, ALL_A, decode
from palace.prefs.lighting_pref import cct_pref

class Planner:
    def __init__(self, plan_zone, cm, ctrl, rng):
        self.z, self.cm, self.H, self.rng = plan_zone, cm, ctrl["horizon"], rng; self.w = cm.w; self.cem = ctrl["cem"]; self.prev = None
        self.Jmax = self.H * cm.jmax_step

    def _cost(self, o, p_occ, prev_a, belief, hours, first_occ=None):
        """o: rollout dict (B,H). p_occ: (H,) occupancy weights (step 0 may be replaced by first_occ). Returns J (B,)."""
        B, H = o["Ta"].shape; occ_w = np.broadcast_to(p_occ, (B, H)).copy()
        if first_occ is not None: occ_w[:, 0] = first_occ
        dth = self.cm.d_th(o["Ta"], belief["delta"]); cpref = cct_pref(belief["a"], belief["b"], hours)[None, :]
        dvis = self.cm.d_vis(o["lux"], belief["lux"], o["_cct"], cpref, True)
        e = self.cm.energy_norm(o["P_light"], o["P_hvac"])
        prev = np.concatenate([np.broadcast_to(np.array(prev_a, float)[None, None, :], (B, 1, 3)), np.stack([o["_dim"], o["_cct"], o["_sp"]], -1)[:, :-1]], 1)
        sm = self.cm.smooth(o["_dim"], o["_cct"], o["_sp"], (prev[..., 0], prev[..., 1], prev[..., 2]))
        return (self.w["w_e"] * e + self.w["w_c"] * occ_w * 0.5 * (dth + dvis) + self.w["w_s"] * sm).sum(1)

    def _roll(self, x, dim, cct, sp, t, n_plan):
        o = self.z.rollout(x, dim, cct, sp, t, n_plan=n_plan); o["_dim"], o["_cct"], o["_sp"] = dim, cct, sp; return o

    def plan(self, x, t, p_hat, n_typ, prev_a, belief, hours):
        """CEM over H steps using forecast occupancy p_hat (H,) [from origin t-1; EXCLUDES y_t]. Returns the best sequence (dim,cct,sp each (H,))."""
        H, B = self.H, self.cem["population"]; n_el = max(2, int(self.cem["elite_frac"] * B)); n_plan = p_hat * n_typ
        if self.prev is None: P = [np.full((H, n), 1.0 / n) for n in (ND, NC, NS)]
        else: P = [np.vstack([pp[1:], np.full((1, pp.shape[1]), 1.0 / pp.shape[1])]) for pp in self.prev]
        best = None
        for it in range(self.cem["iterations"]):
            idx = [np.stack([self.rng.choice(n, size=B, p=pp[h]) for h in range(H)], 1) for pp, n in zip(P, (ND, NC, NS))]
            dim, cct, sp = DIMS[idx[0]], CCTS[idx[1]], SPS[idx[2]]
            o = self._roll(x, dim, cct, sp, t, n_plan); J = self._cost(o, p_hat, prev_a, belief, hours); el = np.argsort(J)[:n_el]
            if best is None or J[el[0]] < best[0]: best = (J[el[0]], dim[el[0]].copy(), cct[el[0]].copy(), sp[el[0]].copy())
            for k, n in enumerate((ND, NC, NS)):
                for h in range(H):
                    f = np.bincount(idx[k][el, h], minlength=n) / n_el; P[k][h] = (1 - self.cem["smoothing"]) * P[k][h] + self.cem["smoothing"] * f + 1e-3; P[k][h] /= P[k][h].sum()
        self.prev = P
        return best[1], best[2], best[3]

    def first_action_utility(self, x, t, p_hat, n_typ, tail, prev_a, belief, hours, y_t):
        """u(a)=1-J~(a) for ALL 1089 first actions, tail (steps 1..H-1) fixed from the plan; step-0 comfort weight = live y_t."""
        H = self.H; da, ca, sa = decode(ALL_A); B = len(ALL_A); n_plan = p_hat * n_typ
        dim = np.concatenate([da[:, None], np.broadcast_to(tail[0][1:][None], (B, H - 1))], 1); cct = np.concatenate([ca[:, None], np.broadcast_to(tail[1][1:][None], (B, H - 1))], 1)
        sp = np.concatenate([sa[:, None], np.broadcast_to(tail[2][1:][None], (B, H - 1))], 1)
        o = self._roll(x, dim, cct, sp, t, n_plan); J = self._cost(o, p_hat, prev_a, belief, hours, first_occ=y_t)
        return np.clip(1 - J / self.Jmax, 0, 1)
