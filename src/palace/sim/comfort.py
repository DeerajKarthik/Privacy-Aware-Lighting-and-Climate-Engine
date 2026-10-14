"""PMV/PPD per ISO 7730 / ASHRAE 55 (Fanger iterative solution) and visual discomfort."""
from __future__ import annotations
import numpy as np

def pmv(ta, tr, vel, rh, met, clo, wme=0.0):
    ta, tr, vel, rh, met, clo = (np.asarray(x, float) for x in (ta, tr, vel, rh, met, clo))
    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (ta + 235))
    icl = 0.155 * clo; m = met * 58.15; w = wme * 58.15; mw = m - w
    fcl = np.where(icl <= 0.078, 1 + 1.29 * icl, 1.05 + 0.645 * icl)
    hcf = 12.1 * np.sqrt(vel); taa = ta + 273; tra = tr + 273
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
    p1 = icl * fcl; p2 = p1 * 3.96; p3 = p1 * 100; p4 = p1 * taa; p5 = 308.7 - 0.028 * mw + p2 * (tra / 100) ** 4
    xn = tcla / 100; xf = tcla / 50; hc = hcf
    for _ in range(150):
        xf = (xf + xn) / 2
        hcn = 2.38 * np.abs(100 * xf - taa) ** 0.25
        hc = np.maximum(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
    tcl = 100 * xn - 273
    hl1 = 3.05e-3 * (5733 - 6.99 * mw - pa); hl2 = np.where(mw > 58.15, 0.42 * (mw - 58.15), 0.0)
    hl3 = 1.7e-5 * m * (5867 - pa); hl4 = 0.0014 * m * (34 - ta)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100) ** 4); hl6 = fcl * hc * (tcl - ta)
    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    return ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

def ppd(p): return 100 - 95 * np.exp(-0.03353 * p ** 4 - 0.2179 * p ** 2)

def visual_discomfort(e_wp, lux_pref, cct, cct_pref, cap=1.0, task=None, floor=300.0):
    """D_vis in [0,1]: |ln(E/L*)| normalised by `cap`, + penalty if below task floor, + CCT mismatch |c-c*|/3800 (each clipped, mean-combined)."""
    d_lux = np.clip(np.abs(np.log(np.maximum(e_wp, 1e-3) / lux_pref)) / cap, 0, 1)
    pen = 0.0 if task is None else np.where(task & (e_wp < floor), 1.0, 0.0)
    d_cct = np.clip(np.abs(cct - cct_pref) / 3800.0, 0, 1)
    return np.clip((d_lux + pen + d_cct) / 3.0 * 1.5, 0, 1)   # weights documented: equal thirds, rescaled so a full miss on 2 of 3 saturates
