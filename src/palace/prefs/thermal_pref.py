"""Thermal-preference model on ASHRAE DB II (REAL survey data). Output p(no change | Ta, clo, met, context)."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

CLASSES = ["cooler", "no change", "warmer"]
NUM = ["ta", "clo", "met", "rh", "t_out_monthly"]
CAT = ["season", "building_type"]

def load_clean(root: Path):
    m = pd.read_csv(root / "data/raw/ashrae/db_measurements_v2.1.0.csv.gz", low_memory=False)
    meta = pd.read_csv(root / "data/raw/ashrae/db_metadata.csv")[["building_id", "contributor", "country", "building_type", "climate", "cooling_type"]]
    n0 = len(m); d = m.merge(meta, on="building_id", how="left")
    d = d[d.thermal_preference.isin(CLASSES) & d.ta.notna()].copy()
    stats = dict(n_raw=n0, n_clean=len(d), n_india=int((d.country == "india").sum()), n_contributors=int(d.contributor.nunique()))
    d["y"] = d.thermal_preference.map({c: i for i, c in enumerate(CLASSES)}); d["contributor"] = d.contributor.fillna("unknown")
    return d, stats

def contributor_split(d, seed=0, fr=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed); cs = d.contributor.unique(); rng.shuffle(cs); n = len(cs)
    a, b = int(fr[0] * n), int((fr[0] + fr[1]) * n); g = {c: 0 for c in cs[:a]} | {c: 1 for c in cs[a:b]} | {c: 2 for c in cs[b:]}
    s = d.contributor.map(g); return d[s == 0], d[s == 1], d[s == 2]

def design(d, cats=None):
    X = d[NUM + CAT].copy()
    for c in CAT: X[c] = pd.Categorical(X[c], categories=cats[c] if cats else sorted(d[c].dropna().unique()))
    return X

class Calibrated:
    """LightGBM + temperature scaling fitted on the VALIDATION contributors only."""
    def __init__(self, model, T, cats): self.model, self.T, self.cats = model, T, cats
    def predict_proba(self, X):
        lg = np.log(np.clip(self.model.predict_proba(X), 1e-9, 1)) / self.T; lg -= lg.max(1, keepdims=True); p = np.exp(lg); return p / p.sum(1, keepdims=True)
    def p_no_change(self, ta, clo=0.6, met=1.2, rh=50.0, t_out_monthly=30.0, season="summer", building_type="office"):
        ta = np.atleast_1d(np.asarray(ta, float))
        X = pd.DataFrame({"ta": ta, "clo": clo, "met": met, "rh": rh, "t_out_monthly": t_out_monthly, "season": season, "building_type": building_type})
        for c in CAT: X[c] = pd.Categorical(X[c], categories=self.cats[c])
        return self.predict_proba(X)[:, 1]
