"""Train LightGBM thermal-preference model with contributor-held-out split; temperature scaling on val; save + metrics."""
import json, sys, joblib; from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, accuracy_score, f1_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.prefs.thermal_pref import *; from palace.models.metrics import ece
d, stats = load_clean(ROOT); print(stats)
res = {"data": stats, "seeds": {}}
cats = {c: sorted(d[c].dropna().unique()) for c in CAT}; best = None
for seed in range(5):
    tr, va, te = contributor_split(d, seed)
    Xtr, Xva, Xte = (design(x, cats) for x in (tr, va, te))
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(Xtr, tr.y, eval_set=[(Xva, va.y)], callbacks=[lgb.early_stopping(30, verbose=False)])
    pv = m.predict_proba(Xva); f = lambda T: log_loss(va.y, (lambda l: np.exp(l) / np.exp(l).sum(1, keepdims=True))(np.log(np.clip(pv, 1e-9, 1)) / T), labels=[0, 1, 2])
    T = float(minimize_scalar(f, bounds=(0.3, 5), method="bounded").x); cal = Calibrated(m, T, cats); pt = cal.predict_proba(Xte); pred = pt.argmax(1)
    base = np.bincount(tr.y, minlength=3) / len(tr)
    r = dict(T=T, acc=float(accuracy_score(te.y, pred)), macro_f1=float(f1_score(te.y, pred, average="macro")), logloss=float(log_loss(te.y, pt, labels=[0, 1, 2])), ece=ece(pt, te.y.values),
             logloss_prior_baseline=float(log_loss(te.y, np.tile(base, (len(te), 1)), labels=[0, 1, 2])), n_test=len(te), n_test_contrib=int(te.contributor.nunique()))
    res["seeds"][seed] = r; print(seed, r)
    if seed == 0: best = cal
joblib.dump(best, ROOT / "results/thermal_pref_lgbm.joblib")
for k in ("acc", "macro_f1", "logloss", "ece", "logloss_prior_baseline"):
    v = np.array([res["seeds"][s][k] for s in res["seeds"]]); res[k + "_mean"], res[k + "_sd"] = float(v.mean()), float(v.std(ddof=1))
res["note"] = "REAL ASHRAE DB II data; contributor-held-out splits over 5 random contributor partitions (seed = partition seed, not model-only seed)"
json.dump(res, open(ROOT / "results/thermal_pref_metrics.json", "w"), indent=1)
print({k: round(v, 3) for k, v in res.items() if k.endswith("_mean")})
print("p_no_change vs Ta:", {t: round(float(best.p_no_change(t)[0]), 2) for t in (22, 24, 26, 28, 30, 32)})
