"""Adversary observing released actuation logs (dim, CCT, setpoint) and/or the zone meter trace; predicts hidden presence y_t and daily arrival time.
Trained on logs from DISJOINT days (validation-split replay of the same site/controller). Strong assumption: knows the controller & has comparable data volume."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

def features(series, use_actuation=True, use_meter=True, W=8):
    dim, cct, sp, kw = series["dim"], series["cct"], series["sp"], series["kw"]; T = len(dim); idx = np.asarray(series["idx"]).astype("datetime64[ns]")
    h = (idx.astype("datetime64[m]").astype(np.int64) % 1440) / 60.0; cols = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]
    sig = ([dim, cct / 6500.0, sp / 30.0] if use_actuation else []) + ([kw] if use_meter else [])
    for s in sig:
        s = np.asarray(s, float)
        for lag in range(-W // 2, W // 2 + 1):          # a window around t (adversary sees the full log, incl. later entries)
            cols.append(np.roll(s, -lag) if True else s)
    return np.stack(cols, 1)

def fit_eval(train_series, test_series, use_actuation=True, use_meter=True, seed=0):
    Xtr = np.concatenate([features(s, use_actuation, use_meter) for s in train_series]); ytr = np.concatenate([s["presence"] > 0 for s in train_series]).astype(int)
    Xte = np.concatenate([features(s, use_actuation, use_meter) for s in test_series]); yte = np.concatenate([s["presence"] > 0 for s in test_series]).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed).fit(Xtr, ytr); p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan")
    # arrival time: first step of each day with p>0.5 (per 96-step day)
    d_true, d_pred = [], []; pp = p.reshape(-1, 96); yy = yte.reshape(-1, 96)
    for a, b in zip(pp, yy):
        if b.any(): d_true.append(np.argmax(b) * 15); d_pred.append(np.argmax(a > 0.5) * 15 if (a > 0.5).any() else 12 * 60)
    return dict(auroc=auc, auprc=float(average_precision_score(yte, p)) if len(np.unique(yte)) > 1 else float("nan"), leakage=(auc - 0.5) / 0.5 if auc == auc else float("nan"),
                arrival_mae_min=float(np.mean(np.abs(np.array(d_true) - np.array(d_pred)))) if d_true else float("nan"), n_test=int(len(yte)))
