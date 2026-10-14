import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, f1_score

def ece(p, y, bins=15):
    """Expected calibration error, binary (p=prob of positive) or multiclass (p: (n,K), y int) via top-label confidence."""
    p = np.asarray(p, float); y = np.asarray(y)
    if p.ndim == 1: conf = np.where(p >= .5, p, 1 - p); pred = (p >= .5).astype(int); acc = (pred == y)
    else: conf = p.max(1); acc = (p.argmax(1) == y)
    e = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(conf, e[1:-1]), 0, bins - 1); tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(tot)

def binary_report(p, y, mask=None):
    p = np.asarray(p, float); y = np.asarray(y)
    if mask is not None: p, y = p[mask > 0], y[mask > 0]
    if len(np.unique(y)) < 2: return dict(auroc=np.nan, auprc=np.nan, brier=float(brier_score_loss(y, p)), ece=ece(p, y), n=len(y))
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)), ece=ece(p, y), n=int(len(y)))
