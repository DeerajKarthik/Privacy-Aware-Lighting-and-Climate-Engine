from __future__ import annotations
import numpy as np, torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from palace.fl.data import CTX, NORIG, H, BPD
from palace.models.metrics import binary_report

@torch.no_grad()
def predict_logits(model, rec, dev, bs=32):
    model.eval(); L, Aout = [], []
    for i in range(0, len(rec["X"]), bs):
        lo, ar = model(rec["X"][i:i + bs].to(dev)); L.append(lo.float().cpu()); Aout.append(ar.float().cpu())
    return torch.cat(L).numpy(), torch.cat(Aout).numpy()

def fit_temperature(logits, y, m):
    z, t, w = logits[m > 0], y[m > 0], None
    if len(t) == 0: return 1.0
    def nll(T): p = 1 / (1 + np.exp(-z / T)); p = np.clip(p, 1e-7, 1 - 1e-7); return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.2, 10), method="bounded").x)

def best_threshold(p, y):
    if len(np.unique(y)) < 2: return 0.5
    ths = np.linspace(0.05, 0.95, 37); f = [f1_score(y, p >= t, zero_division=0) for t in ths]; return float(ths[int(np.argmax(f))])

def eval_site(model, c, dev):
    """c: Client (has .val .test records). Temperature + F1 threshold are fitted on VALIDATION only."""
    lv, av = predict_logits(model, c.val, dev); lt, at = predict_logits(model, c.test, dev)
    yv, mv = c.val["Y"].numpy(), c.val["M"].numpy(); yt, mt = c.test["Y"].numpy(), c.test["M"].numpy()
    T = fit_temperature(lv, yv, mv); pv = 1 / (1 + np.exp(-lv / T)); pt = 1 / (1 + np.exp(-lt / T))
    th = best_threshold(pv[mv > 0], yv[mv > 0]); r = binary_report(pt[mt > 0], yt[mt > 0]); r["f1"] = float(f1_score(yt[mt > 0], pt[mt > 0] >= th, zero_division=0)); r["T"] = T; r["thr"] = th; r["val_auroc"] = binary_report(pv[mv > 0], yv[mv > 0])["auroc"]
    A = c.test["A"].numpy(); ok = A[:, 1] > 0
    r["arrival_mae_min"] = float(np.mean(np.abs(at[ok] - A[ok, 0])) * 24 * 60) if ok.any() else float("nan"); r["n_days_test"] = int(len(yt))
    return r, pt

# ---------- naive baselines (fit on TRAIN only) ----------
def _target_bins(n_days):
    o = np.arange(NORIG)[:, None] + 1 + np.arange(H)[None, :]; return o            # bin-of-day of each target [88,8]

def baseline_reports(c):
    tr, te, va = c.train, c.test, c.val; out = {}
    yt, mt = te["Y"].numpy(), te["M"].numpy(); n = len(yt); tb = _target_bins(n)
    # persistence: current presence at origin repeated
    cur = te["X"][:, CTX:CTX + NORIG, 0].numpy()[:, :, None] * np.ones((1, 1, H)); out["persistence"] = binary_report(cur[mt > 0], yt[mt > 0])
    # time-of-day x weekday profile (per site) from train
    def dow(rec): return np.asarray(rec["days"].dayofweek)
    prof = np.zeros((7, BPD)); cnt = np.zeros((7, BPD))
    Ytr = tr["X"][:, CTX:CTX + BPD, 0].numpy(); Mtr = tr["X"][:, CTX:CTX + BPD, 8].numpy()
    for d_, y_, m_ in zip(dow(tr), Ytr, Mtr): prof[d_] += y_ * m_; cnt[d_] += m_
    prof = np.where(cnt > 0, prof / np.maximum(cnt, 1), Ytr.mean() if len(Ytr) else 0.5)
    def prof_feat(rec): return np.stack([prof[d_][tb] for d_ in dow(rec)])           # [n,88,8]
    pt = prof_feat(te); out["profile"] = binary_report(pt[mt > 0], yt[mt > 0])
    # logistic regression on [current presence, mean last 4 bins, profile prob, horizon one-hot]
    def feats(rec):
        x = rec["X"][:, :, 0].numpy(); cur = x[:, CTX:CTX + NORIG]; last4 = np.stack([x[:, CTX + o - 3:CTX + o + 1].mean(1) for o in range(NORIG)], 1); pf = prof_feat(rec)
        f = [np.broadcast_to(cur[..., None], pf.shape), np.broadcast_to(last4[..., None], pf.shape), pf]; f += [np.broadcast_to((np.arange(H) == h).astype(float)[None, None], pf.shape) for h in range(H)]
        return np.stack(f, -1)
    Ftr, Fte = feats(tr), feats(te); ytr, mtr = tr["Y"].numpy(), tr["M"].numpy()
    if len(np.unique(ytr[mtr > 0])) > 1:
        lr = LogisticRegression(max_iter=300).fit(Ftr[mtr > 0], ytr[mtr > 0]); out["logreg"] = binary_report(lr.predict_proba(Fte[mt > 0])[:, 1], yt[mt > 0])
    # arrival baseline: median train arrival per weekday
    A = tr["A"].numpy(); ok = A[:, 1] > 0; med = {d_: np.median(A[ok & (dow(tr) == d_), 0]) if (ok & (dow(tr) == d_)).any() else np.median(A[ok, 0]) for d_ in range(7)} if ok.any() else None
    At = te["A"].numpy(); okt = At[:, 1] > 0
    if med is not None and okt.any(): out["profile"]["arrival_mae_min"] = float(np.mean(np.abs(np.array([med[d_] for d_ in dow(te)])[okt] - At[okt, 0])) * 24 * 60)
    return out
