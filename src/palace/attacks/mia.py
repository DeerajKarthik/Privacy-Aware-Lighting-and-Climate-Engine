"""Membership inference on the occupancy forecaster. Members = training site-days; non-members = held-out (val+test) site-days of the SAME site.
(a) loss-threshold attack on the final FL models; (b) LiRA-lite: 8 shadow models + 1 target, trained as SINGLE-SITE (local) DP-SGD proxies of the FL local step.
An empirical AUDIT, not a proof of privacy."""
from __future__ import annotations
import numpy as np, torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from palace.models.occupancy_tcn import OccTCN, loss_fn
from palace.dp.accountant import delta_for

@torch.no_grad()
def day_losses(model, rec, dev, fw=0.25, gm=2.0):
    model.eval(); out = []
    for i in range(len(rec["X"])):
        x, y, m, a = (rec[k][i:i + 1].to(dev) for k in ("X", "Y", "M", "A")); out.append(float(loss_fn(model(x), y, m, a, fw, gm)))
    return np.array(out)

def attack_metrics(member_scores, nonmember_scores):
    """scores: higher => 'member'. Returns AUC and TPR@1%FPR."""
    y = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]; s = np.r_[member_scores, nonmember_scores]
    fpr, tpr, _ = roc_curve(y, s); t1 = float(np.interp(0.01, fpr, tpr))
    return dict(auc=float(roc_auc_score(y, s)), tpr_at_1fpr=t1, fpr=fpr.tolist(), tpr=tpr.tolist(), n_in=len(member_scores), n_out=len(nonmember_scores))

def loss_threshold_mia(model, rec_train, rec_heldout, dev, max_n=200, seed=0):
    rng = np.random.default_rng(seed); n = min(max_n, len(rec_train["X"]), len(rec_heldout["X"]))
    sub = lambda r: {k: (v[idx] if hasattr(v, "__getitem__") and k in "XYMA" else v) for k, v in r.items()}
    it, ih = rng.choice(len(rec_train["X"]), n, replace=False), rng.choice(len(rec_heldout["X"]), n, replace=False)
    rt = {k: rec_train[k][it] for k in "XYMA"}; rh = {k: rec_heldout[k][ih] for k in "XYMA"}
    return attack_metrics(-day_losses(model, rt, dev), -day_losses(model, rh, dev))

def _train_local(rec, eps, epochs, seed, cfg, dev, batch_days=16):
    torch.manual_seed(seed); m = cfg["model"]; net = OccTCN(ch=m["channels"], k=m["kernel"], dil=tuple(m["dilations"])).to(dev); n = len(rec["X"]); q = min(1.0, batch_days / n)
    ds = torch.utils.data.TensorDataset(*(rec[k].to(dev) for k in "XYMA")); opt = torch.optim.Adam(net.parameters(), lr=cfg["train"]["lr"])
    ld = torch.utils.data.DataLoader(ds, batch_size=max(1, int(round(q * n))), shuffle=True, generator=torch.Generator().manual_seed(seed))
    if not np.isinf(eps):
        steps = epochs * int(round(1 / q)); sig = get_noise_multiplier(target_epsilon=eps, target_delta=delta_for(n), sample_rate=q, steps=steps, accountant="rdp")
        net, opt, ld = PrivacyEngine(accountant="rdp").make_private(module=net, optimizer=opt, data_loader=ld, noise_multiplier=sig, max_grad_norm=cfg["train"]["max_grad_norm"], poisson_sampling=True)
    net.train()
    for _ in range(epochs):
        for X, Y, M, A in ld:
            if len(X) == 0: continue
            opt.zero_grad(); loss_fn(net(X), Y, M, A, cfg["train"]["focal_weight"], cfg["train"]["focal_gamma"]).backward(); opt.step()
    return net._module if hasattr(net, "_module") else net

def lira_lite(rec_pop, eps, cfg, dev, n_shadow=8, epochs=15, seed=0):
    """rec_pop: records of one site (train+val+test days pooled, dict X,Y,M,A). Each model trains on a random half. Target = model 0; shadows = 1..n_shadow."""
    N = len(rec_pop["X"]); rng = np.random.default_rng(seed); masks = np.stack([rng.permutation(N) < N // 2 for _ in range(n_shadow + 1)])
    losses = []
    for k in range(n_shadow + 1):
        idx = np.flatnonzero(masks[k]); net = _train_local({kk: rec_pop[kk][idx] for kk in "XYMA"}, eps, epochs, seed * 100 + k, cfg, dev); losses.append(np.log(day_losses(net, rec_pop, dev) + 1e-8))
    losses = np.stack(losses); tgt, sh, msh = losses[0], losses[1:], masks[1:]; score = np.zeros(N)
    for i in range(N):
        li, lo = sh[msh[:, i], i], sh[~msh[:, i], i]
        if len(li) < 2 or len(lo) < 2: score[i] = 0.0; continue
        sd = max(0.5 * (li.std() + lo.std()), 0.1) ; score[i] = norm.logpdf(tgt[i], li.mean(), sd) - norm.logpdf(tgt[i], lo.mean(), sd)
    r = attack_metrics(score[masks[0]], score[~masks[0]]); r["n_shadow"] = n_shadow; return r
