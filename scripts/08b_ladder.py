"""Granularity ladder (reduced): central non-DP forecaster trained on information rungs, evaluated for forecast quality (AUROC) and MIA exposure.
R1 raw features (presence, motion, active-sensor count, door events)  | R2 aggregate counts only (presence + motion count)
R3 R2 + local noise: Laplace(1/eps) on motion count, randomised response on presence bit, PER BIN (not a day-level guarantee)  | R4 k-anonymous zones: NOT RUN (office sites expose 1 zone sensor; see limitations)."""
import sys, yaml, json; from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.fl.data import build_records; from palace.attacks import mia as M; from palace.fl.evaluate import predict_logits, fit_temperature
from palace.models.metrics import binary_report
cfg = yaml.safe_load((ROOT / "configs/fl.yaml").read_text()); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SITES = ["ac1", "ac3", "ac4", "scone", "tokyo", "stair01", "stair02", "stair03", "stair04"]
def transform(X, rung, eps, rng):
    X = X.clone()
    if rung in ("R2", "R3"): X[..., 2] = 0.0; X[..., 3] = 0.0                       # drop sensor-identity-level channels
    if rung == "R3":
        obs = (X[..., 8] > 0).float(); raw = torch.expm1(X[..., 1]).numpy(); noisy = np.maximum(0, raw + rng.laplace(0, 1 / eps, raw.shape)); X[..., 1] = torch.from_numpy(np.log1p(noisy)).float() * obs
        flip = torch.from_numpy(rng.random(X[..., 0].shape) < 1 / (1 + np.exp(eps))); X[..., 0] = torch.where(flip, 1 - X[..., 0], X[..., 0]) * (X[..., 8] > 0).float()
    return X
rows = []; recs = {s: (build_records(ROOT, s, "train", 300), build_records(ROOT, s, "val", 100), build_records(ROOT, s, "test", 100)) for s in SITES}
for rung, eps in [("R1", None), ("R2", None), ("R3", 4.0), ("R3", 1.0)]:
    for seed in range(3):
        rng = np.random.default_rng(seed); tf = lambda r: {**r, "X": transform(r["X"], rung, eps, rng)}
        T = {s: tf(recs[s][0]) for s in SITES}; V = {s: tf(recs[s][1]) for s in SITES}; Te = {s: tf(recs[s][2]) for s in SITES}
        pool = {k: torch.cat([T[s][k] for s in SITES]) for k in "XYMA"}; net = M._train_local(pool, float("inf"), 12, seed, cfg, dev, batch_days=16).to(dev)
        au, mi = [], []
        for s in SITES:
            if len(Te[s]["X"]) < 10: continue
            lv, _ = predict_logits(net, V[s], dev); lt, _ = predict_logits(net, Te[s], dev); Tm = fit_temperature(lv, V[s]["Y"].numpy(), V[s]["M"].numpy())
            p = 1 / (1 + np.exp(-lt / Tm)); au.append(binary_report(p[Te[s]["M"].numpy() > 0], Te[s]["Y"].numpy()[Te[s]["M"].numpy() > 0])["auroc"])
            ho = {k: torch.cat([V[s][k], Te[s][k]]) for k in "XYMA"}; mi.append(M.loss_threshold_mia(net, T[s], ho, dev, seed=seed)["auc"])
        rows.append(dict(rung=rung + (f"_eps{eps:g}" if eps else ""), seed=seed, auroc=float(np.mean(au)), mia_auc=float(np.mean(mi)))); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/tables/granularity_ladder.csv", index=False)
