"""Attacks: (a) loss-threshold MIA on final FL models, (b) LiRA-lite (8 shadows, single-site DP proxy), (c) actuation adversary vs eps_act. Usage: 07_attacks.py [mia|lira|adv|all]"""
import glob, json, re, sys, yaml, pickle; from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.fl.data import build_records; from palace.models.occupancy_tcn import OccTCN
from palace.attacks import mia as M; from palace.attacks import actuation_adversary as AA; from palace.eval import aggregate as AG
cfg = yaml.safe_load((ROOT / "configs/fl.yaml").read_text()); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); what = sys.argv[1] if len(sys.argv) > 1 else "all"
OUT = ROOT / "results/tables"; OUT.mkdir(parents=True, exist_ok=True)

def cat(a, b): return {k: torch.cat([a[k], b[k]]) for k in "XYMA"}
if what in ("mia", "all"):
    SITES = ["ac1", "ac3", "ac4", "scone", "tokyo", "stair01", "stair02", "stair03", "stair04"]; recs = {}
    for s in SITES:
        tr = build_records(ROOT, s, "train", cfg["data"]["max_train_days_per_client"]); ho = cat(build_records(ROOT, s, "val", 100), build_records(ROOT, s, "test", 100)); recs[s] = (tr, ho)
    rows = []; roc = {}
    for f in sorted(glob.glob(str(ROOT / "results/runs/models/fast_office_*.pt"))):
        m = re.match(r"fast_office_(\w+?)_eps([\w.]+)_s(\d+)_", Path(f).name); method, eps, seed = m.group(1), m.group(2), int(m.group(3))
        if seed > 2: continue
        ck = torch.load(f, map_location="cpu")
        for s in SITES:
            if s not in ck: continue
            net = OccTCN(ch=cfg["model"]["channels"], k=cfg["model"]["kernel"], dil=tuple(cfg["model"]["dilations"])); net.load_state_dict(ck[s]["state"]); net.to(dev)
            r = M.loss_threshold_mia(net, recs[s][0], recs[s][1], dev, seed=seed); roc[(method, eps, seed, s)] = (r.pop("fpr"), r.pop("tpr")); rows.append(dict(method=method, eps=eps, seed=seed, site=s, **r))
    df = pd.DataFrame(rows); df.to_csv(OUT / "mia_loss_threshold.csv", index=False); pickle.dump(roc, open(ROOT / "results/runs/mia_roc.pkl", "wb")); print(df.groupby(["method", "eps"])[["auc", "tpr_at_1fpr"]].mean().round(3))
if what in ("lira", "all"):
    rows = []; roc = {}
    for site in ("scone", "ac1"):
        pop = cat(cat(build_records(ROOT, site, "train", 300), build_records(ROOT, site, "val", 100)), build_records(ROOT, site, "test", 100))
        for eps in (2.0, 8.0, float("inf")):
            r = M.lira_lite(pop, eps, cfg, dev, n_shadow=8, epochs=12, seed=0); roc[(site, eps)] = (r.pop("fpr"), r.pop("tpr")); rows.append(dict(site=site, eps=eps, **r)); print(site, eps, round(r["auc"], 3), round(r["tpr_at_1fpr"], 3), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "mia_lira_lite.csv", index=False); pickle.dump(roc, open(ROOT / "results/runs/lira_roc.pkl", "wb"))
if what in ("adv", "all"):
    te = AG.load_cells(ROOT, True, "test"); va = AG.load_cells(ROOT, True, "val"); rows = []; rocs = {}
    key = lambda r: (r["site"], r["label"])
    from collections import defaultdict
    T, V = defaultdict(list), defaultdict(list)
    for r in te: T[key(r)].append(r["series"])
    for r in va: V[key(r)].append(r["series"])
    for k in sorted(T):
        # label mapping: validation logs for the same controller config (B1; PALACE configs at eps_train=2)
        if k not in V: continue
        for name, ua, um in (("actuation+meter", True, True), ("meter_only", False, True), ("actuation_only", True, False)):
            r = AA.fit_eval(V[k], T[k], ua, um); rows.append(dict(site=k[0], config=k[1], adversary=name, **r))
    df = pd.DataFrame(rows); df.to_csv(OUT / "actuation_adversary.csv", index=False); print(df.groupby(["config", "adversary"])[["auroc", "leakage", "arrival_mae_min"]].mean().round(3))
