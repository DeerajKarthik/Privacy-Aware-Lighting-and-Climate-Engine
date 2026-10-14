from __future__ import annotations
import glob, pickle, json
from pathlib import Path
import numpy as np, pandas as pd

def label(c):
    m = c["method"]; tx = c.get("tagx")
    if m in ("B0", "B1", "B2", "B3", "B4"): return m
    if m == "B5": return f"B5_e{c['eps_train']:g}"
    if tx: return f"PALACE_{tx}"
    return f"PALACE_e{c['eps_train']:g}_a{c.get('eps_act', float('inf')):g}"

def load_cells(root: Path, with_series=False, split="test"):
    rows = []
    for f in glob.glob(str(root / "results/runs/control/*.pkl")):
        r = pickle.load(open(f, "rb")); c = r["cell"]
        if c.get("split", "test") != split: continue
        s = r.pop("series", None); r["label"] = label(c); r["file"] = f
        if with_series: r["series"] = s
        rows.append(r)
    return rows

def table(root: Path, split="test"):
    rows = load_cells(root, False, split); df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("cell", "perturb")} for r in rows])
    if len(df):
        df["ctrl_kwh_m2_day"] = df.ctrl_kwh / df.area_m2 / df.days; df["total_kwh_m2_day"] = df.total_kwh / df.area_m2 / df.days
    return df

def occupancy_table(root: Path, tag="fast"):
    rows = []
    for f in glob.glob(str(root / f"results/runs/occupancy/{tag}_office_*.json")):
        d = json.load(open(f))
        for s, r in d["per_site"].items(): rows.append(dict(method=d["method"], eps=d["eps"], seed=d["seed"], site=s, **{k: r[k] for k in ("auroc", "auprc", "brier", "ece", "f1", "arrival_mae_min", "n_days_test")},
                                                            eps_spent=(d["stats"].get("eps_spent", {}) or {}).get(s), kwh=d["carbon"].get("kwh"), kg=d["carbon"].get("kg_co2e"), wall=d["stats"]["train_wall_s"], gpu_mb=d["stats"]["gpu_peak_mb"], comm_mb=d["stats"]["comm_mb"], params=d["stats"]["params"]))
    return pd.DataFrame(rows)
