"""Run controller grid cells in parallel with caching. Usage: 06_run_controllers.py --grid core|ablate|adv|all [--procs 10] [--sites ..] [--seeds 0-2]"""
import argparse, glob, hashlib, json, pickle, sys, time, yaml; from pathlib import Path
from multiprocessing import Pool
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
INF = float("inf"); G = yaml.safe_load((ROOT / "configs/grid_fast.yaml").read_text()); OUT = ROOT / "results/runs/control"; OUT.mkdir(parents=True, exist_ok=True)

def model_file(method, eps, seed):
    m = {"B2": "central", "B3": "local", "B4": "fedavg", "B5": "dpfedavg", "PALACE": "dpfedper"}[method]; e = f"{float(eps)}" if method in ("B5", "PALACE") else "inf"
    fs = sorted(glob.glob(str(ROOT / f"results/runs/models/fast_office_{m}_eps{e}_s{seed}_*.pt"))); return fs[-1] if fs else None

def cells(grid, sites, seeds, seasons):
    C = []
    for site in sites:
        for season in seasons:
            for seed in seeds:
                base = dict(site=site, season=season, seed=seed)
                if grid in ("core", "all"):
                    C += [dict(base, method="B0"), dict(base, method="B1")]
                    for m in ("B2", "B3", "B4"): C.append(dict(base, method=m, personalise=False))
                    for e in G["eps_train"]:
                        C.append(dict(base, method="B5", eps_train=e, personalise=False))
                        C.append(dict(base, method="PALACE", eps_train=e, eps_act=G["default"]["eps_act"], apf_on=True, personalise=True))
                if grid in ("ablate", "all"):
                    d = G["default"]
                    for ea in (0.5, 2): C.append(dict(base, method="PALACE", eps_train=d["eps_train"], eps_act=ea, apf_on=True, personalise=True))
                    C.append(dict(base, method="PALACE", eps_train=d["eps_train"], apf_on=False, personalise=True, tagx="apf_off"))
                    C.append(dict(base, method="PALACE", eps_train=d["eps_train"], eps_act=d["eps_act"], apf_on=True, personalise=False, tagx="no_pers"))
                    C.append(dict(base, method="PALACE", eps_train=d["eps_train"], eps_act=d["eps_act"], apf_on=True, personalise=True, perturb={"Rao": 1.3, "Ca": 1.3, "COP_ref": 0.7, "window_area": 1.3, "DF": 0.7}, tagx="sim_plus30"))
                    C.append(dict(base, method="PALACE", eps_train=d["eps_train"], eps_act=d["eps_act"], apf_on=True, personalise=True, perturb={"Rao": 0.7, "Ca": 0.7, "COP_ref": 1.3, "window_area": 0.7, "DF": 1.3}, tagx="sim_minus30"))
                if grid in ("adv", "all"):   # validation-split replay logs the adversary trains on
                    for c in [dict(base, method="B1"), dict(base, method="PALACE", eps_train=2, apf_on=False, personalise=True, tagx="apf_off")] + \
                             [dict(base, method="PALACE", eps_train=2, eps_act=ea, apf_on=True, personalise=True) for ea in (0.5, 2, 8)]: C.append(dict(c, split="val"))
    return C

def key(c): return hashlib.md5(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]

def work(c):
    f = OUT / f"{c['site']}_{c['season']}_{c['method']}_{key(c)}.pkl"
    if f.exists(): return str(f), 0.0
    from palace.eval.runner import run_cell
    t0 = time.time(); kw = {k: v for k, v in c.items() if k not in ("tagx",)}
    if c["method"] not in ("B0", "B1"): kw["model_file"] = model_file(c["method"], c.get("eps_train", INF), c["seed"])
    if c["method"] not in ("B0", "B1") and kw["model_file"] is None: return None, 0.0
    r = run_cell(ROOT, **kw); r["cell"] = c; pickle.dump(r, open(f, "wb")); return str(f), time.time() - t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="core"); ap.add_argument("--procs", type=int, default=10); ap.add_argument("--sites"); ap.add_argument("--seeds", default="0-2"); ap.add_argument("--seasons"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); lo, hi = a.seeds.split("-") if "-" in a.seeds else (a.seeds, a.seeds)
    C = cells(a.grid, a.sites.split(",") if a.sites else G["sites"], range(int(lo), int(hi) + 1), a.seasons.split(",") if a.seasons else G["seasons"])
    if a.limit: C = C[:a.limit]
    print(len(C), "cells", flush=True); t0 = time.time(); done = skipped = 0
    with Pool(a.procs) as p:
        for i, (f, dt) in enumerate(p.imap_unordered(work, C)):
            done += f is not None; skipped += f is None
            if i % 20 == 0: print(i, "/", len(C), round(time.time() - t0), "s", "missing-model" if f is None else "", flush=True)
    print("done", done, "skipped(no model)", skipped, round(time.time() - t0), "s")
