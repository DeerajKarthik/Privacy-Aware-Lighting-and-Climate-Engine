"""Run forecaster experiments (cached per run). Usage: 04_train_occupancy.py --group office --methods central,local,fedavg,dpfedavg,dpfedper --eps 0.5,1,2,4,8 --seeds 0-4 [--rounds 15] [--sites a,b]"""
import argparse, json, sys, time, yaml, hashlib; from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.fl.engine import run_fl, DEV
from palace.fl.evaluate import eval_site, baseline_reports
from palace.eval.carbon import track
ap = argparse.ArgumentParser(); ap.add_argument("--group", default="office"); ap.add_argument("--methods", default="central,local,fedavg,dpfedavg,dpfedper")
ap.add_argument("--eps", default="0.5,1,2,4,8"); ap.add_argument("--seeds", default="0-4"); ap.add_argument("--rounds", type=int); ap.add_argument("--sites"); ap.add_argument("--tag", default="fast")
ap.add_argument("--C", type=float); ap.add_argument("--no_carbon", action="store_true"); a = ap.parse_args()
cfg = yaml.safe_load((ROOT / "configs/fl.yaml").read_text())
if a.C: cfg["train"]["max_grad_norm"] = a.C
sites = a.sites.split(",") if a.sites else cfg["clients"][a.group]
lo, hi = (a.seeds.split("-") + [a.seeds])[:2] if "-" in a.seeds else (a.seeds, a.seeds); seeds = range(int(lo), int(hi) + 1)
chash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]; out = ROOT / "results/runs/occupancy"; out.mkdir(parents=True, exist_ok=True)
for method in a.methods.split(","):
    epss = [float(e) for e in a.eps.split(",")] if method.startswith("dp") else [float("inf")]
    for eps in epss:
        for seed in seeds:
            f = out / f"{a.tag}_{a.group}_{method}_eps{eps}_s{seed}_{chash}.json"
            if f.exists(): continue
            with track(f"{method}_{eps}_{seed}", enabled=not a.no_carbon) as cb:
                res = run_fl(sites, cfg, ROOT, method, eps, seed, a.rounds)
            per = {}
            for s, c in res["clients"].items():
                if len(c.test["X"]) >= cfg["data"]["min_eval_test_days"]:
                    r, _ = eval_site(res["models"][s], c, DEV); per[s] = r
            import torch; md = ROOT / "results/runs/models"; md.mkdir(parents=True, exist_ok=True)
            torch.save({s_: {"state": {k: v.detach().cpu() for k, v in res["models"][s_].state_dict().items()}, "T": per.get(s_, {}).get("T", 1.0)} for s_ in res["models"]}, md / (f.stem + ".pt"))
            rec = dict(method=method, eps=eps, seed=seed, group=a.group, config_hash=chash, tag=a.tag, sites=sites, stats=res["stats"], carbon=cb, per_site=per)
            if method == "central" and seed == seeds[0]:
                rec["baselines"] = {s: baseline_reports(c) for s, c in res["clients"].items() if len(c.test["X"]) >= cfg["data"]["min_eval_test_days"]}
            json.dump(rec, open(f, "w"), indent=1, default=lambda o: None if o != o else str(o))
            m = np.nanmean([r["auroc"] for r in per.values()]); print(method, eps, seed, "AUROC", round(float(m), 4), "wall", round(cb["wall_s"], 1), "eps_spent", {k: round(v, 2) for k, v in list(res["stats"].get("eps_spent", {}).items())[:2]}, flush=True)
