"""SYNTHETIC-ground-truth experiment: posterior error vs #votes and eps_fb; misspecification (heavy-tailed truth, Gaussian learner)."""
import json, sys; from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.prefs.feedback_rr import simulate
rows = []
for kind in ("thermal", "lighting"):
    for eps in (0.5, 1, 2, 4, np.inf):
        for nv in (0, 5, 10, 20, 50, 100):
            for heavy in (False, True):
                for seed in range(5):
                    rmse, mae = simulate(kind, eps, nv, 100, seed, heavy_truth=heavy)
                    rows.append(dict(kind=kind, eps_fb=eps, n_votes=nv, misspecified=heavy, seed=seed, rmse=rmse, mae=mae))
df = pd.DataFrame(rows); df.to_csv(ROOT / "results/tables/feedback_rr_posterior.csv", index=False)
g = df.groupby(["kind", "misspecified", "eps_fb", "n_votes"]).rmse.agg(["mean", "std"]).reset_index(); g.to_csv(ROOT / "results/tables/feedback_rr_summary.csv", index=False)
pd.set_option("display.width", 200)
for k, unit in (("thermal", "degC"), ("lighting", "ln lux")):
    print(k, unit); print(g[(g.kind == k) & (~g.misspecified)].pivot(index="eps_fb", columns="n_votes", values="mean").round(3))
    print(k, "MISSPECIFIED"); print(g[(g.kind == k) & (g.misspecified)].pivot(index="eps_fb", columns="n_votes", values="mean").round(3))
