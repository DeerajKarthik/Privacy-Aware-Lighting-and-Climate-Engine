import sys, time; from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.viz import style; style.apply()
S = pd.read_csv(ROOT / "data/processed/site_summary.csv"); sites = list(S.site)
fig, ax = plt.subplots(len(sites), 1, figsize=(style.COL2, 0.55 * len(sites) + 0.6), sharex=True); t_load = []
for a, s in zip(ax, sites):
    t0 = time.time(); d = pd.read_parquet(ROOT / f"data/processed/sites/{s}.parquet", columns=["presence", "mask"]); t_load.append(time.time() - t0)
    v = d[d["mask"] == 1]; h = v.groupby(v.index.hour + v.index.minute // 15 / 4)["presence"].mean()
    a.imshow(h.values[None, :], aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=[0, 24, 0, 1]); a.set_yticks([]); a.set_ylabel(s, rotation=0, ha="right", va="center")
ax[-1].set_xlabel("Hour of day (h)"); fig.tight_layout(); style.save(fig, ROOT / "results/figures/eda_presence_heatmaps")
print("max parquet load s:", round(max(t_load), 2))
