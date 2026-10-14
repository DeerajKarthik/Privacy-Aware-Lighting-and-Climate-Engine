"""Run B0/B1 over a replayed test-free (train-split) year for each office site; compare EUI & weekday load shape to BDG2 offices."""
import json, sys, time, yaml; from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.sim.year import build_inputs; from palace.sim.zone_env import Zone; from palace.control.baselines import b0, b1
from palace.data.bdg2 import office_stats; from palace.viz import style
style.apply(); cfg = yaml.safe_load((ROOT / "configs/sim.yaml").read_text())
SITES = ["ac1", "ac3", "ac4", "scone", "tokyo", "museum", "stair01", "stair02", "stair03", "stair04"]
bd = office_stats(ROOT / "data/raw/bdg2"); p = bd["eui_pcts"]; lo, hi = p[0.1], p[0.9]
rows = []; shapes = {}
for s in SITES:
    inp = build_inputs(ROOT, cfg, s, "surat", 2019, "train", seed=0); z = Zone(cfg, inp); T = len(inp["To"]); idx = inp["index"]
    for m in ("B0", "B1"):
        dim, cct, sp = b0(idx, cfg) if m == "B0" else b1(inp["presence"], z.lux_day(np.arange(T)), cfg)
        t0 = time.time(); o = z.rollout(np.array([[27.0, 27.0]]), dim[None], cct[None], sp[None], 0); dt = time.time() - t0
        k = {n: o[n][0].sum() * 0.25 / 1000 / z.area for n in ("P_light", "P_hvac", "P_equip")}
        tot = sum(k.values()); P = (o["P_light"] + o["P_hvac"] + o["P_equip"])[0] / z.area
        wk = pd.Series(P, index=idx); wk = wk[wk.index.dayofweek < 5].groupby(wk[wk.index.dayofweek < 5].index.hour).mean(); shapes[(s, m)] = wk.values
        rows.append(dict(site=s, method=m, light_kwh_m2=k["P_light"], hvac_kwh_m2=k["P_hvac"], equip_kwh_m2=k["P_equip"], total_kwh_m2=tot, in_bdg2_p10_p90=bool(lo <= tot <= hi),
                         ta_max=float(o["Ta"].max()), rollout_s=dt))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); out = ROOT / "results/tables"; out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "sim_calibration.csv", index=False)
json.dump({"bdg2_n_office": bd["n_buildings"], "bdg2_eui_pcts": {str(k): v for k, v in p.items()}, "weather": "surat 2019 (replayed)", "n_in_band": int(df.in_bdg2_p10_p90.sum()), "n_total": len(df)}, open(ROOT / "results/sim_calibration_summary.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(style.COL2, 2.4)); h = np.arange(24)
ax[0].fill_between(h, bd["shape_pcts"][.1], bd["shape_pcts"][.9], color="0.85", label="BDG2 offices 10th-90th pct"); ax[0].plot(h, bd["shape_pcts"][.5], "k--", lw=0.8, label="BDG2 median")
for (s, m), v in shapes.items():
    if s in ("ac1", "tokyo", "scone") and m == "B1": ax[0].plot(h, v, lw=0.9, label=f"simulated {s} (B1)")
ax[0].set_xlabel("Hour of day (h)"); ax[0].set_ylabel("Weekday electric load (W/m$^2$)"); ax[0].legend(frameon=False, fontsize=6)
b = df[df.method == "B1"]; ax[1].axhspan(lo, hi, color="0.85", label="BDG2 10th-90th pct"); ax[1].bar(b.site, b.total_kwh_m2, color=style.OKABE_ITO[0]); ax[1].axhline(p[0.5], color="k", ls="--", lw=0.8)
ax[1].set_ylabel("Annual electricity (kWh/m$^2$)"); ax[1].tick_params(axis="x", rotation=60); ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout(); style.save(fig, ROOT / "results/figures/F3_sim_validation"); (ROOT / "results/figures/sim_validation.pdf").write_bytes((ROOT / "results/figures/F3_sim_validation.pdf").read_bytes())
