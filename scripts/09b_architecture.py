import sys; from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import FancyBboxPatch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); from palace.viz import style; style.apply()
fig, ax = plt.subplots(figsize=(style.COL2, 2.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
def box(x, y, w, h, t, c="#DDEBF7"): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="k", lw=0.6)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=0.7))
box(0.1, 4.2, 2.1, 1.0, "CASAS PIR/door\n(site-local, never leaves)"); box(2.8, 4.2, 2.3, 1.0, "Site-day features\n(15 min, outage mask)"); box(5.7, 4.2, 2.1, 1.0, "DP-FedPer TCN\n(day-level DP-SGD)", "#FCE4D6"); box(8.2, 4.2, 1.7, 1.0, "Occupancy\nforecast p(t..t+2h)")
box(0.1, 2.5, 2.1, 1.0, "ASHRAE DB II\nthermal preference", "#E2EFDA"); box(2.8, 2.5, 2.3, 1.0, "Preference posterior\n(RR feedback, synthetic)", "#E2EFDA"); box(5.7, 2.5, 2.1, 1.0, "CEM-MPC\n(2R2C zone model)"); box(8.2, 2.5, 1.7, 1.0, "Exponential-\nmechanism APF", "#FCE4D6")
box(5.7, 0.7, 2.1, 1.0, "Released logs:\ndim, CCT, setpoint, kW", "#FFF2CC"); box(8.2, 0.7, 1.7, 1.0, "Adversary\n(presence/arrival)", "#F8CBAD"); box(0.1, 0.7, 2.1, 1.0, "Weather + solar\n(Open-Meteo, pvlib)", "#E2EFDA"); box(2.8, 0.7, 2.3, 1.0, "Zone simulator\n(AC, LED, comfort)")
for a in [(2.2, 4.7, 2.8, 4.7), (5.1, 4.7, 5.7, 4.7), (7.8, 4.7, 8.2, 4.7), (9.05, 4.2, 9.05, 3.5), (2.2, 3.0, 2.8, 3.0), (5.1, 3.0, 5.7, 3.0), (7.8, 3.0, 8.2, 3.0), (6.75, 2.5, 6.75, 1.7), (8.2, 2.7, 7.8, 1.5), (7.8, 1.2, 8.2, 1.2), (2.2, 1.2, 2.8, 1.2), (5.1, 1.5, 5.8, 2.5)]: arr(*a)
fig.tight_layout(); style.save(fig, ROOT / "results/figures/F1_architecture")
