"""Tables T3-T6, paired stats (Holm), Pareto, sustainability accounting, figures F4-F12, paper_numbers.json."""
import json, pickle, sys, yaml, hashlib; from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.eval import aggregate as AG, stats as ST, pareto as PA; from palace.viz import style
style.apply(); TB = ROOT / "results/tables"; FG = ROOT / "results/figures"; TB.mkdir(exist_ok=True, parents=True)
base = yaml.safe_load((ROOT / "configs/base.yaml").read_text()); GF = base["carbon"]["grid_factor_kg_per_kwh"]; band = base["carbon"]["sensitivity_frac"]
chash = hashlib.md5(open(ROOT / "configs/control.yaml", "rb").read() + open(ROOT / "configs/fl.yaml", "rb").read()).hexdigest()[:8]
PN = {}
def put(key, value, lo=None, hi=None, n=None, src="", mode="fast"): PN[key] = dict(value=None if value is None or value != value else float(value), ci_low=lo, ci_high=hi, n_seeds=n, source_file=src, config_hash=chash, run_mode=mode)
df = AG.table(ROOT); occ = AG.occupancy_table(ROOT); print(len(df), "control cells;", len(occ), "occupancy rows")
K = ["site", "season", "seed"]
# ---------- T3 forecaster ----------
if len(occ):
    g = occ.groupby(["method", "eps"])
    t3 = g.agg(auroc=("auroc", "mean"), auprc=("auprc", "mean"), brier=("brier", "mean"), ece=("ece", "mean"), f1=("f1", "mean"), arrival_mae_min=("arrival_mae_min", "mean"), n_runs=("seed", "nunique")).reset_index()
    cis = {k: ST.boot_ci(v.groupby(["seed", "site"]).auroc.mean().values) for k, v in g}; t3["auroc_ci_low"] = [cis[(m, e)][1] for m, e in zip(t3.method, t3.eps)]; t3["auroc_ci_high"] = [cis[(m, e)][2] for m, e in zip(t3.method, t3.eps)]
    t3.to_csv(TB / "T3_forecaster.csv", index=False); open(TB / "T3_forecaster.tex", "w").write(t3.round(3).to_latex(index=False))
    for r in t3.itertuples(): put(f"forecast.{r.method}.eps{r.eps:g}.auroc", r.auroc, r.auroc_ci_low, r.auroc_ci_high, r.n_runs, "results/tables/T3_forecaster.csv")
    # baselines (naive) from the first central run
    import glob; bl = {}
    for f in glob.glob(str(ROOT / "results/runs/occupancy/fast_office_central_epsinf_s0_*.json")):
        d = json.load(open(f)); 
        for s, b in d.get("baselines", {}).items():
            for k, v in b.items(): bl.setdefault(k, []).append(v["auroc"])
    for k, v in bl.items(): put(f"forecast.baseline.{k}.auroc", np.mean(v), n=len(v), src="results/runs/occupancy/*central*s0*.json")
    # F4
    fig, ax = plt.subplots(1, 2, figsize=(style.COL2, 2.3))
    for j, met in enumerate(["auroc", "brier"]):
        for m, lab, col in [("central", "B2", "B2"), ("local", "B3", "B3"), ("fedavg", "B4", "B4"), ("dpfedavg", "B5", "B5"), ("dpfedper", "PALACE", "PALACE")]:
            d = t3[t3.method == m]
            if not len(d): continue
            if m in ("dpfedavg", "dpfedper"): d = d.sort_values("eps"); ax[j].plot(d.eps, d[met], "o-", color=style.METHOD_COLORS[col], label=lab, ms=3, lw=1)
            else: ax[j].axhline(float(d[met].iloc[0]), color=style.METHOD_COLORS[col], ls="--", lw=0.9, label=lab)
        ax[j].set_xscale("log"); ax[j].set_xlabel("Training privacy budget $\\varepsilon_{train}$ (dimensionless)"); ax[j].set_ylabel("AUROC (dimensionless)" if met == "auroc" else "Brier score (dimensionless)")
    ax[0].legend(frameon=False, fontsize=6); fig.tight_layout(); style.save(fig, FG / "F4_forecast_vs_eps")
    # T6 compute
    t6 = occ.groupby(["method", "eps"]).agg(wall_s=("wall", "mean"), gpu_mb=("gpu_mb", "mean"), params=("params", "first"), comm_mb=("comm_mb", "mean"), kwh=("kwh", "mean"), kg=("kg", "mean")).reset_index(); t6.to_csv(TB / "T6_compute.csv", index=False); open(TB / "T6_compute.tex", "w").write(t6.round(4).to_latex(index=False))
if not len(df): json.dump(PN, open(ROOT / "results/paper_numbers.json", "w"), indent=1); sys.exit(0)
# ---------- controller tables ----------
piv = lambda col: df.pivot_table(index=K, columns="label", values=col)
E, CS = piv("ctrl_kwh_m2_day"), piv("comfort_score"); TOT = piv("total_kwh_m2_day")
dflt = "PALACE_e2_a8"; rows = []
for lab in sorted(df.label.unique()):
    d = df[df.label == lab]; r = dict(label=lab, n=len(d))
    for col in ("ctrl_kwh_m2_day", "total_kwh_m2_day", "comfort_score", "d_th", "d_vis", "pmv_discomfort_h", "lux_lt300_frac", "peak_kw"):
        m, lo, hi = ST.boot_ci(d[col].values); r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi; put(f"ctrl.{lab}.{col}", m, lo, hi, d.seed.nunique(), "results/tables/T4_headline.csv")
    for ref in ("B0", "B1"):
        if ref in E and lab in E: s = ((E[ref] - E[lab]) / E[ref]).dropna(); m, lo, hi = ST.boot_ci(s.values); r[f"saving_vs_{ref}"], r[f"saving_vs_{ref}_lo"], r[f"saving_vs_{ref}_hi"] = m, lo, hi; put(f"ctrl.{lab}.ctrl_energy_saving_vs_{ref}", m, lo, hi, d.seed.nunique(), "T4_headline.csv")
    kg = d.total_kwh / d.days * 365 / d.area_m2 * GF; r["carbon_kg_m2_yr"] = kg.mean(); r["carbon_lo_-30%"], r["carbon_hi_+30%"] = kg.mean() * (1 - band), kg.mean() * (1 + band); rows.append(r)
t4 = pd.DataFrame(rows); t4.to_csv(TB / "T4_headline.csv", index=False); open(TB / "T4_headline.tex", "w").write(t4.round(3).to_latex(index=False))
# ---------- pre-registered paired tests ----------
tests = []
def pt(name, a, b, col, hyp):
    A, B = piv(col)[a], piv(col)[b]; r = ST.paired(A.values, B.values); r.update(name=name, metric=col, a=a, b=b, hypothesis=hyp); tests.append(r)
for lab in (dflt,):
    if lab in E and "B1" in E:
        pt("PALACE vs B1 energy", lab, "B1", "ctrl_kwh_m2_day", "H1"); pt("PALACE vs B1 comfort", lab, "B1", "comfort_score", "H2")
    if lab in E and "B2" in E:
        pt("PALACE vs B2 energy", lab, "B2", "ctrl_kwh_m2_day", "H3"); pt("PALACE vs B2 comfort", lab, "B2", "comfort_score", "H3")
adv = pd.read_csv(TB / "actuation_adversary.csv") if (TB / "actuation_adversary.csv").exists() else None
if adv is not None:
    a = adv[adv.adversary == "actuation+meter"].pivot_table(index="site", columns="config", values="leakage")
    if dflt in a and "PALACE_apf_off" in a: r = ST.paired(a[dflt].values, a["PALACE_apf_off"].values); r.update(name="APF on vs off leakage", metric="leakage", a=dflt, b="PALACE_apf_off", hypothesis="H4"); tests.append(r)
if tests:
    ad = ST.holm([t["p"] for t in tests])
    for t, p in zip(tests, ad): t["p_holm"] = p; put(f"test.{t['name']}", t["mean"], t["ci_low"], t["ci_high"], t["n"], "results/tables/prereg_tests.csv")
    pd.DataFrame(tests).to_csv(TB / "prereg_tests.csv", index=False); print(pd.DataFrame(tests)[["name", "mean", "ci_low", "ci_high", "cohen_dz", "p", "p_holm"]].round(4))
# ---------- adversary / MIA tables (T5) ----------
t5 = []
if adv is not None:
    for (c, ad_), d in adv.groupby(["config", "adversary"]): t5.append(dict(kind="actuation_adversary", config=c, adversary=ad_, auroc=d.auroc.mean(), leakage=d.leakage.mean(), arrival_mae_min=d.arrival_mae_min.mean())); put(f"attack.actuation.{c}.{ad_}.auroc", d.auroc.mean(), *ST.boot_ci(d.auroc.values)[1:], n=len(d), src="results/tables/actuation_adversary.csv")
mi = pd.read_csv(TB / "mia_loss_threshold.csv") if (TB / "mia_loss_threshold.csv").exists() else None
if mi is not None:
    for (m, e), d in mi.groupby(["method", "eps"]): t5.append(dict(kind="mia_loss_threshold", config=f"{m}_eps{e}", auroc=d.auc.mean(), tpr_at_1fpr=d.tpr_at_1fpr.mean())); put(f"attack.mia.{m}.eps{e}.auc", d.auc.mean(), *ST.boot_ci(d.auc.values)[1:], n=d.seed.nunique(), src="results/tables/mia_loss_threshold.csv")
if (TB / "mia_lira_lite.csv").exists():
    for r in pd.read_csv(TB / "mia_lira_lite.csv").itertuples(): t5.append(dict(kind="lira_lite_proxy", config=f"{r.site}_eps{r.eps:g}", auroc=r.auc, tpr_at_1fpr=r.tpr_at_1fpr)); put(f"attack.lira.{r.site}.eps{r.eps:g}.auc", r.auc, src="results/tables/mia_lira_lite.csv")
if t5: pd.DataFrame(t5).to_csv(TB / "T5_privacy_attacks.csv", index=False); open(TB / "T5_privacy_attacks.tex", "w").write(pd.DataFrame(t5).round(3).to_latex(index=False))
# ---------- Pareto (F5, F6) ----------
agg = df.groupby("label").agg(ctrl=("ctrl_kwh_m2_day", "mean"), comfort=("comfort_score", "mean"), total=("total_kwh_m2_day", "mean")).reset_index(); agg["discomfort"] = 1 - agg.comfort; agg["carbon_g_m2_day"] = agg.total * GF * 1000
if adv is not None:
    lk = adv[adv.adversary == "actuation+meter"].groupby("config").leakage.mean(); agg["leakage"] = agg.label.map(lk)
else: agg["leakage"] = np.nan
fig, ax = plt.subplots(figsize=(style.COL1, 2.6)); sc = ax.scatter(agg.carbon_g_m2_day, agg.discomfort, c=agg.leakage.fillna(-1), cmap="viridis", s=18, vmin=0, vmax=max(0.01, np.nanmax(agg.leakage.values) if agg.leakage.notna().any() else 1))
for r in agg.itertuples(): ax.annotate(r.label.replace("PALACE_", "P_"), (r.carbon_g_m2_day, r.discomfort), fontsize=4.5, xytext=(2, 2), textcoords="offset points")
ax.set_xlabel("Carbon emissions (g CO$_2$e per m$^2$ per day)"); ax.set_ylabel("Occupant discomfort (dimensionless)"); cb = fig.colorbar(sc); cb.set_label("Actuation leakage $L$ (dimensionless)"); fig.tight_layout(); style.save(fig, FG / "F5_pareto_energy_comfort")
P = agg.dropna(subset=["leakage"])
if len(P) >= 2:
    P3 = P[["carbon_g_m2_day", "discomfort", "leakage"]].values; ref = P3.max(0) * 1.1 + 1e-9; pm = PA.pareto_mask(P3); put("pareto.n_nondominated", pm.sum(), src="results/tables/pareto.csv"); put("pareto.hypervolume_all_configs", PA.hypervolume(P3, ref), src="results/tables/pareto.csv")
    P.assign(pareto=pm).to_csv(TB / "pareto.csv", index=False)
agg.to_csv(TB / "pareto_points.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(style.COL2, 2.3)); p = agg[agg.label.str.startswith("PALACE_e2_a")].copy(); p["ea"] = p.label.str.extract(r"_a([\d.]+)").astype(float)
ax[0].plot(p.sort_values("ea").ea, p.sort_values("ea").leakage, "o-", ms=3, color=style.METHOD_COLORS["PALACE"]); ax[0].set_xscale("log"); ax[0].set_xlabel("Actuation privacy budget $\\varepsilon_{act}$ (dimensionless)"); ax[0].set_ylabel("Leakage $L$ (dimensionless)")
b1 = agg[agg.label == "B1"]; ax[1].scatter(((float(b1.ctrl.iloc[0]) - agg.ctrl) / float(b1.ctrl.iloc[0]) * 100) if len(b1) else agg.ctrl, agg.leakage, s=14, c=[style.METHOD_COLORS.get(l.split("_")[0], "k") for l in agg.label]); ax[1].set_xlabel("Lighting+HVAC energy saving vs B1 (%)"); ax[1].set_ylabel("Leakage $L$ (dimensionless)"); fig.tight_layout(); style.save(fig, FG / "F6_pareto_privacy")
# ---------- F8 ablations ----------
ab = [l for l in ["PALACE_e2_a8", "PALACE_e2_a2", "PALACE_e2_a0.5", "PALACE_apf_off", "PALACE_no_pers"] if l in E]
if ab and "B1" in E:
    fig, ax = plt.subplots(1, 2, figsize=(style.COL2, 2.3)); 
    for j, (M_, yl) in enumerate([(E, "Lighting+HVAC energy (kWh/m$^2$/day)"), (CS, "Comfort score (dimensionless)")]):
        ms = [ST.boot_ci(M_[l].dropna().values) for l in ab]; ax[j].bar(range(len(ab)), [m[0] for m in ms], yerr=[[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]], color=style.METHOD_COLORS["PALACE"], capsize=2); ax[j].set_xticks(range(len(ab))); ax[j].set_xticklabels([l.replace("PALACE_", "") for l in ab], rotation=45, ha="right"); ax[j].set_ylabel(yl)
    fig.tight_layout(); style.save(fig, FG / "F8_ablations")
# ---------- F12 robustness ----------
rb = [l for l in ["PALACE_sim_minus30", dflt, "PALACE_sim_plus30"] if l in E]
if len(rb) == 3:
    fig, ax = plt.subplots(1, 2, figsize=(style.COL2, 2.2))
    for j, (M_, yl) in enumerate([(E, "Lighting+HVAC energy (kWh/m$^2$/day)"), (CS, "Comfort score (dimensionless)")]):
        ms = [ST.boot_ci(M_[l].dropna().values) for l in rb]; ax[j].errorbar(["-30 %", "nominal", "+30 %"], [m[0] for m in ms], yerr=[[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]], fmt="o-", ms=3, capsize=2); ax[j].set_xlabel("Simulator perturbation (parameters scaled)"); ax[j].set_ylabel(yl)
    fig.tight_layout(); style.save(fig, FG / "F12_robustness")
# ---------- F9 seasonal time series (series from one cell per season) ----------
cells = AG.load_cells(ROOT, True); site0 = "ac1"
fig, ax = plt.subplots(3, 3, figsize=(style.COL2, 4.6), sharex=True)
for j, season in enumerate(["summer", "monsoon", "winter"]):
    r = next((c for c in cells if c["site"] == site0 and c["season"] == season and c["label"] == dflt and c["seed"] == 0), None)
    if r is None: continue
    s = r["series"]; x = np.arange(96 * 2) / 4; sl = slice(96, 96 * 3)
    ax[0, j].plot(x, s["lux"][sl], lw=0.7); ax[0, j].set_ylabel("Workplane illuminance (lx)"); ax[1, j].plot(x, s["sp"][sl], lw=0.7, label="setpoint"); ax[1, j].plot(x, s["Ta"][sl], lw=0.7, label="air"); ax[1, j].set_ylabel("Temperature ($^\\circ$C)")
    ax[2, j].plot(x, s["kw"][sl], lw=0.7); ax[2, j].fill_between(x, 0, s["presence"][sl] * s["kw"][sl].max(), alpha=0.2, color="gray", step="post"); ax[2, j].set_xlabel("Hour from start of day 2 (h)"); ax[2, j].set_ylabel("Electric power (kW)")
ax[1, 0].legend(frameon=False, fontsize=6); fig.tight_layout(); style.save(fig, FG / "F9_seasonal_timeseries")
# ---------- F11 net sustainability ----------
if dflt in E and "B1" in E and len(occ):
    sav_kwh_day = float(((E["B1"] - E[dflt]) * 25.0).dropna().mean()); sav_ci = ST.boot_ci(((E["B1"] - E[dflt]) * 25.0).dropna().values)
    tr = occ[(occ.method == "dpfedper") & (occ.eps == 2.0)]; tr_kwh = float(tr.groupby("seed").kwh.first().mean()) if tr.kwh.notna().any() else np.nan
    comm_mb = float(tr.comm_mb.mean()); NET_KWH_PER_GB = 0.06  # ASSUMED upper-range network energy intensity; UNVERIFIED (to verify before citing)
    comm_kwh = comm_mb / 1024 * NET_KWH_PER_GB; one_off = (0 if tr_kwh != tr_kwh else tr_kwh) + comm_kwh
    put("sustain.saving_kwh_per_zone_day_vs_B1", sav_kwh_day, sav_ci[1], sav_ci[2], 3, "results/tables/T4_headline.csv"); put("sustain.training_kwh_one_off", tr_kwh, src="results/tables/T6_compute.csv"); put("sustain.comm_kwh_one_off_ASSUMED", comm_kwh, src="assumption 0.06 kWh/GB (unverified)")
    fig, ax = plt.subplots(figsize=(style.COL1, 2.3)); days = np.arange(0, 366)
    for lab, tag in [(dflt, "PALACE (APF on, eps_act=8)"), ("PALACE_apf_off", "PALACE (APF off)")]:
        if lab not in E: continue
        d_ = ((E["B1"] - E[lab]) * 25.0).dropna(); sv = float(d_.mean()); ci = ST.boot_ci(d_.values)
        put(f"sustain.{lab}.saving_kwh_per_zone_day_vs_B1", sv, ci[1], ci[2], 3, "results/tables/T4_headline.csv"); put(f"sustain.{lab}.carbon_avoided_kg_per_zone_year", sv * 365 * GF, ci[1] * 365 * GF * (1 - band if ci[1] > 0 else 1 + band), ci[2] * 365 * GF * (1 + band if ci[2] > 0 else 1 - band), 3, "grid factor unverified; +-30% band")
        for nz, ls in [(4, "--"), (40, "-")]:
            ax.plot(days, sv * nz * days - one_off, ls, label=f"{tag}, {nz} zones", lw=1, color=style.METHOD_COLORS["PALACE"] if "off" not in tag else style.METHOD_COLORS["B2"])
            if sv > 0: put(f"sustain.{lab}.payback_days.{nz}zones", one_off / (sv * nz), src="F11")
    put("sustain.training_kwh_one_off", tr_kwh, src="results/tables/T6_compute.csv"); put("sustain.comm_kwh_one_off_ASSUMED", comm_kwh, src="assumption 0.06 kWh/GB (unverified)")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("Days of deployment (days)"); ax.set_ylabel("Net electricity saved vs B1 (kWh)"); ax.legend(frameon=False, fontsize=5); fig.tight_layout(); style.save(fig, FG / "F11_net_sustainability")
# ---------- F10 ROC ----------
fig, ax = plt.subplots(1, 2, figsize=(style.COL2, 2.3))
if (ROOT / "results/runs/mia_roc.pkl").exists():
    roc = pickle.load(open(ROOT / "results/runs/mia_roc.pkl", "rb"))
    for (m, e), c in [(("dpfedper", "8.0"), "PALACE"), (("dpfedper", "2.0"), "B2"), (("dpfedper", "0.5"), "B3"), (("central", "inf"), "B1")]:
        f = [v for k, v in roc.items() if k[0] == m and k[1] == e and k[2] == 0]
        if f: fp = np.linspace(0, 1, 101); ax[0].plot(fp, np.mean([np.interp(fp, a, b) for a, b in f], 0), label=f"{m} $\\varepsilon$={e}", lw=1)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.6); ax[0].set_xlabel("False positive rate (dimensionless)"); ax[0].set_ylabel("True positive rate (dimensionless)"); ax[0].legend(frameon=False, fontsize=5)
ax[1].set_xlabel("False positive rate (dimensionless)"); ax[1].set_ylabel("True positive rate (dimensionless)"); ax[1].plot([0, 1], [0, 1], "k:", lw=0.6)
fig.tight_layout(); style.save(fig, FG / "F10_attacks_roc")
# ---------- F7 ladder ----------
if (TB / "granularity_ladder.csv").exists():
    L = pd.read_csv(TB / "granularity_ladder.csv").groupby("rung").agg(auroc=("auroc", "mean"), mia=("mia_auc", "mean")).reset_index(); fig, ax = plt.subplots(figsize=(style.COL1, 2.4)); ax.scatter(L.mia, L.auroc, c=style.OKABE_ITO[:len(L)], s=20)
    for r in L.itertuples(): ax.annotate(r.rung, (r.mia, r.auroc), fontsize=6, xytext=(3, 3), textcoords="offset points"); put(f"ladder.{r.rung}.auroc", r.auroc, src="results/tables/granularity_ladder.csv"); put(f"ladder.{r.rung}.mia_auc", r.mia, src="results/tables/granularity_ladder.csv")
    ax.set_xlabel("Membership-inference AUC (dimensionless)"); ax.set_ylabel("Forecast AUROC (dimensionless)"); fig.tight_layout(); style.save(fig, FG / "F7_granularity_ladder")
json.dump(PN, open(ROOT / "results/paper_numbers.json", "w"), indent=1); print("paper_numbers keys:", len(PN))
