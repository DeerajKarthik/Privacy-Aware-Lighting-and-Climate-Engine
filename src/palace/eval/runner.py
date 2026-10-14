"""Closed-loop evaluation of one (site, season, method, seed) cell. TRUE zone is simulated with real (replayed) occupancy; the controller sees only
forecaster probabilities (from origin t-1), live PIR presence y_t (only via the step-0 comfort weight / baselines), noisy weather forecast, and RR-privatised votes.
HIDDEN SYNTHETIC occupant (delta_u, L*_u, c*_u) defines true discomfort. SYNTHETIC; see ETHICS/limitations."""
from __future__ import annotations
import copy, joblib, yaml, numpy as np, pandas as pd, torch
from pathlib import Path
from palace.sim.zone_env import Zone, perturb_cfg
from palace.sim.year import build_inputs, load_site, replay_days
from palace.sim import lighting
from palace.sim.comfort import pmv
from palace.control import baselines
from palace.control.costs import CostModel, decode, ALL_A
from palace.control.cem_mpc import Planner
from palace.control import apf as APF
from palace.prefs.feedback_rr import Posterior, true_votes, sample_users
from palace.prefs.lighting_pref import sample_cct_params, cct_pref
from palace.dp.rr import randomise
from palace.fl.data import build_records
from palace.models.occupancy_tcn import OccTCN

torch.set_num_threads(1)
_PNC = {}

def pnc_table(root):
    if "t" not in _PNC:
        m = joblib.load(root / "results/thermal_pref_lgbm.joblib"); g = np.arange(16, 40.01, 0.25); _PNC["t"] = (g, m.p_no_change(g))
    return _PNC["t"]

def forecasts(root, fl_cfg, model_file, site, src_days):
    """p_hat[day, b, h]: probability that bin b+1+h-1... i.e. for control step b: forecast (from origin b-1) of presence at bins b..b+7. Uses temperature T from validation."""
    ck = torch.load(model_file, map_location="cpu"); e = ck[site]; m = fl_cfg["model"]
    net = OccTCN(ch=m["channels"], k=m["kernel"], dil=tuple(m["dilations"])); net.load_state_dict(e["state"]); net.eval()
    rec = build_records(root, site, "test", days_list=src_days)
    with torch.no_grad(): lg = net.forward_all(rec["X"]).numpy()
    return 1 / (1 + np.exp(-lg / e["T"]))

def e_step_max(cfg):
    area = cfg["zone"]["floor_area_m2"]; p_l = area * cfg["lighting"]["pmax_w_per_m2"] * 1.15 * 1.05; p_h = cfg["ac"]["Qmax_W"] / cfg["ac"]["cop_floor"] + cfg["ac"]["fan_standby_W"]
    return (p_l + p_h) * cfg["dt_s"] / 3.6e6

def hidden_user(seed, site):
    import zlib; rng = np.random.default_rng(zlib.crc32(f'{site}-{seed}'.encode())); d, l = sample_users(1, rng); a, b = sample_cct_params(1, rng)
    return dict(delta=float(d[0]), lux=float(np.exp(l[0])), a=float(a[0]), b=float(b[0]))

def run_cell(root, site, season, method, seed=0, eps_act=float("inf"), eps_train=float("inf"), model_file=None, personalise=False, apf_on=False,
             perturb=None, climate="surat", granularity="R2", tag="fast", split="test", sim_cfg=None, ctrl_cfg=None, fl_cfg=None, keep_series=True):
    root = Path(root); sim_cfg = sim_cfg or yaml.safe_load((root / "configs/sim.yaml").read_text()); ctrl = ctrl_cfg or yaml.safe_load((root / "configs/control.yaml").read_text())
    fl_cfg = fl_cfg or yaml.safe_load((root / "configs/fl.yaml").read_text()); cfg = perturb_cfg(sim_cfg, perturb); rng = np.random.default_rng(1000 * seed + 7)
    start, nd = ctrl["seasons"][season]; wy = ctrl["weather_year"]; d, _ = load_site(root, site)
    doy0 = (pd.Timestamp(start) - pd.Timestamp(f"{wy}-01-01")).days
    dates = pd.date_range(start, periods=nd, freq="D"); src = replay_days(d, split, dates, seed=seed)
    inp = build_inputs(root, cfg, site, climate, wy, split, seed, n_days=nd, start_day=doy0, days_override=src)
    T = len(inp["To"]); idx = inp["index"]; hours = (idx.hour + idx.minute / 60).values; zone = Zone(cfg, inp); area = zone.area
    user = hidden_user(seed, site); H = ctrl["horizon"]; g, pn = pnc_table(root); cm = CostModel(cfg, ctrl, (g, pn), e_step_max(cfg)); x = np.array([[27.0, 27.0]])
    dim_s = np.zeros(T); cct_s = np.full(T, 4000.0); sp_s = np.zeros(T); out = {k: np.zeros(T) for k in ("Ta", "lux", "P_light", "P_hvac", "P_equip")}
    lux_day = zone.lux_day(np.arange(T))
    if method in ("B0", "B1"):
        dim, cct, sp = baselines.b0(idx, cfg) if method == "B0" else baselines.b1(inp["presence"], lux_day, cfg)
        o = zone.rollout(x, dim[None], cct[None], sp[None], 0); dim_s, cct_s, sp_s = dim, cct, sp
        for k in out: out[k] = o[k][0]
    else:
        p_all = forecasts(root, fl_cfg, model_file, site, src).reshape(-1, 8); n_typ = max(1.0, float(inp["n_people"][inp["presence"] > 0].mean())) if (inp["presence"] > 0).any() else 1.0
        # planning zone: forecast weather (observed + noise, documented sigma); occupancy overridden per replan
        fw = ctrl["weather_forecast"]; pin = dict(inp); pin["To"] = inp["To"] + rng.normal(0, fw["sigma_To_C"], T); pin["solar_w"] = inp["solar_w"] * (1 + rng.normal(0, fw["rel_sigma_solar"], T)).clip(0)
        pin["ghi"] = inp["ghi"] * (1 + rng.normal(0, fw["rel_sigma_solar"], T)).clip(0); pin = {k: (np.pad(v, (0, H + 1), mode="edge") if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == T else v) for k, v in pin.items()}; pz = Zone(cfg, pin)
        pl = Planner(pz, cm, ctrl, rng); du = APF.sensitivity(cm.w, H); pT = Posterior("thermal", ctrl["feedback"]["eps_fb"], 0.0, 1.0); pL = Posterior("lighting", ctrl["feedback"]["eps_fb"], np.log(500), 0.25)
        prev_a = (0.0, 4000.0, 28.0); a_pop, b_pop = 4400.0, 600.0
        for t in range(T):
            ph = np.ones(H) * p_all[t, 0]; ph = p_all[t, :H].copy(); ph = np.concatenate([ph, np.full(max(0, H - len(ph)), ph[-1])]) if len(ph) < H else ph
            belief = dict(delta=pT.mean() if personalise else 0.0, lux=float(np.exp(pL.mean())) if personalise else 500.0, a=a_pop, b=b_pop)
            hz = (hours[t] + 0.25 * np.arange(H)) % 24; tail = pl.plan(x, t, ph, n_typ, prev_a, belief, hz)
            u = pl.first_action_utility(x, t, ph, n_typ, tail, prev_a, belief, hz, float(inp["presence"][t]))
            a = APF.release(u, eps_act if apf_on else float("inf"), du, rng); da, ca, sa = decode(a)
            x, o = zone.step(x, float(da), float(ca), float(sa), t); dim_s[t], cct_s[t], sp_s[t] = da, ca, sa; prev_a = (float(da), float(ca), float(sa))
            for k in out: out[k][t] = np.ravel(o[k])[0]
            if personalise and inp["presence"][t] > 0 and rng.random() < ctrl["feedback"]["vote_prob_per_hour"] / 4:
                vt = int(true_votes("thermal", float(np.ravel(o["Ta"])[0]), user["delta"], rng)); pT.update(float(np.ravel(o["Ta"])[0]), int(randomise(np.array([vt]), ctrl["feedback"]["eps_fb"], rng)[0]))
                vl = int(true_votes("lighting", np.log(max(float(np.ravel(o["lux"])[0]), 1.0)), np.log(user["lux"]), rng)); pL.update(np.log(max(float(np.ravel(o["lux"])[0]), 1.0)), int(randomise(np.array([vl]), ctrl["feedback"]["eps_fb"], rng)[0]))
    # ---------------- metrics (TRUE hidden user) ----------------
    occ = inp["n_people"] > 0; w = occ.astype(float); wsum = max(w.sum(), 1.0)
    lux = lighting.total_lux(dim_s, lux_day, cfg["lighting"], occ) if method in ("B0", "B1") else out["lux"]
    dth = cm.d_th(out["Ta"], user["delta"]) if method not in ("B0", "B1") else CostModel(cfg, ctrl, pnc_table(root), 1.0).d_th(out["Ta"], user["delta"])
    cmm = CostModel(cfg, ctrl, pnc_table(root), 1.0); dth = cmm.d_th(out["Ta"], user["delta"]); dvis = cmm.d_vis(lux, user["lux"], cct_s, cct_pref(user["a"], user["b"], hours), True)
    pm = pmv(out["Ta"], out["Ta"], 0.1, inp["rh"], inp["met"], 0.6)
    kwh = lambda P: float(P.sum() * 0.25 / 1000)
    r = dict(site=site, season=season, method=method, seed=seed, eps_act=eps_act, eps_train=eps_train, apf=apf_on, personalise=personalise, climate=climate, perturb=perturb, run_mode=tag,
             light_kwh=kwh(out["P_light"]), hvac_kwh=kwh(out["P_hvac"]), equip_kwh=kwh(out["P_equip"]), area_m2=area, days=int(nd),
             ctrl_kwh=kwh(out["P_light"] + out["P_hvac"]), total_kwh=kwh(out["P_light"] + out["P_hvac"] + out["P_equip"]), peak_kw=float((out["P_light"] + out["P_hvac"] + out["P_equip"]).max() / 1000),
             d_th=float((w * dth).sum() / wsum), d_vis=float((w * dvis).sum() / wsum), comfort_score=float(1 - (w * 0.5 * (dth + dvis)).sum() / wsum),
             pmv_discomfort_h=float(((np.abs(pm) > 0.5) & occ).sum() * 0.25), lux_lt300_frac=float(((lux < 300) & occ).sum() / wsum), occupied_h=float(occ.sum() * 0.25))
    if keep_series:
        r["series"] = dict(dim=dim_s, cct=cct_s, sp=sp_s, kw=(out["P_light"] + out["P_hvac"] + out["P_equip"]) / 1000, presence=inp["presence"], Ta=out["Ta"], lux=lux, idx=np.asarray(idx.astype("int64")))
    return r
