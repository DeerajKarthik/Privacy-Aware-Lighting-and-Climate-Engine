import copy, yaml, numpy as np
from pathlib import Path
from palace.sim.thermal import Thermal
from palace.sim.zone_env import Zone, perturb_cfg
from palace.sim.occupants import n_people
from palace.sim.comfort import pmv
from palace.sim import hvac, lighting

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/sim.yaml").read_text())

def synth(T=96 * 3, To=35.0, npl=2.0, sol=300.0, ghi=500.0):
    return dict(To=np.full(T, To), solar_w=np.full(T, sol), ghi=np.full(T, ghi), n_people=np.full(T, npl), met=np.full(T, 1.2))

def test_energy_conservation_adiabatic():
    z = CFG["zone"]; th = Thermal(z["Ca_J_per_K"], z["Cm_J_per_K"], 1e14, z["Ram_K_per_W"], 900.0)
    x = np.array([[25.0, 25.0]]); Q = 800.0; n = 200
    for _ in range(n): x = th.step(x, np.array([[40.0, 0.0, Q, 0.0]]))
    stored = z["Ca_J_per_K"] * (x[0, 0] - 25) + z["Cm_J_per_K"] * (x[0, 1] - 25)
    assert abs(stored - Q * 900 * n) / (Q * 900 * n) < 1e-6

def test_steady_state_analytic():
    z = CFG["zone"]; th = Thermal(z["Ca_J_per_K"], z["Cm_J_per_K"], z["Rao_K_per_W"], z["Ram_K_per_W"], 900.0)
    x = np.array([[20.0, 20.0]])
    for _ in range(4000): x = th.step(x, np.array([[30.0, 0.0, 500.0, 0.0]]))
    assert abs(x[0, 0] - (30.0 + 500.0 * z["Rao_K_per_W"])) < 1e-6 and abs(x[0, 1] - x[0, 0]) < 1e-6

def _energy(cfg, inp, sp):
    zone = Zone(cfg, inp); T = len(inp["To"]); o = zone.rollout(np.array([[28.0, 28.0]]), np.full((1, T), 0.5), np.full((1, T), 4000.0), np.full((1, T), float(sp)), 0)
    return o["P_hvac"].sum() * 0.25 / 1000, o

def test_monotonicity():
    e_cool, _ = _energy(CFG, synth(To=33.0), 24); e_hot, _ = _energy(CFG, synth(To=37.0), 24)
    assert e_hot > e_cool > 0
    e_lo, _ = _energy(CFG, synth(), 23); e_hi, _ = _energy(CFG, synth(), 27)
    assert e_hi < e_lo

def test_no_nan_extremes():
    for To in (-30.0, 55.0):
        for sp in (22, 30):
            _, o = _energy(CFG, synth(To=To, sol=5000.0, ghi=1200.0, npl=4), sp)
            assert all(np.isfinite(v).all() for k, v in o.items())

def test_batch_equals_loop():
    inp = synth(T=96); z = Zone(CFG, inp); rng = np.random.default_rng(0)
    B, H = 7, 48; dim = rng.integers(0, 11, (B, H)) / 10; cct = rng.choice(np.arange(2700, 6501, 400), (B, H)).astype(float); sp = rng.integers(22, 31, (B, H)).astype(float)
    ob = z.rollout(np.array([[28.0, 27.0]]), dim, cct, sp, 5)
    for b in range(B):
        o1 = z.rollout(np.array([[28.0, 27.0]]), dim[b:b + 1], cct[b:b + 1], sp[b:b + 1], 5)
        for k in ("Ta", "P_hvac", "P_light", "lux"): assert np.abs(ob[k][b] - o1[k][0]).max() < 1e-9

def test_perturbation_hook():
    c2 = perturb_cfg(CFG, {"Rao": 1.3, "COP_ref": 0.7}); assert abs(c2["zone"]["Rao_K_per_W"] / CFG["zone"]["Rao_K_per_W"] - 1.3) < 1e-12
    assert CFG["ac"]["cop_ref"] == yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/sim.yaml").read_text())["ac"]["cop_ref"]

def test_occupant_function():
    assert (n_people([0, 1, 1, 1], [3, 1, 2.2, 9], capacity=4) == np.array([0, 1, 3, 4])).all()

def test_cop_and_light_monotone():
    c = CFG["ac"]; assert hvac.cop(25, c["cop_ref"], c["cop_t_ref_C"], c["cop_slope_per_K"], c["cop_floor"]) > hvac.cop(45, c["cop_ref"], c["cop_t_ref_C"], c["cop_slope_per_K"], c["cop_floor"])
    p = lighting.power_w(np.linspace(0, 1, 11), 4000, 25.0, CFG["lighting"]); assert (np.diff(p) > 0).all()
    assert lighting.power_w(0.5, 2700, 25, CFG["lighting"]) > lighting.power_w(0.5, 4000, 25, CFG["lighting"])

def test_pmv_iso7730_reference():
    # ISO 7730 Annex D reference values (ta=tr, v=0.10, rh=60, 1.2 met, 0.5 clo): 22 C -> -0.75, 27 C -> +0.77
    assert abs(float(pmv(22, 22, 0.1, 60, 1.2, 0.5)) - (-0.75)) < 0.06
    assert abs(float(pmv(27, 27, 0.1, 60, 1.2, 0.5)) - 0.77) < 0.06
