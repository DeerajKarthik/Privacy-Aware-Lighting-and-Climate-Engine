import sys, time, glob; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.eval.runner import run_cell
mf = glob.glob(str(ROOT / "results/runs/models/fast_office_central_epsinf_s0_*.pt"))[0]
for kw in [dict(method="B0"), dict(method="B1"), dict(method="B2", model_file=mf), dict(method="PALACE", model_file=mf, apf_on=True, eps_act=2.0, personalise=True)]:
    t0 = time.time(); r = run_cell(ROOT, "ac1", "summer", seed=0, keep_series=False, **kw)
    print(kw["method"], round(time.time() - t0, 1), "s", {k: round(r[k], 3) for k in ("ctrl_kwh", "total_kwh", "d_th", "d_vis", "comfort_score", "pmv_discomfort_h", "lux_lt300_frac")})
