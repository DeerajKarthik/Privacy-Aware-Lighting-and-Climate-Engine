import sys, time, yaml; from pathlib import Path; import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.sim.year import build_inputs; from palace.sim.zone_env import Zone; from palace.control.baselines import b0, b1
cfg = yaml.safe_load((ROOT / "configs/sim.yaml").read_text())
t0 = time.time(); inp = build_inputs(ROOT, cfg, "ac1", "surat", 2019, "train"); print("build", round(time.time() - t0, 2), len(inp["To"]))
z = Zone(cfg, inp); T = len(inp["To"])
for name in ["b0", "b1"]:
    if name == "b0": dim, cct, sp = b0(inp["index"], cfg)
    else: dim, cct, sp = b1(inp["presence"], z.lux_day(np.arange(T)), cfg)
    t0 = time.time(); o = z.rollout(np.array([[27.0, 27.0]]), dim[None], cct[None], sp[None], 0); dt = time.time() - t0
    kwh = {k: o[k].sum() * 0.25 / 1000 / z.area for k in ["P_light", "P_hvac", "P_equip"]}
    print(name, "rollout s", round(dt, 2), {k: round(v, 1) for k, v in kwh.items()}, "total", round(sum(kwh.values()), 1), "Ta max", o["Ta"].max().round(1))
