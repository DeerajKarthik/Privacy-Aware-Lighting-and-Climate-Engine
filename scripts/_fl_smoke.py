import sys, time, yaml; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.fl.engine import run_fl
cfg = yaml.safe_load((ROOT / "configs/fl.yaml").read_text())
for mode, eps in [("fedavg", float("inf")), ("dpfedavg", 1.0)]:
    t0 = time.time(); out = run_fl(["ac1", "scone"], cfg, ROOT, mode, eps, 0, rounds=3, verbose=False)
    s = out["stats"]; print(mode, round(time.time() - t0, 1), "s", {k: s[k] for k in ("params", "gpu_peak_mb", "comm_mb")}, s.get("eps_spent"), s.get("sigma"), s.get("delta"))
