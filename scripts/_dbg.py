import sys, glob, numpy as np; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import palace.sim.zone_env as Z
orig = np.stack
def dbg(a, axis=0):
    try: return orig(a, axis=axis)
    except ValueError: print("SHAPES", [np.shape(x) for x in a]); raise
Z.np.stack = dbg
from palace.eval.runner import run_cell
mf = glob.glob(str(ROOT / "results/runs/models/fast_office_central_epsinf_s0_*.pt"))[0]
run_cell(ROOT, "ac1", "summer", method="B2", model_file=mf, seed=0, keep_series=False)
