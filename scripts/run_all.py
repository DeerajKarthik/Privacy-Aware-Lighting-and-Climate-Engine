"""Convenience wrapper: python scripts/run_all.py --mode fast  (runs `make all`)."""
import subprocess, sys
sys.exit(subprocess.call(["make", "all"], cwd=__import__("pathlib").Path(__file__).resolve().parents[1]))
