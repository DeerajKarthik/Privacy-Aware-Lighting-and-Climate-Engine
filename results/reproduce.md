# Reproduce (Linux; Python 3.13 venv with torch 2.9.0+cu128; see requirements.txt)
1. `python3 -m venv --system-site-packages .venv && .venv/bin/pip install statsmodels opacus gymnasium stable-baselines3 pvlib codecarbon lightgbm holidays`
2. `make data features sim prefs` (downloads ~0.7 GB; network-limited; office zip is resumable)
3. `make occupancy` (~2–3 h on RTX 3050 with three parallel processes), `make control` (~45 min on 11 CPU processes), `make attacks`, `make stats`, `python scripts/10_package_report_bundle.py`
Note: eps_train=0.5 DP runs were limited to seed 0 in the reported results (run with --seeds 0-2 to extend). Tests: `make test` (23 tests).
