PY=.venv/bin/python
all: data features sim prefs occupancy control attacks stats package
data: ; $(PY) scripts/00_check_env.py; python3 scripts/01_download_data.py; python3 scripts/01b_download_weather.py
features: ; $(PY) scripts/02_build_features.py
sim: ; $(PY) scripts/03_calibrate_sim.py
prefs: ; $(PY) scripts/05a_train_thermal_pref.py; $(PY) scripts/05b_feedback_experiment.py
occupancy: ; $(PY) scripts/04_train_occupancy.py --methods central,local,fedavg --seeds 0-4 --tag fast; $(PY) scripts/04_train_occupancy.py --methods dpfedavg,dpfedper --eps 0.5,2,8 --seeds 0-2 --tag fast
control: ; $(PY) scripts/06_run_controllers.py --grid all --procs 8 --seeds 0-2
attacks: ; $(PY) scripts/07_attacks.py all; $(PY) scripts/08b_ladder.py
stats: ; $(PY) scripts/09_stats_figures.py
package: ; $(PY) scripts/10_package_report_bundle.py
test: ; $(PY) -m pytest tests -q
