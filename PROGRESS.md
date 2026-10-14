# PROGRESS
- Phase1 partial: homes parsed (5 files, 5 s). Office zip downloading (slow net). ASHRAE+BDG2 downloaded. Torch install still in progress.
## Phase 1 — DATA (done 2026-10-14)
- Downloaded + manifests: CASAS offices (md5 OK, 2.02 GB uncompressed, not 108 GB), CASAS homes (md5 OK), ASHRAE DB II (109,033 rows; licence NOT stated in repo -> logs/ISSUES.md), BDG2 (CC BY-SA 4.0; 261 offices, EUI p10/p50/p90 = 53/124/293 kWh/m2/yr).
- Streaming parser (message-based classification), 15-min bins, outage masking, presence w/ hold; hold sensitivity in site_summary_hold{0,30,60}.csv.
- 15 sites (10 office-type, 5 home files) all meet min days; ac2 dropped. Only Aruba is verified single-resident.
- Issues: weather years do not overlap most sites (replay rule D007/D013); single-sensor office sites; museum/cairo/milan have ~0 test days under 7-day gaps.
- Tests: tests/test_data.py 4 passed. Parquet load max 0.02 s. EDA: results/figures/F2_eda_presence.pdf.
- Env: py3.13 + torch 2.9.0+cu128 (reused), GPU 3.7 GiB total. Python deviates from spec (D009).
## Phase 2 — SIMULATOR (done)
- 2R2C exact discretisation, cooling-only AC (COP(To), part-load), tunable-white LED, pvlib solar gain, PMV (ISO 7730 ref values pass), perturbation hook.
- tests/test_sim.py: 9 passed (adiabatic conservation, analytic steady state, monotonicity, no-NaN, batch==loop 1e-9, perturbation, occupants, PMV ref).
- Site-year rollout ~2.5-2.9 s on CPU (limit 3 s; tokyo 2.9 s).
- B0/B1 EUI 70-144 kWh/m2/yr, all in BDG2 p10-p90 (results/tables/sim_calibration.csv, F3_sim_validation.pdf). Weather: Surat+Delhi 2018-2023 downloaded.
