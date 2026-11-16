# PALACE — Privacy-Aware Lighting And Climate Engine

Course project for **AI for Sustainability**, SVNIT Surat.

**GitHub:** <https://github.com/DeerajKarthik/Privacy-Aware-Lighting-and-Climate-Engine>

## Group members

| Name | Roll No. |
|---|---|
| S K Deeraj | U23AI050 |
| Aditya Kumar | U23AI029 |
| Jaimin Vankar | U23AI058 |

---

## 1. Project overview

Personalised lighting and cooling saves energy only if a building learns *who is where, when, and what they like*. That same knowledge is surveillance data. PALACE tries to get the savings while limiting the privacy cost:

1. **DP-federated occupancy forecaster.** A TCN is trained across sites with DP-SGD (Opacus) inside federated learning (FedAvg and FedPer variants). The RDP accountant tracks the privacy budget.
2. **Preference models.** A thermal-preference model is trained on the ASHRAE Global Thermal Comfort Database II. Randomised-response feedback is used to update per-occupant preference posteriors.
3. **Model-predictive controller.** A CEM-MPC sets per-zone dimming, colour temperature (CCT) and AC setpoint on a calibrated reduced-order building simulator (2R2C thermal model, tunable-white LED, cooling-only AC, PMV comfort).
4. **Actuation Privacy Filter (APF).** An exponential-mechanism filter on the control output gives a per-step ε_act guarantee with respect to the live presence signal.
5. **Attacks and audits.** Membership-inference attacks (loss-threshold and LiRA-lite) are run against the forecaster. An *actuation adversary* tries to recover presence from lighting, setpoint and meter logs.
6. **Net sustainability accounting.** The ML's own energy (CodeCarbon) is charged against the energy and carbon savings.

The study measures a three-way frontier of **energy/carbon, comfort and privacy** against baselines B0–B5. The full design is in [PALACE_project_design.md](PALACE_project_design.md). Hypotheses were fixed in advance in [PREREGISTERED.md](PREREGISTERED.md).

### Headline results (fast mode)

All numbers trace to [results/paper_numbers.json](results/paper_numbers.json) and [results/tables/](results/tables). The full write-up, including every negative result, is in [results/RESULTS_SUMMARY.md](results/RESULTS_SUMMARY.md).

- **Pre-registered H1–H3 fail** for the default PALACE configuration with the APF on (ε_act = 8). It uses more energy and has worse comfort than baseline B1.
- **The same controller without the APF** saves **20.6 %** lighting+HVAC energy versus B1 (CI 17.7–23.7 %). The comfort difference is within the pre-set margin.
- **Forecaster AUROC:** central 0.898, DP-FedAvg 0.876 at ε = 8, DP-FedPer 0.858 at ε = 8. DP-FedPer drops to 0.691 at ε = 0.5 (one seed). Federation gave no accuracy gain over local training at these data volumes.
- **MIA and LiRA-lite are near chance** (AUC 0.49–0.55). They cannot tell ε levels apart, so this is a null result and not evidence of strong privacy.
- **The APF cuts actuation-only leakage, but the electricity meter still leaks presence** (AUROC ≈ 0.98). The APF's energy and comfort cost is large.

### Synthetic components and limits

These parts are synthetic and labelled as such throughout the code and reports:

- Lighting preferences and personal thermal offsets.
- The link between CASAS sites and ASHRAE personas.
- The pairing of occupancy years with weather years.

Results come from a reduced-order simulator, not field measurements. Runs cover one week per season, 3 seeds, and a single climate. See §7–8 of [results/RESULTS_SUMMARY.md](results/RESULTS_SUMMARY.md) for the full list of limitations and deviations.

---

## 2. Repository structure

```
palace/
├── README.md
├── CLAUDE.md                  # project rules / working conventions
├── PALACE_project_design.md   # design dossier
├── PREREGISTERED.md           # pre-registered hypotheses
├── DECISIONS.md               # logged design choices
├── PROGRESS.md                # per-phase progress log
├── Makefile                   # stage targets (data → package)
├── requirements.in / .txt     # direct deps / pinned environment
├── pytest.ini
├── configs/                   # YAML configs: base, sim, fl, control, grid_fast
├── scripts/                   # numbered, resumable pipeline stages
│   ├── 00_check_env.py            environment check
│   ├── 01_download_data.py        CASAS, ASHRAE, BDG2 + MANIFEST (hashes, licences)
│   ├── 01b_download_weather.py    Open-Meteo weather (Surat, Delhi)
│   ├── 02_build_features.py       features, splits, EDA
│   ├── 03_calibrate_sim.py        simulator calibration / validation
│   ├── 04_train_occupancy.py      central / local / FedAvg / DP-FedAvg / DP-FedPer
│   ├── 05a_train_thermal_pref.py  ASHRAE thermal-preference model
│   ├── 05b_feedback_experiment.py randomised-response feedback posterior
│   ├── 06_run_controllers.py      controller grid (B0–B5, PALACE)
│   ├── 07_attacks.py              MIA, LiRA-lite, actuation adversary
│   ├── 08b_ladder.py              granularity ladder
│   ├── 09_stats_figures.py        statistics, tables, figures
│   └── 10_package_report_bundle.py
├── src/palace/
│   ├── data/      stream parsing, features, BDG2, weather
│   ├── sim/       thermal, HVAC, lighting, daylight, occupants, comfort (PMV), yearly rollout
│   ├── models/    occupancy TCN, metrics
│   ├── fl/        federated engine, data, evaluation
│   ├── dp/        privacy accountant, randomised response
│   ├── prefs/     thermal, lighting and RR-feedback preference models
│   ├── control/   CEM-MPC, baselines, APF, cost terms
│   ├── attacks/   membership inference, actuation adversary
│   ├── eval/      runner, aggregation, carbon, Pareto, stats
│   ├── viz/       plotting style
│   └── utils/     I/O, hashing, manifests
├── tests/                     # test_data, test_sim, test_dp, test_prefs, test_control (23 tests)
├── results/                   # tables, figures, paper_numbers.json, RESULTS_SUMMARY.md, reproduce.md
├── logs/                      # run logs and ISSUES.md (failures and fallbacks)
├── report_bundle/             # packaged deliverables (also report_bundle.zip)
└── data/                      # GITIGNORED: raw/ and processed/ (not in the repo)
```

---

## 3. Datasets and resources

Raw data is **not committed**. `scripts/01_download_data.py` downloads it and records sizes, MD5 hashes and licences in `data/raw/*/MANIFEST.json`.

| Dataset | Used for | Source | Licence (as read at download) |
|---|---|---|---|
| **CASAS unscripted office behaviour** (motion, door, light) | Occupancy forecasting in office-type sites | Zenodo, DOI [10.5281/zenodo.15713496](https://doi.org/10.5281/zenodo.15713496) | CC BY 4.0 |
| **CASAS smart-home free living** (Aruba, Cairo, Milan, Tulum) | Home-type occupancy sites, granularity "individual" regime | Zenodo, DOI [10.5281/zenodo.17180309](https://doi.org/10.5281/zenodo.17180309) (layout record [15708568](https://zenodo.org/records/15708568)) | CC BY 4.0 |
| **ASHRAE Global Thermal Comfort Database II** (v2.1.0) | Thermal-preference model | [CenterForTheBuiltEnvironment/ashrae-db-II](https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II) | **Not stated in the repo.** Logged in [logs/ISSUES.md](logs/ISSUES.md). The Dryad record (doi:10.6078/D1F671) must be checked manually. |
| **Building Data Genome Project 2 (BDG2)** | Plausibility check of simulated office EUI | [buds-lab/building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2) | CC BY-SA 4.0 |
| **Open-Meteo historical archive** (Surat, Delhi, 2018–2023) | Weather for the simulator | <https://archive-api.open-meteo.com/v1/archive> | Open-Meteo terms. Source terms not yet read in full, see ISSUES.md. |
| **Grid emission factor** 0.727 kg CO₂/kWh (CEA v20.0, FY2023-24) | Carbon accounting | Secondary listing | **Unverified**, reported with a ±30 % band. See ISSUES.md. |

Other references are in [results/references_verified.bib](results/references_verified.bib).

> **Privacy note:** the CASAS data describes real households and offices. Never print or commit raw per-event data. Only derived features are kept in `data/processed/`. See [results/ETHICS_NOTES.md](results/ETHICS_NOTES.md).

---

## 4. Environment setup

### Tested environment

| Component | Value |
|---|---|
| OS | Linux (the scripts use `pathlib`, so they should also run on Windows with `num_workers=0`) |
| Python | 3.13.5. The spec asked for 3.10–3.11. The deviation is logged as D009 in [DECISIONS.md](DECISIONS.md). |
| PyTorch | 2.9.0 (CUDA 12.8 build) |
| GPU | NVIDIA RTX 3050 Laptop, 6 GB. Only about 3.7 GiB was available, and the code keeps usage under 5 GB. |
| Differential privacy | Opacus 1.6.0 |
| Disk and network | About 0.7 GB of downloads. The CASAS office zip is large and the download is resumable. |

### Install

```bash
git clone https://github.com/DeerajKarthik/Privacy-Aware-Lighting-and-Climate-Engine.git
cd Privacy-Aware-Lighting-and-Climate-Engine

# venv that reuses an existing CUDA build of torch from the system site-packages
python3 -m venv --system-site-packages .venv

# extra packages
.venv/bin/pip install statsmodels opacus gymnasium stable-baselines3 pvlib \
    codecarbon lightgbm holidays

# or, for the full pinned environment
# .venv/bin/pip install -r requirements.txt
```

Install PyTorch separately with the CUDA build that matches your driver (<https://pytorch.org>) if it is not already on the system. Direct dependencies are listed in [requirements.in](requirements.in). Exact versions are in [requirements.txt](requirements.txt).

### Check the environment

```bash
.venv/bin/python scripts/00_check_env.py
```

---

## 5. Running the pipeline

Each stage is resumable and caches its outputs. Stages can be run with `make` or by calling the scripts directly.

```bash
make data        # download datasets, write MANIFESTs, fetch weather
make features    # build features, splits, EDA
make sim         # calibrate and validate the simulator
make prefs       # thermal preference model + RR feedback experiment
make occupancy   # forecaster: central, local, FedAvg, DP-FedAvg, DP-FedPer
make control     # controller grid (8 processes by default)
make attacks     # MIA / LiRA-lite / actuation adversary / granularity ladder
make stats       # statistics, tables, figures
make package     # build report_bundle/
make all         # everything above, in order
make test        # run the test suite (23 tests)
```

Approximate costs on the reference laptop (see [results/reproduce.md](results/reproduce.md)):

- `make occupancy` takes about 2–3 h on the RTX 3050 with three parallel processes.
- `make control` takes about 45 min on 11 CPU processes.

Runs record seeds, library versions, the git hash and a config hash for determinism. Seeds are never dropped. The ε_train = 0.5 DP runs in the reported results use seed 0 only. Pass `--seeds 0-2` to extend them.

### Where to look

- Results narrative: [results/RESULTS_SUMMARY.md](results/RESULTS_SUMMARY.md)
- Numbers cited in the report: [results/paper_numbers.json](results/paper_numbers.json)
- Tables: [results/tables/](results/tables)
- Figures: [results/figures/](results/figures), indexed in [results/figure_inventory.md](results/figure_inventory.md)
- Failures, fallbacks and deviations: [logs/ISSUES.md](logs/ISSUES.md) and [DECISIONS.md](DECISIONS.md)

---

## 6. Working rules

The project follows the rules in [CLAUDE.md](CLAUDE.md). The main ones are:

- No fabricated numbers. Every reported number traces to a file, and null or negative results are reported.
- Nothing is tuned on the test split.
- Synthetic components are labelled as synthetic.
- Licences, hashes and column names are verified at runtime.
- `data/` is gitignored.
