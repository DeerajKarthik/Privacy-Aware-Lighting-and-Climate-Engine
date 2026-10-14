# PALACE: Privacy-Aware Lighting And Climate Engine
**Course project, AI for Sustainability · SVNIT Surat**
Authors: S K Deeraj (U23AI050), Aditya Kumar (U23AI029), Jaimin Vankar (U23AI058)
Design dossier v1 · 14 Oct 2026 · Hardware target: RTX 3050 6 GB laptop, public data only

---

## 1. One-paragraph pitch

Personalised lighting and cooling saves energy only if the building learns *who is where, when, and what they like*. That same knowledge is surveillance data. PALACE learns occupant movement and preference patterns with **differentially private federated learning**, uses them in a **model-predictive controller** that sets per-zone dimming, colour temperature (CCT) and AC setpoint, and then protects the **control output itself** with an exponential-mechanism actuation filter. We measure a **three-way frontier of energy and carbon, comfort, and privacy**, where privacy is both a formal budget (ε) and an empirical attack success rate. We also charge the ML its own electricity bill, so the headline claim is **net** sustainability.

## 2. How I read the brief (flag if wrong)

| Phrase in brief | Interpretation used |
|---|---|
| "individual or aggregate user movement" | A **granularity ladder**: raw individual trace, then aggregate counts, then DP-noised aggregates. We measure what each rung costs in control quality and buys in privacy. |
| "localized lighting frequencies" | Per-zone **spectral/colour tuning (CCT) and dimming level**. Electrical flicker/PWM frequency is *not* optimised. It is held fixed at a flicker-safe value (≥ 1 kHz). Say this explicitly in the report. |
| "ambient temperatures" | Per-zone AC cooling setpoint (Surat is cooling-dominated). |
| "privacy-preserving ML" | DP-SGD inside federated learning, randomised-response feedback, and DP at the actuation layer. |

## 3. Gap analysis (what exists, what does not)

Verified by reading the sources below (summaries from the papers' own text; searched Oct 2026, not a systematic review).

| Prior work | What it does | What it leaves open |
|---|---|---|
| Jia, Dong, Sastry, Spanos, *Privacy-enhanced architecture for occupancy-based HVAC control* (arXiv:1607.03140) | Distorts occupancy data with mutual-information privacy, MPC for HVAC | HVAC only, no lighting, no learning across sites, simulation only, assumes independent Markov occupants |
| *Employing Federated Learning for Training Autonomous HVAC Systems* (arXiv:2405.00389) | FedAvg-trained SAC controllers, Sinergym, 11 clients | **No formal privacy** (authors say so), data-centre building, no occupant preference, no lighting |
| *A Privacy-Preserving Framework for Cloud-Based HVAC Control* (arXiv:2312.05183) | CKKS-encrypted MPC, event-triggered | ~360× compute overhead, informal timing-privacy claim, no learning, no lighting |
| *Decentralized intelligence in SBs: Federated LSTM digital twins* (Frontiers Built Env. 2025) | FL occupancy detection | FL only (no DP, authors admit inversion risk), one dataset split into 3 pseudo-clients, energy savings are *modelled* assumptions |
| Federated thermal-comfort papers (e.g. Sci. Direct S2352710225027135, S0952197624016221) | Personalised comfort prediction by FL | Comfort prediction only, not joint lighting + climate control, privacy not measured against attacks |
| DP-pFedDSU (arXiv:2503.09192) | Good DP personalised FL method | Image benchmarks only, no buildings |

**Gap.** I found no work that (a) controls lighting *and* climate jointly from learned behaviour, (b) gives formal DP accounting for the learning, (c) protects and attacks the *actuation traces* that a cloud, utility or neighbour could observe, and (d) reports the result net of the ML's own energy. Treat this as "to our knowledge, from the searches we ran", never as "first".

## 4. Contributions (what the paper will claim, and only if the data supports it)

1. **PALACE**: DP-federated occupancy forecaster + preference model + CEM-MPC controller for joint per-zone lighting and AC control.
2. **Actuation privacy filter (APF)**: exponential-mechanism selection over a discretised action set, giving a per-step ε_act guarantee with respect to the live presence signal. Includes a plain-language statement of what it does *not* protect.
3. **Actuation-leakage attack and metric**: an adversary who sees only lighting/setpoint/meter logs tries to recover presence. Reported as attack AUC versus ε.
4. **Granularity ladder** study: individual → aggregate → DP aggregate.
5. **Net-sustainability accounting**: energy, carbon and peak demand saved, minus measured training energy (CodeCarbon), with a payback time in days.

If contribution 3 or 5 fails to show anything interesting, the paper still stands on 1, 2, 4 plus honest negative findings. Do not promise more than the experiments give.

## 5. Data plan (what is real, what is synthetic)

| Need | Source | Verified facts | Role |
|---|---|---|---|
| Movement (aggregate, offices) | CASAS office dataset, Zenodo 15713496, CC BY 4.0 | PIR motion + door sensors, 11 settings (ac1–ac4, museum, scone, stair01–04, tokyo), fields: date/time/sensor/message. Zip ≈ 424 MB | Clients = offices (use ac1–ac4, scone, tokyo; museum/stairs optional) |
| Movement (individual, homes) | CASAS Aruba/Cairo/Milan/Tulum, Zenodo 17180309, CC BY 4.0 | PIR, door, temperature sensors, activity labels, zip ≈ 33 MB | Clients = homes. Activity labels map to metabolic rate (met) and lighting task |
| Thermal preference (real) | ASHRAE Global Thermal Comfort Database II (GitHub `CenterForTheBuiltEnvironment/ashrae-db-II`, Dryad doi:10.6078/D1F671) | Columns include `thermal_sensation`, `thermal_preference`, `ta`, `clo`, `met`, `building_type`, `country` | Trains the real thermal-preference model |
| Load realism | Building Data Genome 2 (GitHub `buds-lab/building-data-genome-project-2`) | 1,636 buildings, 3,053 meters, hourly, 2016–2017 | Calibrate office energy intensity to a plausible range, sanity-check simulator |
| Weather + solar | Open-Meteo archive API (no key for non-commercial) | Hourly `temperature_2m`, `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation`, `cloud_cover` | Surat (21.17 N, 72.83 E) primary, New Delhi secondary |
| Lighting preference | **None public at per-user level** | Searched; only closed or small field studies | **Synthetic**, grounded in lighting literature. Always labelled synthetic |

**Be upfront in the report:** thermal preference is learned from real survey data. Lighting preference and the link from a CASAS site to an ASHRAE "persona" are synthetic. Mitigation: sensitivity analysis over the generator's parameters and a misspecification test (train on one generator, test on another).

**Unverified, check at download time:** licences of ASHRAE DB II and BDG2 (the pages I read did not state them), whether ASHRAE DB II raw files need Dryad access, whether CASAS raw data really expands to the sizes listed on Zenodo (the page shows 108 GB and 25 GB totals against much smaller zips).
Dropped: ecobee Donate Your Data (requires an application, not open).

## 6. Method in brief

```
CASAS PIR streams ──► per-site hourly features ──► DP-FedAvg occupancy forecaster ─┐
(local, never leaves)    (arrival, dwell, density)    (day-level DP-SGD, Opacus)   │ p(occupied | t..t+H)
                                                                                    ▼
ASHRAE DB II ──► thermal-preference model (FL) ─┐                          CEM-MPC controller
Synthetic lighting prefs + RR feedback ──► prefs ┴──► preference posterior ──► per-zone (dim, CCT, setpoint)
                                                                                    │
                       Surat weather + 2R2C zone sim + LED/AC models ◄──────────────┤
                                                                                    ▼
                                                      Actuation privacy filter (exp. mechanism)
                                                                                    │ released logs
                                                                                    ▼
                                                   Adversary: presence inference from actuation + meter
```

**Occupancy model.** Small 1-D TCN or GRU (≤ 0.5 M parameters) predicting next-H presence per zone from the last 24–48 h. One training example is one *site-day*, so DP is day-level. Local DP-SGD with Poisson sampling, then FedAvg. Personalisation by local head (FedPer).

**Preference model.** Thermal: gradient-boosted or small MLP trained on ASHRAE DB II to give P(no change | air temp, clo, met, context). Lighting: parametric synthetic users with latent preferred lux and CCT. Per-user offsets are learned online from sparse "too warm / too cold / too dim" votes, which are shared only through **randomised response** (local DP).

**Controller.** Cross-entropy-method MPC, horizon 8 steps (2 h at 15 min), 256 candidates, 3 iterations, vectorised rollouts through the zone simulator. Cost = carbon-weighted energy + expected discomfort (weighted by forecast occupancy) + actuation smoothness.

**Simulator.** 2R2C thermal zone (air and mass nodes), split-AC with outdoor-temperature-dependent COP, daylight from global horizontal irradiance via luminous efficacy and a daylight factor, dimmable tunable-white LED. Parameters in YAML, perturbed ±30 % in robustness runs. EnergyPlus/Sinergym cross-check is a stretch goal.

**Why a custom simulator rather than only Sinergym?** Sinergym (arXiv:2412.08293, EnergyPlus 24.1, Gymnasium API) has no native tunable lighting, and thousands of controller rollouts must run on a laptop. A reduced-order model is fast and transparent. The cost is realism, so we calibrate to BDG2 energy intensities and say so.

## 7. Baselines and ablations

Baselines: B0 fixed schedule · B1 PIR-reactive rule · B2 centralised non-private forecaster + MPC (utility ceiling) · B3 local-only forecaster · B4 FedAvg no DP · B5 DP-FedAvg no personalisation · B6 PPO (centralised, no privacy; SB3, CPU).
Ablations: ε_train ∈ {0.5, 1, 2, 4, 8, ∞} · ε_act ∈ {0.5, 1, 2, 4, 8, ∞} · with/without personalisation · with/without APF · granularity rung · horizon · number of clients · simulator ±30 % · generator misspecification · season · second climate.
Statistics: 5 seeds minimum, 95 % bootstrap CIs, paired tests with Holm correction. Chronological and leave-one-site-out splits.

## 8. Risks and how they are handled

| Risk | Handling |
|---|---|
| Few clients make client-level DP vacuous | Use day-level DP and say so. Do not claim user-level DP |
| Lighting preference is synthetic | Label everywhere. Run sensitivity and misspecification tests |
| CASAS PIR is not person-identified | "Individual" regime only for single-resident homes. Verify from dataset docs |
| APF may cost too much comfort | That is a finding. Report the frontier honestly |
| Raw data larger than expected | Stream-parse, cache hourly Parquet, cap years per site |
| Simulator not trusted | Calibration to BDG2, sensitivity runs, optional EnergyPlus check |
| 6 GB GPU | Models are tiny. The simulator and MPC run on CPU/vectorised. GPU used for forecaster only |

## 9. Timeline (5–6 weeks; adjust to your deadline)

| Week | Goal | Exit check |
|---|---|---|
| 1 | Environment, data download, validation, hourly features, EDA | `data/processed/*.parquet` + EDA figure |
| 2 | Simulator + unit tests, baselines B0/B1, calibration to BDG2 | Energy intensity in plausible range, physics tests pass |
| 3 | Occupancy FL → DP-FL, preference models, attacks (MIA) | Accuracy vs ε curve, MIA AUC |
| 4 | CEM-MPC, APF, actuation attack, B2–B6 | End-to-end run on one site, all baselines |
| 5 | Full grid, stats, figures, carbon accounting | `paper_numbers.json`, all figures |
| 6 | **Report** in IEEE format (I write it from your results), polish, buffer | Final PDF/DOCX |

Split of work across three authors: (1) data + simulator, (2) FL/DP + attacks, (3) controller + sustainability accounting. Everyone reads and edits the report.

## 10. Report plan (IEEE two-column, A4, 6–8 pages)

Abstract · Introduction · Related Work · Problem Formulation and Threat Model · PALACE Method · Experimental Setup (data, simulator, baselines, metrics) · Results (forecast vs ε, Pareto frontier, granularity ladder, ablations, attacks) · Sustainability Analysis (net energy, carbon, payback) · Limitations and Ethics · Conclusion · Acknowledgment · References (numeric brackets, per the template you supplied; no numbered text heads typed by hand, template styles only).
I will fill this from `results/paper_numbers.json` so no number is invented.

## 11. Sources checked

- CASAS office dataset: https://zenodo.org/records/15713496
- CASAS homes dataset: https://www.zenodo.org/records/17180309
- ASHRAE DB II: https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II
- Building Data Genome 2: https://github.com/buds-lab/building-data-genome-project-2
- Sinergym: https://arxiv.org/html/2412.08293v1
- Jia et al.: https://ar5iv.labs.arxiv.org/html/1607.03140
- Federated HVAC RL: https://arxiv.org/pdf/2405.00389
- Encrypted cloud HVAC: https://arxiv.org/pdf/2312.05183
- Federated LSTM digital twins: https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2025.1696702/full
- DP-pFedDSU: https://arxiv.org/html/2503.09192v1
- Open-Meteo: https://open-meteo.com/en/docs/historical-weather-api
- Two ScienceDirect federated-comfort papers were found by search but their pages were blocked from fetching; they are cited from titles only. Re-read them before citing.