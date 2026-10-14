# Pre-registered primary hypotheses (stub, dated 2026-10-14, written BEFORE running the grid)
Paired differences by (site, season, seed); 95% bootstrap CI (10,000 resamples); Holm correction across this family only. Everything else is exploratory.
- H1: PALACE (default eps_train, eps_act) reduces total energy vs B1 (PIR-reactive).
- H2: PALACE does not increase occupancy-weighted discomfort vs B1 by more than a margin (non-inferiority margin to be frozen after validation sweep: TBD before grid).
- H3: PALACE(eps_train in {1,4}) AUROC/energy within a margin of B2 (non-private ceiling); margin TBD before grid.
- H4: APF on vs off lowers actuation-adversary leakage L at the default eps_act.
Falsification: any hypothesis not supported is reported as such in RESULTS_SUMMARY.md.
