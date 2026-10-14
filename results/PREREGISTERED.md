# Pre-registered primary hypotheses (stub, dated 2026-10-14, written BEFORE running the grid)
Paired differences by (site, season, seed); 95% bootstrap CI (10,000 resamples); Holm correction across this family only. Everything else is exploratory.
- H1: PALACE (default eps_train, eps_act) reduces total energy vs B1 (PIR-reactive).
- H2: PALACE does not increase occupancy-weighted discomfort vs B1 by more than a margin (non-inferiority margin to be frozen after validation sweep: TBD before grid).
- H3: PALACE(eps_train in {1,4}) AUROC/energy within a margin of B2 (non-private ceiling); margin TBD before grid.
- H4: APF on vs off lowers actuation-adversary leakage L at the default eps_act.
Falsification: any hypothesis not supported is reported as such in RESULTS_SUMMARY.md.

## Amendment (2026-10-14, BEFORE running the controller grid)
- Default operating point of PALACE = (eps_train=2, eps_act=8), fixed after a SINGLE smoke cell (ac1/summer/seed 0) showed that eps_act=2 makes the exponential mechanism nearly uniform because the first-action utility spread (~0.15) is only ~4x the sensitivity Delta_u=w_c/(H*sum(w)) ~ 0.04. The full eps_act in {0.5,2,8,inf} frontier is reported; the default is NOT selected on the test metrics of the grid.
- Non-inferiority margins (H2, H3): comfort_score difference >= -0.05 (absolute) vs B1; ctrl-energy no worse than B2 by more than 10 %. Fixed here before the grid.
- Primary family for Holm correction: {PALACE vs B1 energy, PALACE vs B1 comfort, PALACE(eps_train=2) vs B2 energy, PALACE(eps_train=2) vs B2 AUROC-equivalent comfort, APF on vs off leakage}.
