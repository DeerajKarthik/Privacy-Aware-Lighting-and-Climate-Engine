# PALACE — project rules (Claude Code)
Privacy-Aware Lighting And Climate Engine. Course: AI for Sustainability, SVNIT Surat. Design: PALACE_project_design.md.

## Non-negotiable rules
1. Never fabricate numbers, results, citations, dataset facts. Failures go in logs/ISSUES.md; use documented fallback.
2. Every report number traces to a file; headline numbers go to results/paper_numbers.json.
3. Report negative/null results. Never tune on test split. Never drop seeds.
4. Label synthetic things as synthetic (lighting prefs, CASAS->ASHRAE persona link).
5. Verify licences, hashes, column names at runtime. Don't hard-code unverified facts.
6. No interactive questions; log choices in DECISIONS.md.
7. Resumable stages with caching; commit after each phase.
8. Laptop: RTX 3050 6GB (keep <5GB GPU; currently only ~3GB free), low RAM/disk. pathlib; num_workers=0 on Windows.
9. Determinism: seeds, library versions, git hash, config hash per run.
10. Privacy hygiene: never print raw per-event data; data/ is gitignored; keep only derived features in data/processed.

## Commands (venv: .venv/bin/python)
- scripts/00_check_env.py  env check
- scripts/01_download_data.py  download + MANIFEST
- scripts/02_build_features.py  features/splits/EDA
- (later stages 03..10 per master prompt)

## Current phase
Phase 8 — packaging (done in fast mode)
