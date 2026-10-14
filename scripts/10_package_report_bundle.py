import json, shutil, sys, zipfile; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; B = ROOT / "report_bundle"; shutil.rmtree(B, ignore_errors=True)
for d in ("figures", "tables"): shutil.copytree(ROOT / "results" / d, B / d)
for f in ["paper_numbers.json", "RESULTS_SUMMARY.md", "PREREGISTERED.md", "GRID_DESIGN.md", "ETHICS_NOTES.md", "references_verified.bib", "figure_inventory.md", "reproduce.md"]:
    p = ROOT / "results" / f
    if p.exists(): shutil.copy(p, B / f)
    else: print("MISSING", f)
for f in ["DECISIONS.md", "logs/ISSUES.md", "requirements.txt"]: shutil.copy(ROOT / f, B / Path(f).name)
with zipfile.ZipFile(ROOT / "report_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in B.rglob("*"):
        if p.is_file(): z.write(p, p.relative_to(B))
print("bundle:", len(list(B.rglob('*'))), "entries ->", ROOT / "report_bundle.zip")
