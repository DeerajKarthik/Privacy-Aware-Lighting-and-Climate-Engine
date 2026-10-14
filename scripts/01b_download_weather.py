import sys, json; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.data.weather import fetch_year
from palace.utils.io import update_manifest
out = ROOT / "data/raw/weather"; out.mkdir(parents=True, exist_ok=True)
for loc in ["surat", "delhi"]:
    for y in range(2018, 2024):
        try: p = fetch_year(loc, y, out); print("ok", p.name, flush=True)
        except Exception as e:
            print("FAIL", loc, y, e, flush=True); open(ROOT / "logs/ISSUES.md", "a").write(f"- weather {loc} {y} failed: {e!r}\n")
update_manifest(out, {"name": "open-meteo archive 2018-2023", "url": "https://archive-api.open-meteo.com/v1/archive",
    "licence_as_read": "Open-Meteo: CC BY 4.0 for data, non-commercial free API (verify at https://open-meteo.com/en/terms)", "note": "licence text to be verified"})
print("FIN")
