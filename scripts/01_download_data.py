"""Download datasets with manifests/hashes/licences. Usage: 01_download_data.py [casas_office casas_home ashrae bdg2]"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.utils.io import http_get, download, file_hash, update_manifest

RAW = ROOT / "data" / "raw"; ISSUES = ROOT / "logs" / "ISSUES.md"
def issue(msg):
    with open(ISSUES, "a") as f: f.write(f"- {msg}\n")
    print("ISSUE:", msg)

def zenodo(rec, name, dest_dir, expect_md5):
    meta = http_get(f"https://zenodo.org/api/records/{rec}").json()
    lic = meta["metadata"].get("license", {}).get("id")
    f = next(x for x in meta["files"] if x["key"] == name)
    dest = dest_dir / name
    if not dest.exists() or dest.stat().st_size != f["size"]:
        download(f"https://zenodo.org/records/{rec}/files/{name}?download=1", dest)
    md5 = file_hash(dest)
    if md5 != expect_md5: issue(f"{name} md5 {md5} != expected {expect_md5}")
    update_manifest(dest_dir, {"name": name, "url": f"https://zenodo.org/records/{rec}", "doi": meta.get("doi"),
        "size": dest.stat().st_size, "md5": md5, "md5_expected": expect_md5, "md5_ok": md5 == expect_md5,
        "licence_as_read": lic, "zenodo_checksum": f["checksum"]})
    # also save record description (for resident counts etc.)
    (dest_dir / f"record_{rec}.json").write_text(json.dumps(meta["metadata"], indent=1))

def casas_office():
    zenodo(15713496, "officedata.zip", RAW / "casas_office", "9f51e29428e2e4f24ac7cb0cb5fc76d6")
def casas_home():
    zenodo(17180309, "new_labeled_data.zip", RAW / "casas_home", "86954063e1d2099d288f59227b19a749")
    d = RAW / "casas_home"
    r = http_get("https://zenodo.org/api/records/15708568").json()
    (d / "layout_record_15708568.json").write_text(json.dumps(r["metadata"], indent=1))

GH = "https://api.github.com/repos/{}/contents/{}"
def gh_licence(repo):
    br = http_get(f"https://api.github.com/repos/{repo}").json().get("default_branch", "master")
    out = {"branch": br}
    for fn in ["LICENSE", "LICENSE.md", "LICENSE.txt"]:
        try: out["licence_file"] = fn; out["licence_text"] = http_get(f"https://raw.githubusercontent.com/{repo}/{br}/{fn}").text[:600]; return out
        except Exception: pass
    out["licence_file"] = None
    rd = http_get(f"https://raw.githubusercontent.com/{repo}/{br}/README.md").text
    out["readme_licence_lines"] = [l for l in rd.splitlines() if re.search(r"licen|creative commons|cc[- ]by", l, re.I)][:8]
    return out

def ashrae():
    d = RAW / "ashrae"; repo = "CenterForTheBuiltEnvironment/ashrae-db-II"; lic = gh_licence(repo)
    for fn in ["db_measurements_v2.1.0.csv.gz", "db_metadata.csv"]:
        p = d / fn
        if not p.exists():
            download(f"https://raw.githubusercontent.com/{repo}/{lic['branch']}/v2.1.0/{fn}", p)
        update_manifest(d, {"name": fn, "url": f"https://github.com/{repo}/tree/{lic['branch']}/v2.1.0", "size": p.stat().st_size,
                            "sha256": file_hash(p, "sha256"), "licence_as_read": lic})
    if not lic.get("licence_file") and not lic.get("readme_licence_lines"):
        issue("ASHRAE DB II: no licence statement in repo; record Dryad licence manually (doi:10.6078/D1F671).")

def bdg2():
    d = RAW / "bdg2"; repo = "buds-lab/building-data-genome-project-2"; lic = gh_licence(repo); br = lic["branch"]
    media = f"https://media.githubusercontent.com/media/{repo}/{br}/"
    raw = f"https://raw.githubusercontent.com/{repo}/{br}/"
    listing = http_get(GH.format(repo, "data/meters/cleaned")).json()
    print("bdg2 cleaned:", [(x["name"], x["size"]) for x in listing])
    files = ["data/metadata/metadata.csv", "data/meters/cleaned/electricity_cleaned.csv", "data/weather/weather.csv"]
    for fn in files:
        p = d / Path(fn).name
        if not p.exists():
            try: download(media + fn, p)
            except Exception as e: issue(f"BDG2 {fn}: {e}"); continue
        head = p.read_bytes()[:60]
        if head.startswith(b"version https://git-lfs"): issue(f"BDG2 {fn} is LFS pointer, not data"); continue
        update_manifest(d, {"name": p.name, "url": media + fn, "size": p.stat().st_size, "sha256": file_hash(p, "sha256"), "licence_as_read": lic})

if __name__ == "__main__":
    steps = sys.argv[1:] or ["casas_home", "ashrae", "bdg2", "casas_office"]
    for s in steps:
        print("==", s)
        try: globals()[s]()
        except Exception as e: issue(f"download step {s} failed: {e!r}")
