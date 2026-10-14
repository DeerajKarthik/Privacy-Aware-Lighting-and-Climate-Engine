"""Stage: parse raw CASAS (streamed from zip) -> sparse bins cache -> dense features, site summary, splits.
Usage: 02_build_features.py [--hold 15] [--sites a,b]"""
import argparse, io, json, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from palace.data.stream import aggregate
from palace.data.features import densify, add_calendar, chrono_splits

CFG = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
INT = ROOT / "data/interim"; PROC = ROOT / "data/processed/sites"; INT.mkdir(parents=True, exist_ok=True); PROC.mkdir(parents=True, exist_ok=True)

def sparse_sources(only=None):
    hz = ROOT / "data/raw/casas_home/new_labeled_data.zip"
    for member, site in [("aruba.txt", "aruba"), ("cairo.txt", "cairo"), ("milan.txt", "milan"), ("tulum1.txt", "tulum1"), ("tulum2.txt", "tulum2")]:
        yield site, "home", hz, member
    oz = ROOT / "data/raw/casas_office/officedata.zip"
    if oz.exists() and zipfile.is_zipfile(oz):
        for i in zipfile.ZipFile(oz).infolist():
            n = Path(i.filename).name
            if n.endswith((".txt", ".csv")) and not n.startswith(".") and i.file_size > 100_000: yield Path(n).stem, "office", oz, i.filename

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hold", type=int, default=CFG["features"]["presence_hold_min"]); ap.add_argument("--sites")
    a = ap.parse_args(); only = a.sites.split(",") if a.sites else None; F = CFG["features"]; summ = []
    for site, kind, zpath, member in sparse_sources():
        if only and site not in only: continue
        cache = INT / f"{site}_sparse.parquet"
        if not cache.exists():
            with zipfile.ZipFile(zpath).open(member) as f: sp = aggregate(io.TextIOWrapper(f, encoding="utf-8", errors="ignore"))
            sp.to_parquet(cache)
        else: sp = pd.read_parquet(cache)
        d, days = densify(sp, a.hold, F["outage_gap_hours"], F["max_masked_frac_day"]); d = add_calendar(d)
        min_days = F["min_usable_days_site"] if kind == "office" else F["min_usable_days_home"]
        usable = days.index[days.usable]
        sp_ = chrono_splits(usable, CFG["splits"]["train"], CFG["splits"]["val"], CFG["splits"]["gap_days"])
        d["split"] = "none"
        for k, v in sp_.items(): d.loc[d.index.floor("D").isin(v), "split"] = k
        d["site"] = site
        if a.hold == F["presence_hold_min"]:
            d.drop(columns=["activity"], errors="ignore").assign(activity=d["activity"].astype("string")).to_parquet(PROC / f"{site}.parquet")
            days.to_parquet(PROC / f"{site}_days.parquet")
        summ.append(dict(site=site, kind=kind, first=str(d.index[0].date()), last=str(d.index[-1].date()), calendar_days=len(days), usable_days=int(days.usable.sum()),
            pct_masked=round(100 * (1 - d["mask"].mean()), 2), mean_presence=round(days.loc[days.usable, "mean_presence"].mean(), 4),
            arrival_mean_h=round(days.first_arrival_h[days.usable].mean(), 2), arrival_sd_h=round(days.first_arrival_h[days.usable].std(), 2),
            departure_mean_h=round(days.last_departure_h[days.usable].mean(), 2), n_train=len(sp_["train"]), n_val=len(sp_["val"]), n_test=len(sp_["test"]),
            meets_min_days=bool(days.usable.sum() >= min_days), hold_min=a.hold))
        print(summ[-1])
    out = ROOT / "data/processed" / ("site_summary.csv" if a.hold == F["presence_hold_min"] else f"site_summary_hold{a.hold}.csv")
    pd.DataFrame(summ).to_csv(out, index=False)
if __name__ == "__main__": main()
