import numpy as np, pandas as pd, pytest
from pathlib import Path
from palace.data.stream import aggregate
from palace.data.features import densify, chrono_splits

PROC = Path(__file__).resolve().parents[1] / "data/processed/sites"

def test_aggregate_message_classification():
    lines = ["2020-01-01,00:01:00.1,WorkArea,ON", "2020-01-01,00:02:00.1,WorkArea,OFF", "2020-01-01,00:03:00.1,Door,OPEN",
             "2020-01-01,00:20:00.1,T001,21.5", "2020-01-01 00:21:00 M001 ON Sleeping begin"]
    d = aggregate(lines)
    assert d.loc[pd.Timestamp("2020-01-01 00:00"), "motion_count"] == 1
    assert d.loc[pd.Timestamp("2020-01-01 00:00"), "door_events"] == 1
    assert d.loc[pd.Timestamp("2020-01-01 00:15"), "motion_count"] == 1   # temp reading ignored, M001 ON counted

def test_outage_is_masked_not_absent():
    idx = pd.to_datetime(["2020-01-01 10:00", "2020-01-05 10:00"])
    sp = pd.DataFrame({"motion_count": [3, 2], "active_sensors": [1, 1], "door_events": [0, 0], "event_count": [3, 2], "activity": [None, None]}, index=idx)
    d, days = densify(sp, hold_min=15, outage_gap_hours=24)
    assert d.loc["2020-01-03 03:00", "mask"] == 0          # 4-day gap -> missing
    assert d.loc["2020-01-03 03:00", "presence"] == 0 and d.loc["2020-01-01 10:15", "mask"] == 0   # conservative: whole gap masked
    # hold time on a SHORT gap (not an outage): presence persists one bin, then drops
    sp2 = sp.iloc[:1].copy(); sp2.loc[pd.Timestamp("2020-01-01 12:00")] = [1, 1, 0, 1, None]
    d2, _ = densify(sp2, hold_min=15, outage_gap_hours=24)
    assert d2.loc["2020-01-01 10:15", "presence"] == 1 and d2.loc["2020-01-01 10:30", "presence"] == 0 and d2.loc["2020-01-01 10:30", "mask"] == 1

def test_split_disjoint_with_gap():
    days = pd.date_range("2020-01-01", periods=300)
    s = chrono_splits(days, 0.7, 0.15, gap_days=7)
    for a, b in [("train", "val"), ("val", "test")]:
        assert s[a].max() + pd.Timedelta(days=7) <= s[b].min()
    assert len(set(s["train"]) & set(s["test"])) == 0

@pytest.mark.skipif(not PROC.exists() or not list(PROC.glob("*.parquet")), reason="processed data absent")
def test_processed_sites():
    for p in PROC.glob("*[!s].parquet"):
        if p.stem.endswith("_days"): continue
        d = pd.read_parquet(p)
        assert len(d) % 96 == 0
        v = d[d["mask"] == 1]
        assert not v[["motion_count", "active_sensors", "door_events", "presence", "hour_sin"]].isna().any().any()
        sub = d[d.split != "none"]
        days = {k: set(sub.index[sub.split == k].floor("D")) for k in ("train", "val", "test")}
        for a in days:
            for b in days:
                if a < b: assert not (days[a] & days[b])
