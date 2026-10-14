"""Build simulator inputs for a (site, weather location/year, split) by replaying site-days onto a weather year.
Replay rule (DECISIONS D007/D013): each weather date takes a randomly drawn usable site-day from `split` with the SAME weekday and
month (fallback: same weekday). The controller never sees weather-year/occupancy-year pairing as real; it is a SYNTHETIC pairing."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from palace.sim import daylight
from palace.sim.occupants import n_people

def load_site(root: Path, site: str):
    d = pd.read_parquet(root / f"data/processed/sites/{site}.parquet"); days = pd.read_parquet(root / f"data/processed/sites/{site}_days.parquet")
    return d, days

def replay_days(d, split, dates: pd.DatetimeIndex, seed=0):
    rng = np.random.default_rng(seed)
    v = d[(d.split == split)]
    # usable = all masked==1 in the day
    full = v.groupby(v.index.floor("D"))["mask"].min(); ok = full[full == 1].index
    pool = pd.DataFrame({"day": ok, "dow": ok.dayofweek, "month": ok.month})
    if len(pool) == 0: raise ValueError("no usable days in split")
    pick = []
    for dt in dates:
        c = pool[(pool.dow == dt.dayofweek) & (pool.month == dt.month)]
        if len(c) == 0: c = pool[pool.dow == dt.dayofweek]
        if len(c) == 0: c = pool
        pick.append(c.day.values[rng.integers(len(c))])
    return pd.DatetimeIndex(pick)

def build_inputs(root, cfg, site, loc, wyear, split="train", seed=0, n_days=None, start_day=0, met_office=None, days_override=None):
    from palace.data import weather as W
    d, _ = load_site(root, site)
    h = W.load(loc, wyear, root / "data/raw/weather"); w15 = daylight.hourly_to_15min(h)
    w15 = w15[w15.index.year == wyear]
    dates = pd.DatetimeIndex(sorted(set(w15.index.floor("D"))))[start_day:(start_day + n_days if n_days else None)]
    src = replay_days(d, split, dates, seed) if days_override is None else days_override
    pres = np.concatenate([d.loc[s:s + pd.Timedelta("23h45min"), "presence"].values for s in src]).astype(float)
    sens = np.concatenate([d.loc[s:s + pd.Timedelta("23h45min"), "active_sensors"].values for s in src]).astype(float)
    mot = np.concatenate([d.loc[s:s + pd.Timedelta("23h45min"), "motion_count"].values for s in src]).astype(float)
    w15 = w15.loc[dates[0]:dates[-1] + pd.Timedelta("23h45min")]
    assert len(w15) == len(pres), (len(w15), len(pres))
    L = cfg["location"][loc]; win_area = cfg["window"]["area_frac_of_floor"] * cfg["zone"]["floor_area_m2"]
    sol, ghi = daylight.window_gain_w(w15, L, win_area, cfg["window"]["shgc"], cfg["window"]["shading_factor"], cfg["window"]["azimuth_deg"], cfg["window"]["tilt_deg"])
    npl = n_people(pres, sens, cfg["people"]["capacity_per_zone"], cfg["people"]["people_per_active_sensor"])
    met = np.full(len(pres), met_office or cfg["people"]["met_office"])
    return dict(To=w15["temperature_2m"].values, solar_w=sol, ghi=ghi, n_people=npl, presence=pres, motion_count=mot, met=met,
                index=w15.index, win_frac_ref=cfg["window"]["area_frac_of_floor"], rh=w15["relative_humidity_2m"].values)
