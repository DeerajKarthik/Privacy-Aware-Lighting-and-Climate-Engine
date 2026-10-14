"""Solar geometry (pvlib) -> window solar gain per W of gain factor; daylight illuminance."""
from __future__ import annotations
import numpy as np, pandas as pd
import pvlib

def hourly_to_15min(h: pd.DataFrame) -> pd.DataFrame:
    """Open-Meteo radiation is a PRECEDING-hour mean: label shift to interval centre then linear interpolate."""
    r = h.copy(); r.index = r.index - pd.Timedelta(minutes=30)       # value represents (t-1h, t] -> centre at t-30min
    idx = pd.date_range(h.index[0], h.index[-1] + pd.Timedelta(minutes=45), freq="15min")
    r = r.reindex(r.index.union(idx)).interpolate(limit_direction="both").reindex(idx)
    return r

def window_gain_w(w15: pd.DataFrame, loc: dict, win_area, shgc, shading, azimuth, tilt=90):
    """Solar heat gain through window (W) = A*SHGC*shading*POA (isotropic-sky POA from DNI/DHI/GHI)."""
    times = w15.index.tz_localize(loc["tz"]) if w15.index.tz is None else w15.index
    sp = pvlib.solarposition.get_solarposition(times, loc["lat"], loc["lon"], altitude=loc.get("altitude_m", 0))
    dni = w15["direct_normal_irradiance"].values; dhi = w15["diffuse_radiation"].values; ghi = w15["shortwave_radiation"].values
    aoi = pvlib.irradiance.aoi(tilt, azimuth, sp["apparent_zenith"].values, sp["azimuth"].values)
    beam = dni * np.clip(np.cos(np.radians(aoi)), 0, None) * (sp["apparent_zenith"].values < 90)
    sky = dhi * (1 + np.cos(np.radians(tilt))) / 2; refl = ghi * 0.2 * (1 - np.cos(np.radians(tilt))) / 2
    poa = beam + sky + refl
    return win_area * shgc * shading * poa, ghi

def daylight_lux(ghi, efficacy, df, gradient_mean=1.0):
    """Workplane illuminance E_day = DF * E_ext, E_ext = GHI * luminous efficacy (lux)."""
    return df * gradient_mean * ghi * efficacy
