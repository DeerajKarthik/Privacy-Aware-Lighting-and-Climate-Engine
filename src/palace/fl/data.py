"""Site-day records. Record d = inputs over [d-48h, d+24h) (288 bins), labels ONLY from day d (first 88 origins; next-8-bin targets stay inside day d),
arrival label = first arrival hour of day d (predicted from the representation at midnight, i.e. 'next-day arrival')."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch

CTX, H, BPD = 192, 8, 96
NORIG = BPD - H      # 88 origins per day
C_IN = 9

def site_features(d: pd.DataFrame) -> np.ndarray:
    return np.stack([d.presence.values, np.log1p(d.motion_count.values), np.log1p(d.active_sensors.values), np.log1p(d.door_events.values),
                     d.hour_sin.values, d.hour_cos.values, d.dow_sin.values, d.dow_cos.values, d["mask"].values], 1).astype(np.float32)

def build_records(root: Path, site: str, split: str, max_days: int | None = None, days_list=None):
    d = pd.read_parquet(root / f"data/processed/sites/{site}.parquet"); days = pd.read_parquet(root / f"data/processed/sites/{site}_days.parquet")
    F = site_features(d); F_masked = F.copy(); m = d["mask"].values.astype(bool); F_masked[~m, :8] = 0.0   # hide data in masked bins
    pres = d.presence.values.astype(np.float32); mk = d["mask"].values.astype(np.float32)
    dayidx = pd.Series(np.arange(len(d)), index=d.index)
    if days_list is not None: sel = list(pd.DatetimeIndex(days_list))
    else:
        sel = d.index[(d.split == split)].floor("D").unique(); sel = [s for s in sel if days.loc[s, "usable"]]
        if max_days: sel = sel[-max_days:]
    X = np.zeros((len(sel), CTX + BPD, C_IN), np.float32); Y = np.zeros((len(sel), NORIG, H), np.float32); M = np.zeros((len(sel), NORIG, H), np.float32)
    A = np.zeros((len(sel), 2), np.float32)
    for i, s in enumerate(sel):
        st = dayidx[s]; a = st - CTX
        lo = max(a, 0); X[i, lo - a:] = F_masked[lo:st + BPD]
        if a < 0: X[i, :-a, 8] = 0.0                              # padded history is flagged masked
        for o in range(NORIG):
            Y[i, o] = pres[st + o + 1: st + o + 1 + H]; M[i, o] = mk[st + o + 1: st + o + 1 + H]
        fa = days.loc[s, "first_arrival_h"]
        if np.isfinite(fa): A[i] = (fa / 24.0, 1.0)
    return dict(X=torch.from_numpy(X), Y=torch.from_numpy(Y), M=torch.from_numpy(M), A=torch.from_numpy(A), days=pd.DatetimeIndex(sel), site=site)

class DayDataset(torch.utils.data.Dataset):
    def __init__(self, rec, device): self.X, self.Y, self.M, self.A = (rec[k].to(device) for k in "XYMA")
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.M[i], self.A[i]
