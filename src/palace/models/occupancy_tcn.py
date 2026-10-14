"""Causal dilated 1-D TCN (per-timestep channel LayerNorm, no BatchNorm -> DP-compatible, strictly causal). Receptive field = 1 + 2*sum(dilations) = 171 bins <= 192 context."""
import torch, torch.nn as nn, torch.nn.functional as F
from palace.fl.data import CTX, NORIG, H, C_IN, BPD

class ChanNorm(nn.Module):
    """LayerNorm over channels at each timestep (strictly causal; GroupNorm over time would leak future-bin statistics). DP-compatible (no batch stats)."""
    def __init__(s, c): super().__init__(); s.ln = nn.LayerNorm(c)
    def forward(s, x): return s.ln(x.transpose(1, 2)).transpose(1, 2)

class Block(nn.Module):
    def __init__(s, c, k, d):
        super().__init__(); s.pad = (k - 1) * d; s.c1 = nn.Conv1d(c, c, k, dilation=d); s.n1 = ChanNorm(c); s.c2 = nn.Conv1d(c, c, k, dilation=d); s.n2 = ChanNorm(c)
    def forward(s, x):
        y = F.gelu(s.n1(s.c1(F.pad(x, (s.pad, 0))))); y = s.n2(s.c2(F.pad(y, (s.pad, 0)))); return F.gelu(x + y)

class OccTCN(nn.Module):
    """forward(x [B,T=288,C]) -> (presence logits [B,88,8], arrival-hour/24 prediction [B])"""
    def __init__(s, c_in=C_IN, ch=32, k=3, dil=(1, 4, 16, 64)):
        super().__init__(); s.inp = nn.Conv1d(c_in, ch, 1); s.blocks = nn.ModuleList([Block(ch, k, d) for d in dil])
        s.head = nn.Linear(ch, H); s.arr = nn.Linear(ch, 1)
    def forward(s, x):
        h = s.inp(x.transpose(1, 2))
        for b in s.blocks: h = b(h)
        h = h.transpose(1, 2)                         # [B,T,ch]
        logits = s.head(h[:, CTX:CTX + NORIG])        # origins at positions CTX..CTX+87
        arr = torch.sigmoid(s.arr(h[:, CTX - 1])).squeeze(-1)   # from the last context bin (midnight)
        if getattr(s, "_all", False): return s.head(h[:, CTX - 1:CTX + BPD - 1])   # [B,96,8]: origin = bin b-1 for control step b=0..95
        return logits, arr
    def forward_all(s, x):
        s._all = True
        try: return s.forward(x)
        finally: s._all = False

def n_params(m): return sum(p.numel() for p in m.parameters())

def loss_fn(out, Y, M, A, focal_w=0.25, gamma=2.0):
    """Per-day-averaged loss, so that each record (site-day) has a single bounded contribution (needed for DP clipping).
    Returns a [B] loss when called via per_sample=True semantics by Opacus (mean over batch applied by caller)."""
    logits, arr = out
    bce = F.binary_cross_entropy_with_logits(logits, Y, reduction="none")
    p = torch.sigmoid(logits); pt = Y * p + (1 - Y) * (1 - p); foc = ((1 - pt) ** gamma) * bce
    w = M; den = w.sum((1, 2)).clamp_min(1.0)
    l_pres = ((bce + focal_w * foc) * w).sum((1, 2)) / den
    l_arr = F.smooth_l1_loss(arr, A[:, 0], reduction="none", beta=0.05) * A[:, 1]
    return (l_pres + 0.5 * l_arr).mean()
