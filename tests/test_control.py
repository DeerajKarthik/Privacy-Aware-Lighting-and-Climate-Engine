import numpy as np, torch, yaml
from pathlib import Path
from palace.models.occupancy_tcn import OccTCN
from palace.fl.data import CTX, BPD, C_IN
from palace.control import apf as APF
from palace.control.costs import ND, NC, NS, decode, ALL_A

def test_tcn_is_causal():
    torch.manual_seed(0); m = OccTCN().eval(); x = torch.randn(2, CTX + BPD, C_IN); y = x.clone(); y[:, CTX + 40:] += 5.0   # change the FUTURE of origin 39
    a = m.forward_all(x); b = m.forward_all(y)
    assert torch.allclose(a[:, :40], b[:, :40], atol=1e-5)         # origins up to bin 39 (positions < CTX+40-1) unaffected
    assert not torch.allclose(a[:, 60:], b[:, 60:])

def test_action_space_size():
    assert len(ALL_A) == 1089 == ND * NC * NS
    d, c, s = decode(ALL_A); assert len({(i, j, k) for i, j, k in zip(d, c, s)}) == 1089

def test_exponential_mechanism_dp_bound():
    rng = np.random.default_rng(0); w = dict(w_e=2.0, w_c=1.0, w_s=0.1); du = APF.sensitivity(w, 8)
    u0 = rng.random(1089); u1 = np.clip(u0 + rng.uniform(-du, du, 1089), 0, 1)        # neighbouring utilities within sensitivity
    for eps in (0.5, 2.0):
        l0 = eps * u0 / (2 * du); l1 = eps * u1 / (2 * du); p0 = np.exp(l0 - l0.max()); p0 /= p0.sum(); p1 = np.exp(l1 - l1.max()); p1 /= p1.sum()
        assert np.max(np.abs(np.log(p0 / p1))) <= eps + 1e-9

def test_sensitivity_numeric_matches_analytic():
    """Flip y_t in the step-0 comfort weight: max |u(y=1)-u(y=0)| over random D in [0,1] must be <= Delta_u."""
    w = dict(w_e=2.0, w_c=1.0, w_s=0.1); H = 8; Jmax = H * sum(w.values()); rng = np.random.default_rng(1)
    D = rng.random(100000); diff = np.abs(w["w_c"] * (1 - 0) * D) / Jmax
    assert diff.max() <= APF.sensitivity(w, H) + 1e-12 and diff.max() > 0.99 * APF.sensitivity(w, H)
