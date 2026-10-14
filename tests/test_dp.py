import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.analysis import rdp as R
from palace.dp.accountant import hand_rdp_epsilon, cumulative_epsilon, delta_for

def test_accountant_matches_hand_rdp_q1():
    sigma, T, delta = 5.0, 10, 1e-5
    acc = RDPAccountant()
    for _ in range(T): acc.step(noise_multiplier=sigma, sample_rate=1.0)
    alphas = acc.DEFAULT_ALPHAS
    assert abs(acc.get_epsilon(delta) - hand_rdp_epsilon(sigma, T, delta, alphas)) < 1e-6

def test_cumulative_over_rounds_equals_single_run():
    a, b = RDPAccountant(), RDPAccountant()
    for _ in range(12): a.step(noise_multiplier=1.1, sample_rate=0.05)
    for _ in range(3):
        for _ in range(4): b.step(noise_multiplier=1.1, sample_rate=0.05)       # 3 'rounds' x 4 local steps on ONE persistent accountant
    assert abs(cumulative_epsilon(a, 1e-4) - cumulative_epsilon(b, 1e-4)) < 1e-12

def test_delta_rule():
    assert delta_for(500) == 1 / 5000
