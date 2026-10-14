import numpy as np
from palace.dp.rr import randomise, debias_freq, channel, keep_prob
from palace.prefs.feedback_rr import Posterior, true_votes, simulate

def test_rr_unbiased():
    rng = np.random.default_rng(0); true = rng.choice(3, 200000, p=[0.2, 0.5, 0.3])
    for eps in (0.5, 1.0, 4.0):
        obs = np.bincount(randomise(true, eps, rng), minlength=3) / len(true)
        est = debias_freq(obs, eps); assert np.abs(est - np.array([0.2, 0.5, 0.3])).max() < 0.01

def test_channel_is_eps_ldp():
    C = channel(1.0); assert np.isclose(C.sum(1), 1).all(); assert C.max() / C.min() <= np.exp(1.0) + 1e-9
    assert keep_prob(np.inf) == 1.0

def test_posterior_uses_channel():
    # many noisy votes with eps=1: posterior mean should be closer to truth than prior (0), and eps=inf at least as good
    r_noisy = simulate("thermal", 1.0, 60, 150, 1)[0]; r_clean = simulate("thermal", np.inf, 60, 150, 1)[0]
    assert r_clean < r_noisy < 1.0 * 1.05   # prior sd is 1.0 -> error <= prior error
