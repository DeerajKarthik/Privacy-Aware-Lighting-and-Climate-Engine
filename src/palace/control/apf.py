"""Actuation Privacy Filter: exponential mechanism over the FULL discretised first-action set (1,089 actions).
u(a) = 1 - J~(a) in [0,1]. Sensitivity w.r.t. the presence bit y_t (which enters ONLY through the step-0 expected-discomfort weight):
  |u(a; y=1) - u(a; y=0)| <= w_c * D_max / (H * (w_e + w_c + w_s)) =: Delta_u   (D_max = 1)
Release P(a) ∝ exp(eps * u(a) / (2 * Delta_u)) => eps-DP w.r.t. y_t at that step (event level), CONDITIONAL on the state/forecast inputs.
NOT covered: indirect leakage through the physical state / meter / later steps, and whole-trajectory guarantees against side information."""
from __future__ import annotations
import numpy as np

def sensitivity(w, H):
    return w["w_c"] * 1.0 / (H * (w["w_e"] + w["w_c"] + w["w_s"]))

def release(u, eps, delta_u, rng):
    if np.isinf(eps): return int(np.argmax(u))
    z = eps * (u - u.max()) / (2.0 * delta_u); p = np.exp(z); p /= p.sum(); return int(rng.choice(len(u), p=p))

def composition(eps_act, T, delta=1e-5):
    """Per-step eps over T steps: basic = T*eps; advanced (Dwork-Rothblum-Vadhan) = eps*sqrt(2T ln(1/delta)) + T*eps*(e^eps-1);
    zCDP: rho = T*eps^2/8 per EM (Cesar-Rogers bounded-range), eps = rho + 2 sqrt(rho ln(1/delta))."""
    if np.isinf(eps_act): return dict(basic=np.inf, advanced=np.inf, zcdp=np.inf)
    adv = eps_act * np.sqrt(2 * T * np.log(1 / delta)) + T * eps_act * (np.exp(eps_act) - 1); rho = T * eps_act ** 2 / 8
    return dict(basic=T * eps_act, advanced=float(adv), zcdp=float(rho + 2 * np.sqrt(rho * np.log(1 / delta))))
