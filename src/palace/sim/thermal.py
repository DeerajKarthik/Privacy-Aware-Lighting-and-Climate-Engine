"""2R2C zone, exact discretisation (matrix exponential, zero-order hold on inputs).
 Ca dTa/dt = (To-Ta)/Rao + (Tm-Ta)/Ram + Qhvac + Qair
 Cm dTm/dt = (Ta-Tm)/Ram + Qmass
Inputs u = [To, Qhvac, Qair, Qmass]. Infiltration is added to the Qair input by the caller (explicit)."""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm

def continuous(Ca, Cm, Rao, Ram):
    A = np.array([[-(1 / Rao + 1 / Ram) / Ca, 1 / (Ram * Ca)], [1 / (Ram * Cm), -1 / (Ram * Cm)]])
    B = np.array([[1 / (Rao * Ca), 1 / Ca, 1 / Ca, 0.0], [0.0, 0.0, 0.0, 1 / Cm]])
    return A, B

def discretise(A, B, dt):
    n, m = B.shape; M = np.zeros((n + m, n + m)); M[:n, :n] = A * dt; M[:n, n:] = B * dt
    E = expm(M); return E[:n, :n], E[:n, n:]

class Thermal:
    def __init__(self, Ca, Cm, Rao, Ram, dt=900.0):
        self.p = (Ca, Cm, Rao, Ram); self.dt = dt
        self.A, self.B = continuous(Ca, Cm, Rao, Ram); self.Ad, self.Bd = discretise(self.A, self.B, dt)
    def step(self, x, u):
        """x: (...,2) [Ta, Tm]; u: (...,4). Broadcasts over leading dims."""
        return x @ self.Ad.T + u @ self.Bd.T
