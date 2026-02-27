"""Defines the LyapunovProblem dataclass — the common interface
that verifiers and the Lyapunov network are handed."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np


@dataclass
class LyapunovProblem:
    """Specification of a closed-loop system to verify stability for.

    Parameters
    ----------
    f : list of callables
        Dynamics functions [f1, f2, ...].  Each fi takes a column vector
        x (shape (n,1)) and returns a scalar xdot_i.
    domain : tuple of (min, max) pairs
        Bounding box for the region of interest,
        e.g. ((-3, 3), (-3, 3)) for 2-D.
    x_star : np.ndarray
        Equilibrium point (shape (n,) or (n,1)).
    dim : int
        State dimension.
    analytic_V : callable or None
        If available, the true Lyapunov function V(x1, ..., xn) -> float.
    analytic_V_gradient : callable or None
        Gradient of analytic_V, returns (n,1) array.
    zero_level_set_funcs : list of callables or None
        For 2-D problems: functions mapping x1 -> x2 that trace where
        each fi = 0.  Used by the hyperplane verifier to detect sign
        changes across polygon edges.
    """

    f: List[Callable]
    domain: Tuple[Tuple[float, float], ...]
    x_star: np.ndarray
    dim: int
    analytic_V: Optional[Callable] = None
    analytic_V_gradient: Optional[Callable] = None
    zero_level_set_funcs: Optional[List[Callable]] = None

    def f_eval(self, x):
        """Evaluate full dynamics vector at state x (shape (n,1)).
        Returns np.ndarray of shape (n,1)."""
        return np.array([fi(x) for fi in self.f]).reshape(-1, 1)


# ── Pre-built problem definitions ──────────────────────────────────────────


def make_duffing_problem(alpha=1.0, delta=1.0, beta=1.0, domain=((-3, 3), (-3, 3))):
    """Duffing oscillator: x1dot = x2,  x2dot = -delta*x2 - alpha*x1 - beta*x1^3"""

    def f1(x):
        return x[1, 0]

    def f2(x):
        return -delta * x[1, 0] - alpha * x[0, 0] - beta * x[0, 0] ** 3

    A = (alpha**2 + alpha + delta**2) / (2 * delta * alpha)
    B = 1 / (2 * alpha)
    C = (alpha + 1) / (2 * delta * alpha)
    P = np.array([[A, B], [B, C]])

    def analytic_V(x1, x2):
        xv = np.array([[x1], [x2]])
        return float(xv.T @ P @ xv)

    def analytic_V_gradient(x1, x2):
        xv = np.array([[x1], [x2]])
        return 2 * P @ xv

    def f1_boundary(x1):
        return 0

    def f2_boundary(x1):
        return -(alpha / delta) * x1 - (beta / delta) * x1**3

    return LyapunovProblem(
        f=[f1, f2],
        domain=domain,
        x_star=np.array([0.0, 0.0]),
        dim=2,
        analytic_V=analytic_V,
        analytic_V_gradient=analytic_V_gradient,
        zero_level_set_funcs=[f1_boundary, f2_boundary],
    )
