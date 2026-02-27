"""Define a Duffing Oscillator problem.

This includes the NN Lyapunov function, the dynamics, and the region of interest.

Dynamics:
  x1_dot = x2
  x2_dot = -delta*x2 - alpha*x1 - beta*x1^3

Default parameters: delta = alpha = beta = 1.
"""

import numpy as np
import torch
from torch import nn

# REGION OF INTEREST
X1_MIN: float = -10.0
X1_MAX: float = 10.0
X2_MIN: float = -10.0
X2_MAX: float = 10.0

# Default Duffing parameters
DELTA: float = 1.0
ALPHA: float = 1.0
BETA: float = 1.0


class DuffingDynamics(nn.Module):
    def __init__(self, delta: float = DELTA, alpha: float = ALPHA, beta: float = BETA):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward(self, x):
        """f(x) = [x2, -delta * x2 - alpha * x1 - beta * x1^3]"""
        x1, x2 = x[:, 0], x[:, 1]
        dx1 = x2
        dx2 = -self.delta * x2 - self.alpha * x1 - self.beta * x1**3
        return torch.stack([dx1, dx2], dim=1)


class DuffingLyapunov(nn.Module):
    def __init__(self, hidden_size: int = 5):
        """Single hidden layer, ReLU activation, output shape (N, 1).

        The attribute is named `.network` so hyperplane.py can access
        `.network[0]` (input Linear) and `.network[2]` (output Linear).
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)  # shape (N, 1)


# ── Numpy-style dynamics (used by hyperplane.py) ──────────────────────────────
# x is a column vector of shape (2, 1)


def duffing_f1(x, delta=DELTA, alpha=ALPHA, beta=BETA):
    return float(x[1, 0])


def duffing_f2(x, delta=DELTA, alpha=ALPHA, beta=BETA):
    return float(-delta * x[1, 0] - alpha * x[0, 0] - beta * x[0, 0] ** 3)


# ── Zero-level-set boundary functions (x2 as a function of x1) ────────────────
# Used by hyperplane.py to detect dynamics sign changes across polygon edges.


def duffing_f1_boundary(x1, delta=DELTA, alpha=ALPHA, beta=BETA):
    """f1 = 0  ↔  x2 = 0."""
    return 0.0


def duffing_f2_boundary(x1, delta=DELTA, alpha=ALPHA, beta=BETA):
    """f2 = 0  ↔  x2 = -(alpha/delta)*x1 - (beta/delta)*x1^3."""
    return -(alpha / delta) * x1 - (beta / delta) * x1**3


# ── Torch-style component dynamics (used by lyapunov.py training) ─────────────
# x is a batch tensor of shape (N, 2); each returns shape (N, 1).


def duffing_f1_torch(x, delta=DELTA, alpha=ALPHA, beta=BETA):
    return x[:, 1:2]


def duffing_f2_torch(x, delta=DELTA, alpha=ALPHA, beta=BETA):
    x1, x2 = x[:, 0:1], x[:, 1:2]
    return -delta * x2 - alpha * x1 - beta * x1**3


# ── Analytic Lyapunov function (for reference / initial training target) ───────


def duffing_analytic_V(x1, x2, delta=DELTA, alpha=ALPHA, beta=BETA):
    """Quadratic Lyapunov function from linearisation."""
    A = (alpha**2 + alpha + delta**2) / (2 * delta * alpha)
    B = 1.0 / (2 * alpha)
    C = (alpha + 1) / (2 * delta * alpha)
    P = np.array([[A, B], [B, C]])
    v = np.array([[x1], [x2]])
    return float(v.T @ P @ v)
