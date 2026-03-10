"""Bilinear Oscillator.

Dynamics:
    x1_dot = -x1 + x1*x2
    x2_dot = -x2 - x1^2

- Bilinear coupling terms x1*x2, x1^2
- Unique equilibrium at origin
- Linearization: Df(0) = -I
- Global Lyapunov function: V = ||x||^2

Usage:
    python -m verify_relu_lyapunov problems/bilinear_oscillator.py
"""

import torch
from torch import nn, Tensor

from verify_relu_lyapunov import LyapunovProblem


class BilinearDynamics(nn.Module):
    def forward(self, x):
        """f(x) = [-x1 + x1*x2, -x2 - x1^2]"""
        x1, x2 = x[:, 0], x[:, 1]
        dx1 = -x1 + x1 * x2
        dx2 = -x2 - x1**2
        return torch.stack([dx1, dx2], dim=1)


class BilinearLyapunov(nn.Module):
    def __init__(self, hidden_size: int = 5):
        """Single hidden layer, ReLU activation, output shape (N, 1).

        The attribute is named `.network` so hyperplane.py can access
        `.network[0]` (input Linear) and `.network[2]` (output Linear).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.network(x)  # shape (N, 1)


class BilinearProblem(LyapunovProblem):
    """Bilinear Oscillator problem.

    Default region [-2, 2] x [-2, 2] — the phase portrait spirals inward
    quickly so a moderate domain suffices.
    """

    def __init__(self, region: Tensor | None = None, hidden_size: int = 30):
        if region is None:
            region = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
        super().__init__(
            nn_lyapunov=BilinearLyapunov(hidden_size=hidden_size),
            dynamics=BilinearDynamics(),
            region=region,
        )


def make_problem() -> LyapunovProblem:
    """Factory function for the CEGIS runner."""
    return BilinearProblem(hidden_size=30)
