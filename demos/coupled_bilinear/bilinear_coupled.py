"""Coupled Bilinear Oscillators.

Dynamics: two bilinear oscillators coupled through their x2 states.

    x1_dot = -x1 + x1*x2
    x2_dot = -x2 - x1^2 + eps*x3
    x3_dot = -x3 + x3*x4
    x4_dot = -x4 - x3^2 + eps*x1

- Each subsystem is a bilinear oscillator (same structure as BilinearOscillator)
- Coupling: subsystem 2's x2 is nudged by eps*x1, and vice versa
- Equilibrium at origin
- Coupling strength eps defaults to 0.1

Usage:
    python -m relu_vnn problems/bilinear_coupled.py
"""

import torch
from torch import nn, Tensor

from relu_vnn import LyapunovProblem


class BilinearCoupledDynamics(nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """f(x) = [-x1 + x1*x2, -x2 - x1^2 + eps*x3, -x3 + x3*x4, -x4 - x3^2 + eps*x1]"""
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        dx1 = -x1 + x1 * x2
        dx2 = -x2 - x1**2 + self.eps * x3
        dx3 = -x3 + x3 * x4
        dx4 = -x4 - x3**2 + self.eps * x1
        return torch.stack([dx1, dx2, dx3, dx4], dim=1)


class BilinearCoupledLyapunov(nn.Module):
    def __init__(self, hidden_size: int = 5):
        """Single hidden layer, ReLU activation, output shape (N, 1).

        The attribute is named `.network` so hyperplane.py can access
        `.network[0]` (input Linear) and `.network[2]` (output Linear).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.register_buffer("shift", torch.tensor(0.0))

    def forward(self, x):
        return self.network(x) - self.shift  # shape (N, 1)


class BilinearCoupledProblem(LyapunovProblem):
    """Coupled Bilinear Oscillators problem.

    Default region [-2, 2]^4.
    """

    def __init__(
        self,
        region: Tensor | None = None,
        hidden_size: int = 30,
        eps: float = 0.1,
    ):
        if region is None:
            region = torch.tensor([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])
        super().__init__(
            nn_lyapunov=BilinearCoupledLyapunov(hidden_size=hidden_size),
            dynamics=BilinearCoupledDynamics(eps=eps),
            region=region,
        )


def make_problem(hidden_size: int = 30, **kwargs) -> LyapunovProblem:
    """Factory function for the verification runner."""
    return BilinearCoupledProblem(hidden_size=hidden_size, **kwargs)
