"""Simple Pendulum System.

Dynamics:
    $mL\ddot{\theta} = -mg sin(\theta) - \eta L \dot{\theta}

    m is the mass in grams
    L is the length of the pendulum in meters
    theta is the angle off down vertical
    eta is a resistance parameter

State Definition:
    x1 is angle theta of pendulum from
    x2 is dot theta

State dynamics:
    x1_dot = x2
    x2_dot = -g/L * sin(x1) - \eta/m * x2

Equilibrium points:
    x_e1 = [0 0]' and is stable
    x_e2 = [\pi 0]' and is unstable

Usage:
    python -m verify_relu_lyapunov problems/pendulum.py
"""

from typing import Any

import torch
from torch import nn, Tensor

from verify_relu_lyapunov import LyapunovProblem


class PendulumDynamics(nn.Module):
    G = 9.81  # Gravity constant

    def __init__(self, m: float = 1.0, L: float = 1.0, eta: float = 0.5) -> None:
        super().__init__()
        self.m = m
        self.L = L
        self.eta = eta

    def forward(self, x: Tensor) -> Tensor:
        """
        x[..., 0] = theta, x[..., 1] = theta_dot
        """
        theta = x[..., 0]
        theta_dot = x[..., 1]
        x1_dot = theta_dot
        x2_dot = -(self.G / self.L) * torch.sin(theta) - (self.eta / self.m) * theta_dot
        return torch.stack([x1_dot, x2_dot], dim=-1)


class PendulumLyapunov(nn.Module):
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


class PendulumProblem(LyapunovProblem):
    """Define a simple pendulum problem.

    If `region` is not provided, a default is constructed from the dynamics
    parameters:
        theta  (x1): [-pi/2, pi/2]  — stay away from the unstable equilibrium
        theta_dot (x2): [-sqrt(g/L), sqrt(g/L)]  — natural velocity scale

    Additional keyword arguments (m, L, eta) are forwarded to PendulumDynamics.
    """

    def __init__(
        self,
        region: Tensor | None = None,
        hidden_size: int = 5,
        **kwargs: Any,
    ) -> None:
        import math

        dynamics = PendulumDynamics(**kwargs)
        # Intelligently construct the region if not explicitly given
        if region is None:
            theta_lim = math.pi / 2
            theta_dot_lim = math.sqrt(PendulumDynamics.G / dynamics.L)
            region = torch.tensor(
                [[-theta_lim, theta_lim], [-theta_dot_lim, theta_dot_lim]]
            )
        super().__init__(
            nn_lyapunov=PendulumLyapunov(hidden_size=hidden_size),
            dynamics=dynamics,
            region=region,
        )


def make_problem(hidden_size: int = 30, **kwargs) -> LyapunovProblem:
    """Factory function for the CEGIS runner."""
    return PendulumProblem(hidden_size=hidden_size, **kwargs)
