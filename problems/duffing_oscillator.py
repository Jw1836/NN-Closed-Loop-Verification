"""Define a Duffing Oscillator problem.

This includes the NN Lyapunov function, the dynamics, and the region of interest.

Dynamics:
  x1_dot = x2
  x2_dot = -delta*x2 - alpha*x1 - beta*x1^3

Default parameters: delta = alpha = beta = 1.

Usage:
    python -m relu_vnn problems/duffing_oscillator.py
"""

import torch
from torch import nn, Tensor

from relu_vnn import LyapunovProblem

# Default Duffing parameters
DELTA: float = 1.0
ALPHA: float = 1.0
BETA: float = 1.0


class DuffingDynamics(nn.Module):
    def __init__(self, delta: float = 1.0, alpha: float = 1.0, beta: float = 1.0):
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
    def __init__(self, hidden_size: int):
        """Single hidden layer, ReLU activation, output shape (N, 1).

        The attribute is named `.network` so hyperplane.py can access
        `.network[0]` (input Linear) and `.network[2]` (output Linear).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.register_buffer("shift", torch.tensor(0.0))

    def forward(self, x):
        return self.network(x) - self.shift  # shape (N, 1)


class DuffingProblem(LyapunovProblem):
    """Define a Duffing Oscillator Problem.

    Requires region of interest:
        region (torch.Tensor) form [[x1_min, x1_max], [x2_min, x2_max]]
    Optional
        hidden_size (int) of the single layer
         delta, alpha, beta Duffing params, each defaults to 1.0
    """

    def __init__(
        self,
        region: Tensor,
        hidden_size: int,
        delta: float = DELTA,
        alpha: float = ALPHA,
        beta: float = BETA,
    ):
        super().__init__(
            nn_lyapunov=DuffingLyapunov(hidden_size=hidden_size),
            dynamics=DuffingDynamics(delta, alpha, beta),
            region=region,
        )


def make_problem(hidden_size: int = 100, **kwargs) -> LyapunovProblem:
    """Factory function for the CEGIS runner."""
    return DuffingProblem(
        region=torch.tensor([[-2.0, 2.0], [-2.0, 2.0]]),
        hidden_size=hidden_size,
        **kwargs,
    )
