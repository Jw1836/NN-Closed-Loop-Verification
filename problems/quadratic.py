"""Quadratic (linear stable) system with 1-norm Lyapunov function.

Dynamics:
    x1_dot = -x1
    x2_dot = -x2

Lyapunov function:
    V(x) = |x1| + |x2|  (exact ReLU representation)

dV/dt = sign(x1)·(-x1) + sign(x2)·(-x2) = -|x1| - |x2| < 0.

The network weights are fixed analytically — no training needed.

Usage:
    python -m relu_vnn verify problems/quadratic.py
"""

import torch
from torch import nn, Tensor

from relu_vnn import LyapunovProblem


class QuadraticDynamics(nn.Module):
    def forward(self, x):
        """f(x) = -x"""
        return -x


class QuadraticLyapunov(nn.Module):
    """V(x) = relu(x1) + relu(-x1) + relu(x2) + relu(-x2) = |x1| + |x2|."""

    def __init__(self):
        super().__init__()
        l1 = nn.Linear(2, 4, bias=True)
        l1.weight = nn.Parameter(
            torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            requires_grad=False,
        )
        l1.bias = nn.Parameter(torch.zeros(4), requires_grad=False)
        l2 = nn.Linear(4, 1, bias=False)
        l2.weight = nn.Parameter(
            torch.tensor([[1.0, 1.0, 1.0, 1.0]]), requires_grad=False
        )
        self.network = nn.Sequential(l1, nn.ReLU(), l2)
        self.register_buffer("shift", torch.tensor(0.0))

    def forward(self, x):
        return self.network(x) - self.shift


class QuadraticProblem(LyapunovProblem):
    def __init__(self, region: Tensor | None = None):
        if region is None:
            region = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
        super().__init__(
            nn_lyapunov=QuadraticLyapunov(),
            dynamics=QuadraticDynamics(),
            region=region,
        )


def make_problem(**kwargs) -> LyapunovProblem:
    return QuadraticProblem(**kwargs)
