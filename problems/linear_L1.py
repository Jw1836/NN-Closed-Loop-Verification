"""Linear stable system with 1-norm Lyapunov function.

Dynamics:
    dx/dt = -x  (n-dimensional, linear decay)

Lyapunov function:
    V(x) = ||x||_1 = sum_i |x_i|  (1-norm, exact ReLU representation)

dV/dt = sum_i sign(x_i)·(-x_i) = -sum_i |x_i| = -V(x) < 0.

The network weights are fixed analytically — no training needed.

Usage:
    python -m relu_vnn verify problems/linear_l1.py
    python -m relu_vnn verify problems/linear_l1.py --n 4
"""

import torch
from torch import nn, Tensor

from relu_vnn import LyapunovProblem


class LinearDecayDynamics(nn.Module):
    def forward(self, x):
        """f(x) = -x"""
        return -x


class L1NormLyapunov(nn.Module):
    """V(x) = sum_i (relu(x_i) + relu(-x_i)) = sum_i |x_i| = ||x||_1."""

    def __init__(self, n: int):
        super().__init__()
        w1 = torch.zeros(2 * n, n)
        for i in range(n):
            w1[2 * i, i] = 1.0
            w1[2 * i + 1, i] = -1.0
        l1 = nn.Linear(n, 2 * n)
        l1.weight = nn.Parameter(w1, requires_grad=False)
        l1.bias = nn.Parameter(torch.zeros(2 * n), requires_grad=False)
        l2 = nn.Linear(2 * n, 1, bias=False)
        l2.weight = nn.Parameter(torch.ones(1, 2 * n), requires_grad=False)
        self.network = nn.Sequential(l1, nn.ReLU(), l2)

    def forward(self, x):
        return self.network(x)


class LinearL1Problem(LyapunovProblem):
    def __init__(self, hidden_size: int, region: Tensor | None = None):
        if region is None:
            region = torch.full((hidden_size, 2), -3.0)
            region[:, 1] = 3.0
        super().__init__(
            nn_lyapunov=L1NormLyapunov(n=hidden_size),
            dynamics=LinearDecayDynamics(),
            region=region,
        )


def make_problem(hidden_size: int = 2) -> LyapunovProblem:
    return LinearL1Problem(hidden_size=hidden_size)
