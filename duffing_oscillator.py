"""Define a Duffing Oscillator problem.

This includes the NN Lyapunov function, the dynamics, and the region of interest.
"""

from torch import nn
import torch

# REGION OF INTEREST
X1_MIN: float = -10.0
X1_MAX: float = 10.0
X2_MIN: float = -10.0
X2_MAX: float = 10.0


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
    def __init__(self, hidden_size: int = 5):
        """Single hidden layer, ReLU activation, output is scalar."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()
