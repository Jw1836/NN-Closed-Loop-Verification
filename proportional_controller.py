"""Proportional controller for the dot dynamics system (xdot = 1 - x^2 + u).

Contains:
- DotDynamics: the plant (RK4 integrator)
- ProportionalController: analytic P-controller that logs training data
- ProportionalControllerNet: NN trained to mimic the P-controller
- train(): training loop for the NN
"""

import os
import torch
from torch import nn
from scipy.integrate import solve_ivp

MODEL_FILE = os.path.join(os.path.dirname(__file__), "proportional_controller.pth")


# ── Plant ──────────────────────────────────────────────────────────────────


class DotDynamics:
    """Nonlinear scalar system: xdot = 1 - x^2 + u, integrated via RK45."""

    def __init__(self, x0, dt=0.01):
        self.state = float(x0)
        self.dt = dt

    def update(self, u):
        """Advance the state by one timestep dt under constant input u.

        Integrates xdot = 1 - x^2 + u over [0, dt] using adaptive RK45,
        updates self.state in place, and returns the new state.
        """
        u_val = float(u) if not isinstance(u, float) else u
        sol = solve_ivp(
            fun=lambda _, y: [1 - y[0] ** 2 + u_val],
            t_span=(0, self.dt),
            y0=[self.state],
            method="RK45",
            dense_output=False,
        )
        self.state = float(sol.y[0, -1])
        return self.state

    def f(self, state, u):
        return 1 - state**2 + u


# ── Analytic controller (for data generation) ─────────────────────────────


class ProportionalController(nn.Module):
    """u = k * (x - x_r).  Logs (error, control) pairs to CSV."""

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = k

    def forward(self, x_r, x):
        """Returns u = k*error"""
        error = x - x_r
        return self.k * error


# ── NN controller ─────────────────────────────────────────────────────────


class ProportionalControllerNet(nn.Module):
    """Small feedforward net (ReLU) that approximates the P-controller."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 5, output_dim: int = 1):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.output(x)
        return x


def train(model, errors, controls, max_epochs=1000, lr=0.01):
    """Train ProportionalControllerNet on (error, control) data."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(errors.unsqueeze(1))
        loss = criterion(outputs.squeeze(), controls)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

    return model


if __name__ == "__main__":
    k = -20.0
    analytic = ProportionalController(k)
    errors = torch.linspace(-2.0, 2.0, 500)
    with torch.no_grad():
        controls = analytic(x_r=torch.zeros_like(errors), x=errors)

    model = ProportionalControllerNet()
    train(model, errors, controls)

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Trained state dict saved to {MODEL_FILE}.")
