"""Defines the LyapunovProblem dataclass — the common interface
that verifiers and the Lyapunov network are handed."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
from torch import nn, Tensor


@dataclass
class LyapunovProblem:
    """A NN Lyapunov function, dynamics, and region of interest.
    This is collectively what is verified.
    """

    nn_lyapunov: nn.Module
    dynamics: nn.Module
    region: Tensor

    @property
    def state_dim(self) -> int:
        return self.region.shape[0]

    def to(self, device: str | torch.device) -> None:
        """Move nn_lyapunov, dynamics, and region to the same given device."""
        self.nn_lyapunov.to(device)
        self.dynamics.to(device)
        self.region.to(device)

    def __repr__(self) -> str:
        region_str = ", ".join(
            f"x{i + 1} ∈ [{self.region[i, 0].item():.3g}, {self.region[i, 1].item():.3g}]"
            for i in range(self.state_dim)
        )
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"region=[{region_str}], "
            f"lyapunov={self.nn_lyapunov.__class__.__name__}, "
            f"dynamics={self.dynamics.__class__.__name__})"
        )


def lyapunov_loss_function(
    x_train_2d, practice_nn, dynamics: nn.Module, alpha: float = 0.1
):
    """Composite Lyapunov training loss.

    Penalises:
      1. V(0) != 0  (origin condition)
      2. V(x) < 0   (positivity)
      3. V_dot >= 0  (Lie derivative must be strictly negative)
      4. Flat bowl   (prevents V from collapsing to a near-constant)
    """
    x_train_2d.requires_grad_(True)

    V_x = practice_nn(x_train_2d)

    # 1. Origin penalty
    origin = torch.zeros((1, 2), device=x_train_2d.device)
    V_0 = practice_nn(origin)
    origin_penalty = torch.pow(V_0, 2).mean()

    # 2. Positivity penalty
    positive_penalty = torch.relu(-V_x).mean()

    # 3. Lie derivative penalty
    grad_V = torch.autograd.grad(
        outputs=V_x,
        inputs=x_train_2d,
        grad_outputs=torch.ones_like(V_x),
        create_graph=True,
    )[0]
    with torch.no_grad():
        f_vec = dynamics(x_train_2d)
    lie_derivative = torch.sum(grad_V * f_vec, dim=1)
    lie_penalty = torch.relu(lie_derivative + 1e-3).mean()

    # 4. Flatness penalty — ReLU ensures this only fires when V is too flat,
    # not when V is already large (without ReLU the term goes unboundedly negative
    # once the bowl shape is established, destabilizing the other terms).
    x_norm = torch.linalg.vector_norm(x_train_2d, dim=1, keepdim=True)
    flatness_penalty = torch.relu(x_norm - alpha * V_x).mean()

    return origin_penalty + positive_penalty + lie_penalty + flatness_penalty


def train_model_lyapunov_general(
    model,
    x_train_2d,
    num_epochs,
    learning_rate,
    dynamics: nn.Module,
    alpha: float = 1.0,
):
    """Train a neural Lyapunov function using the composite Lyapunov loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = lyapunov_loss_function(x_train_2d, model, dynamics, alpha)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model


def fine_tune_on_counterexamples(
    problem: LyapunovProblem,
    counterexamples: list,
    num_epochs: int = 200,
    learning_rate: float = 1e-4,
) -> None:
    """Refine the Lyapunov network on verifier counterexamples.

    Only the Lie-derivative penalty is applied (plus the origin condition so
    V(0) = 0 is not disturbed).  Using a small learning rate keeps all other
    regions of V approximately intact — equivalent to "freezing" the good parts
    without literally locking any weights.

    Args:
        problem:          LyapunovProblem whose nn_lyapunov will be updated in place.
        counterexamples:  List of (x1, x2) points where V_dot >= 0.
        num_epochs:       Gradient steps on the counterexample batch.
        learning_rate:    Small LR to avoid forgetting already-correct regions.
    """
    model = problem.nn_lyapunov
    device = next(model.parameters()).device

    cex_t = torch.tensor(counterexamples, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        x = cex_t.clone().requires_grad_(True)
        V_x = model(x)

        # Origin condition — don't let it drift
        origin = torch.zeros((1, 2), device=device)
        origin_penalty = model(origin).pow(2).mean()

        # Lie derivative at counterexample points only
        grad_V = torch.autograd.grad(
            outputs=V_x,
            inputs=x,
            grad_outputs=torch.ones_like(V_x),
            create_graph=True,
        )[0]
        with torch.no_grad():
            f_vec = problem.dynamics(cex_t)
        lie_derivative = torch.sum(grad_V * f_vec, dim=1)
        lie_penalty = torch.relu(lie_derivative + 1e-6).mean()

        loss = origin_penalty + lie_penalty
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"  fine-tune epoch [{epoch + 1}/{num_epochs}], lie_penalty={lie_penalty.item():.4f}"
            )


def train_lyapunov_2d(
    problem: LyapunovProblem,
    grid_pts: int = 50,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    alpha: float = 1.0,
):
    # Names easier to work with
    x1_min, x1_max = problem.region[0, 0].item(), problem.region[0, 1].item()
    x2_min, x2_max = problem.region[1, 0].item(), problem.region[1, 1].item()
    model = problem.nn_lyapunov
    device = next(model.parameters()).device

    # First, create a training set from linearly spaced samples in grid
    x1_t = torch.linspace(x1_min, x1_max, grid_pts)
    x2_t = torch.linspace(x2_min, x2_max, grid_pts)
    x1g, x2g = torch.meshgrid(x1_t, x2_t, indexing="ij")
    x_train = torch.stack([x1g.flatten(), x2g.flatten()], dim=1).to(device)

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = lyapunov_loss_function(x_train, model, problem.dynamics, alpha)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
