"""Training functions for neural Lyapunov networks."""

import logging

import numpy as np
import torch
from torch import nn

from .lyapunov import LyapunovProblem

logger = logging.getLogger(__name__)


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
    origin = torch.zeros((1, x_train_2d.shape[1]), device=x_train_2d.device)
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
    lie_penalty = torch.relu(lie_derivative + 1e-2).mean()

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
            logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, num_epochs, loss.item())
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
        counterexamples:  List of state-space points where V_dot >= 0.
        num_epochs:       Gradient steps on the counterexample batch.
        learning_rate:    Small LR to avoid forgetting already-correct regions.
    """
    model = problem.nn_lyapunov

    cex_t = torch.tensor(counterexamples, dtype=torch.float32, device=problem.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        x = cex_t.clone().requires_grad_(True)
        V_x = model(x)

        # Origin condition — don't let it drift
        origin = torch.zeros((1, problem.state_dim), device=problem.device)
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
            logger.info(
                "  fine-tune epoch [%d/%d], lie_penalty=%.4f",
                epoch + 1,
                num_epochs,
                lie_penalty.item(),
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

    # First, create a training set from linearly spaced samples in grid
    x1_t = torch.linspace(x1_min, x1_max, grid_pts)
    x2_t = torch.linspace(x2_min, x2_max, grid_pts)
    x1g, x2g = torch.meshgrid(x1_t, x2_t, indexing="ij")
    x_train = torch.stack([x1g.flatten(), x2g.flatten()], dim=1).to(problem.device)

    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = lyapunov_loss_function(x_train, model, problem.dynamics, alpha)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, num_epochs, loss.item())


def build_base_grid(problem: LyapunovProblem, grid_pts: int) -> torch.Tensor:
    """Build a uniform grid over the problem's region."""
    region_np = problem.region.numpy()
    linspaces = [
        np.linspace(region_np[d, 0], region_np[d, 1], grid_pts)
        for d in range(problem.state_dim)
    ]
    mesh = np.meshgrid(*linspaces, indexing="ij")
    return torch.tensor(
        np.stack([m.ravel() for m in mesh], axis=1), dtype=torch.float32
    )


def retrain(
    problem: LyapunovProblem,
    base_grid: torch.Tensor,
    cexs: list[tuple],
    epochs: int,
    lr: float,
    device: torch.device,
    cex_oversample: int = 3,
):
    """Retrain the Lyapunov network on a grid merged with counterexamples.

    Counterexample points are oversampled (duplicated) in the training tensor
    so they have more influence without a separate weighted loss.
    """
    problem.to(device)
    problem.update_shift()

    _BALL_RADIUS = 0.05  # fraction of region span per dimension
    _N_BALL = 9  # extra samples per counterexample

    # Merge counterexamples into the training grid
    grid_dev = base_grid.to(device)
    if cexs:
        cex_pts = torch.tensor([p[:-1] for p in cexs], dtype=torch.float32)  # (M, d)
        cex_repeated = cex_pts.repeat(cex_oversample, 1)

        # Sample a ball around each counterexample for better cell coverage
        region_span = (problem.region[:, 1] - problem.region[:, 0]).float()  # (d,)
        radius = _BALL_RADIUS * region_span  # per-dimension radius
        m, d = cex_pts.shape
        noise = torch.randn(m * _N_BALL, d)
        noise = noise / noise.norm(dim=1, keepdim=True).clamp(min=1e-8)
        scale = torch.rand(m * _N_BALL, 1) ** (1.0 / d)
        centers = cex_pts.repeat_interleave(_N_BALL, dim=0)
        ball_pts = centers + scale * noise * radius.unsqueeze(0)
        lo = problem.region[:, 0].float()
        hi = problem.region[:, 1].float()
        ball_pts = torch.clamp(ball_pts, lo, hi)
        ball_pts = ball_pts[ball_pts.norm(dim=1) >= problem.hole]

        train_data = torch.cat(
            [grid_dev, cex_repeated.to(device), ball_pts.to(device)], dim=0
        )
    else:
        train_data = grid_dev

    optimizer = torch.optim.Adam(problem.nn_lyapunov.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    for epoch in range(epochs):
        problem.nn_lyapunov.train()
        optimizer.zero_grad()
        loss = lyapunov_loss_function(train_data, problem.nn_lyapunov, problem.dynamics)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f"  retrain epoch [{epoch + 1}/{epochs}]  loss={loss.item():.4f}")

    # Re-zero V(0) after training drift
    problem.update_shift()
