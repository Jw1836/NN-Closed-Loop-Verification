"""Training functions for neural Lyapunov networks."""

import logging

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


def build_base_grid(problem: LyapunovProblem, n_samples: int) -> torch.Tensor:
    """Sample uniformly at random over the problem's region.

    Random sampling avoids the exponential blow-up of a full meshgrid in high
    dimensions (D > 2). The argument is interpreted as a total sample count, not
    pts-per-dimension, so memory usage is predictable regardless of state_dim.
    """
    lo = problem.region[:, 0]  # (D,)
    hi = problem.region[:, 1]  # (D,)
    samples = torch.rand(n_samples, problem.state_dim)  # uniform [0, 1)
    return lo + samples * (hi - lo)  # rescale to region


def default_initial_grid_pts(_problem: LyapunovProblem) -> int:
    """Heuristic for AdamW initial training: target ~500k total samples."""
    return 500_000


def default_finetune_grid_pts(_problem: LyapunovProblem) -> int:
    """Heuristic for L-BFGS fine-tuning: target ~10k total samples."""
    return 10_000


def train_initial(
    problem: LyapunovProblem,
    grid_pts: int | None = None,
    epochs: int = 400,
    lr: float = 1e-3,
    alpha: float = 1.0,
):
    """Train the Lyapunov network from scratch on a uniform grid using AdamW.

    AdamW's stochastic exploration finds the globally structured (radial) minimum
    that L-BFGS misses when starting from random init. Weight decay additionally
    discourages flat-bowl local minima near the origin.
    """
    if grid_pts is None:
        grid_pts = default_initial_grid_pts(problem)
        logger.info("Initial training grid: %d samples (heuristic)", grid_pts)
    model = problem.nn_lyapunov
    train_data = build_base_grid(problem, grid_pts).to(problem.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = lyapunov_loss_function(train_data, model, problem.dynamics, alpha)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            logger.info("Epoch [%d/%d], Loss: %.7f", epoch + 1, epochs, loss.item())


def finetune(
    problem: LyapunovProblem,
    cexs: list[tuple],
    epochs: int,
    device: torch.device,
    alpha: float = 1.0,
    cex_oversample: int = 3,
    grid_pts: int | None = None,
    lbfgs_max_iter: int = 20,
):
    """Fine-tune the Lyapunov network on a fresh small grid merged with counterexamples.

    A fresh grid is built each call (grid_pts per dimension) so the training
    dataset stays small and focused — the gradient from violating cells is not
    diluted by thousands of already-satisfied grid points accumulated over prior
    iterations.

    Uses L-BFGS with strong Wolfe line search — well-suited for polishing a
    nearly-correct solution on a fixed dataset.
    """
    if grid_pts is None:
        grid_pts = default_finetune_grid_pts(problem)
    problem.to(device)

    _BALL_RADIUS = 0.05  # fraction of region span per dimension
    _N_BALL = 9  # extra samples per counterexample

    # Merge counterexamples into the training grid
    grid_dev = build_base_grid(problem, grid_pts).to(device)
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

    model = problem.nn_lyapunov
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = lyapunov_loss_function(train_data, model, problem.dynamics, alpha)
        loss.backward()
        return loss

    for epoch in range(epochs):
        model.train()
        loss = optimizer.step(closure)
        if (epoch + 1) % 10 == 0:
            print(f"  retrain epoch [{epoch + 1}/{epochs}]  loss={loss.item():.7f}")

    # Re-zero V(0) after training drift
    problem.update_shift()
