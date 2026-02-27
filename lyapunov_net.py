"""Neural Lyapunov function: network definition, training, and
counterexample-driven fine-tuning with replay buffer."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralLyapunov(nn.Module):
    """ReLU network that approximates a Lyapunov function V(x).

    Architecture: Linear→ReLU repeated `num_hidden_layers` times,
    then a final Linear output (no activation).  ReLU is required
    so the hyperplane verifier can exploit piecewise-linear structure.
    """

    def __init__(self, input_dim=2, hidden_dim=5, output_dim=1, num_hidden_layers=1):
        super().__init__()
        layers = []
        for i in range(num_hidden_layers):
            in_features = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train(model, problem, grid_points=None, num_epochs=1000, lr=0.01):
    """Train a NeuralLyapunov to approximate a known analytic V.

    Parameters
    ----------
    model : NeuralLyapunov
    problem : LyapunovProblem
        Must have `analytic_V` set.
    grid_points : int or None
        Points per axis for training grid.  Defaults to 250.
    num_epochs : int
    lr : float

    Returns
    -------
    model : the trained network (modified in place)
    """
    if problem.analytic_V is None:
        raise ValueError("train() requires problem.analytic_V to be set")

    if grid_points is None:
        grid_points = 250

    # Build training data over the domain
    axes = [np.linspace(lo, hi, grid_points) for lo, hi in problem.domain]
    grids = np.meshgrid(*axes, indexing="ij")
    flat = np.stack([g.ravel() for g in grids], axis=1)  # (N, dim)

    x_train = torch.tensor(flat, dtype=torch.float32)
    y_train = torch.tensor(
        [problem.analytic_V(*row) for row in flat],
        dtype=torch.float32,
    ).unsqueeze(1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


# ── Replay buffer & fine-tuning ────────────────────────────────────────────


class ReplayBuffer:
    """Stores counterexample points and tracks per-point V_dot history
    to detect regressions."""

    def __init__(self):
        self.points = []  # list of np.ndarray, shape (dim,)
        self.vdot_history = []  # list of lists, one per point

    def add(self, counterexamples):
        """Add new counterexample points (list of tuples or arrays).
        Deduplicates against existing points."""
        for pt in counterexamples:
            pt_arr = np.asarray(pt)
            if not self._already_has(pt_arr):
                self.points.append(pt_arr)
                self.vdot_history.append([])

    def _already_has(self, pt, tol=1e-6):
        for existing in self.points:
            if np.allclose(existing, pt, atol=tol):
                return True
        return False

    def evaluate(self, model, problem):
        """Evaluate V_dot at all buffer points, record history.
        Returns (vdots, regressions) where regressions is a list
        of indices that flipped from satisfied to violated."""
        regressions = []
        vdots = []
        for i, pt in enumerate(self.points):
            vdot = _compute_vdot(model, pt, problem)
            vdots.append(vdot)
            self.vdot_history[i].append(vdot)
            # Check for regression: previously < 0, now >= 0
            if len(self.vdot_history[i]) >= 2:
                prev = self.vdot_history[i][-2]
                if prev < 0 and vdot >= 0:
                    regressions.append(i)
        return np.array(vdots), regressions

    def __len__(self):
        return len(self.points)


def _compute_vdot(model, pt, problem):
    """Compute Lie derivative V_dot = grad(V) . f at a single point."""
    x = torch.tensor(pt.reshape(1, -1), dtype=torch.float32, requires_grad=True)
    V = model(x)
    V_grad = torch.autograd.grad(
        V, x, grad_outputs=torch.ones_like(V), create_graph=False
    )[0]

    x_col = pt.reshape(-1, 1)
    f_val = problem.f_eval(x_col)  # (dim, 1)

    vdot = float(V_grad.detach().numpy().flatten() @ f_val.flatten())
    return vdot


def fine_tune(
    model,
    problem,
    replay_buffer,
    max_epochs=200,
    lr=1e-4,
    grid_points=100,
    lambda_ce=10.0,
):
    """Fine-tune the Lyapunov network using counterexamples in the
    replay buffer while preserving fit on the full domain.

    Loss = MSE(V, V_analytic) on domain grid
         + lambda_ce * mean(clamp(V_dot, min=0)) at buffer points

    Parameters
    ----------
    model : NeuralLyapunov
    problem : LyapunovProblem (must have analytic_V)
    replay_buffer : ReplayBuffer
    max_epochs : int
    lr : float
    grid_points : int  – points per axis for domain grid
    lambda_ce : float  – weight on counterexample penalty

    Returns
    -------
    model, vdots_before, vdots_after, regressions
    """
    if len(replay_buffer) == 0:
        return model, np.array([]), np.array([]), []

    # Evaluate buffer before fine-tuning
    vdots_before, _ = replay_buffer.evaluate(model, problem)

    # Build domain training data
    axes = [np.linspace(lo, hi, grid_points) for lo, hi in problem.domain]
    grids = np.meshgrid(*axes, indexing="ij")
    flat = np.stack([g.ravel() for g in grids], axis=1)
    x_domain = torch.tensor(flat, dtype=torch.float32)
    y_domain = torch.tensor(
        [problem.analytic_V(*row) for row in flat],
        dtype=torch.float32,
    ).unsqueeze(1)

    # Build counterexample tensor
    ce_pts = np.array(replay_buffer.points)  # (N_ce, dim)
    x_ce = torch.tensor(ce_pts, dtype=torch.float32, requires_grad=True)

    # Pre-compute f at counterexample points
    f_at_ce = torch.tensor(
        np.array([problem.f_eval(p.reshape(-1, 1)).flatten() for p in ce_pts]),
        dtype=torch.float32,
    )  # (N_ce, dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        # Domain loss
        loss_domain = criterion(model(x_domain), y_domain)

        # Counterexample loss: penalise V_dot >= 0
        V_ce = model(x_ce)
        V_grad_ce = torch.autograd.grad(
            V_ce,
            x_ce,
            grad_outputs=torch.ones_like(V_ce),
            create_graph=True,
        )[0]  # (N_ce, dim)
        vdot_ce = (V_grad_ce * f_at_ce).sum(dim=1)  # (N_ce,)
        loss_ce = torch.mean(torch.clamp(vdot_ce, min=0))

        loss = loss_domain + lambda_ce * loss_ce
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(
                f"  fine-tune [{epoch + 1}/{max_epochs}] "
                f"domain={loss_domain.item():.4e} "
                f"ce={loss_ce.item():.4e}"
            )

    # Evaluate buffer after fine-tuning
    vdots_after, regressions = replay_buffer.evaluate(model, problem)
    if regressions:
        print(f"  WARNING: {len(regressions)} regressions detected")

    return model, vdots_before, vdots_after, regressions
