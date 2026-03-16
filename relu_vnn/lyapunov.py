"""Defines the LyapunovProblem dataclass — the common interface
that verifiers and the Lyapunov network are handed."""

import time

import numpy as np
import torch
from torch import nn, Tensor
from scipy.spatial import ConvexHull


class LyapunovProblem:
    """A NN Lyapunov function, dynamics, and region of interest.
    This is collectively what is verified.
    """

    def __init__(
        self,
        nn_lyapunov: nn.Module,
        dynamics: nn.Module,
        region: Tensor,
    ) -> None:
        self.nn_lyapunov = nn_lyapunov
        self.dynamics = dynamics
        self.region = region
        self.hole: float = 1e-6  # Hole around origin for numerical stability
        self.early_exit: bool = False

    @property
    def state_dim(self) -> int:
        return self.region.shape[0]

    @property
    def device(self) -> torch.device:
        return next(self.nn_lyapunov.parameters()).device

    def to(self, device: str | torch.device) -> "LyapunovProblem":
        """Move nn_lyapunov and dynamics to *device*.

        Region stays on CPU — it only holds float bounds accessed via .item()
        and never participates in GPU computation.
        """
        self.nn_lyapunov.to(device)
        self.dynamics.to(device)
        return self

    def check_origin(self) -> np.ndarray | None:
        """Check that V(0) = 0 (first Lyapunov condition).

        Returns None if the condition holds, or a 1-D array [x1, ..., xn, V(0)]
        indicating the violation if the condition fails.
        """
        origin = torch.zeros(1, self.state_dim, device=self.device)
        v0 = float(self.nn_lyapunov(origin).item())
        if np.isclose(v0, 0.0):
            return None
        # Return 2-D array (1, state_dim+1) for consistency with other checks
        return np.array([[*([0.0] * self.state_dim), v0]])

    def check_positive(self, H: list[ConvexHull]) -> np.ndarray | None:
        """Check that V(x) > 0 for all x in region, except the origin (second Lyapunov condition).

        Only checks convex hull vertices. Vertices within self.hole of the origin are skipped.

        Returns None if the condition holds, or a 2-D array with each row
        [x1, ..., xn, V(x)] indicating a violation.
        """
        all_verts = np.concatenate([hull.points[hull.vertices] for hull in H], axis=0)
        norms = np.linalg.norm(all_verts, axis=1)
        all_verts = all_verts[norms >= self.hole]

        if len(all_verts) == 0:
            print("Warning: all vertices at origin. Skipping.")
            return None

        x = torch.tensor(all_verts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.nn_lyapunov(x).squeeze(1).cpu().numpy()  # (N,)

        violations_mask = v <= 0.0
        if not np.any(violations_mask):
            return None

        violating_verts = all_verts[violations_mask]
        violating_v = v[violations_mask]
        cex = np.concatenate([violating_verts, violating_v[:, None]], axis=1)

        return cex

    def check_decrease(self, H: list[ConvexHull]) -> np.ndarray | None:
        """Check that ∇V(x)·f(x) < 0 everywhere except the origin (third Lyapunov condition).

        Returns None if the condition holds, or a 2-D array with each row
        [x1, x2, V_dot] indicating a violation.
        """
        # TODO
        return None

    def verify(self) -> dict[str, np.ndarray | list[ConvexHull] | None]:
        """Run all three Lyapunov checks.

        Returns a dict with keys:
            "origin"   — None if V(0)=0 holds, else ndarray [x1,...,xn, V(0)]
            "positive" — None if V(x)>0 holds, else ndarray of violation rows
            "decrease" — None if V_dot<0 holds, else ndarray of violation rows
            "cells"    — list[ConvexHull], the cell decomposition (for plotting)
        """
        from .hyperplane import (
            extract_weights,
            build_bbox_halfspaces,
            enumerate_cells_bfs,
        )

        results: dict[str, np.ndarray | list[ConvexHull] | None] = {
            "origin": None,
            "positive": None,
            "decrease": None,
        }

        x1_min, x1_max = self.region[0, 0].item(), self.region[0, 1].item()
        x2_min, x2_max = self.region[1, 0].item(), self.region[1, 1].item()

        W_matrix, B_vector, _ = extract_weights(self.nn_lyapunov)
        bbox_hs = build_bbox_halfspaces(x1_min, x1_max, x2_min, x2_max)

        print("Enumerating cells via BFS...")
        t0 = time.perf_counter()
        cells = enumerate_cells_bfs(
            W_matrix, B_vector, bbox_hs, x1_min, x1_max, x2_min, x2_max
        )
        print(f"  done ({time.perf_counter() - t0:.3f}s)  — {len(cells)} cells")

        # Criteria 1: V(0) = 0
        print("Checking origin condition...")
        t0 = time.perf_counter()
        results["origin"] = self.check_origin()
        print(f"  done ({time.perf_counter() - t0:.3f}s)")
        if (cex := results["origin"]) is not None:
            print(f"Found {len(cex)} counterexamples.")
            if self.early_exit:
                print("Violation found! Exiting early.")
                return results
        else:
            print("Check passed!")

        # Criteria 2: V(x) > 0
        print("Checking positive condition...")
        t0 = time.perf_counter()
        results["positive"] = self.check_positive(cells)
        print(f"  done ({time.perf_counter() - t0:.3f}s)")
        if (cex := results["positive"]) is not None:
            print(f"Found {len(cex)} counterexamples.")
            if self.early_exit:
                print("Violation found! Exiting early.")
                return results
        else:
            print("Check passed!")

        # Criteria 3: V_dot < 0
        print("Checking decrease condition...")
        t0 = time.perf_counter()
        results["decrease"] = self.check_decrease(cells)
        print(f"  done ({time.perf_counter() - t0:.3f}s)")
        if (cex := results["decrease"]) is not None:
            print(f"Found {len(cex)} counterexamples.")
            if self.early_exit:
                print("Violation found! Exiting early.")
                return results
        else:
            print("Check passed!")

        # Save cells for plotting even if all checks pass.
        results["cells"] = cells
        print("Verification complete.")
        print("Summary of results:")
        for key in ["origin", "positive", "decrease"]:
            if results[key] is None:
                print(f"  {key}: PASSED")
            else:
                print(f"  {key}: FAILED with {len(results[key])} counterexamples")  # type: ignore
        print(f"  cells: {len(results['cells'])} returned for plotting")
        return results

    def __repr__(self) -> str:
        region_str = ", ".join(
            f"x{i + 1} ∈ [{self.region[i, 0].item():.3g}, {self.region[i, 1].item():.3g}]"
            for i in range(self.state_dim)
        )
        hidden_size = getattr(self.nn_lyapunov, "hidden_size", None)
        hidden_str = f", hidden_size={hidden_size}" if hidden_size is not None else ""
        return (
            f"{self.__class__.__name__}("
            f"state_dim={self.state_dim}, "
            f"region=[{region_str}], "
            f"lyapunov={self.nn_lyapunov.__class__.__name__}{hidden_str}, "
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

    cex_t = torch.tensor(counterexamples, dtype=torch.float32, device=problem.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        x = cex_t.clone().requires_grad_(True)
        V_x = model(x)

        # Origin condition — don't let it drift
        origin = torch.zeros((1, 2), device=problem.device)
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
