"""Defines the LyapunovProblem dataclass — the common interface
that verifiers and the Lyapunov network are handed."""

import time
import warnings

import numpy as np
import torch
from torch import nn, Tensor
from scipy.spatial import HalfspaceIntersection


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

        Returns None if the condition holds, or a 2-D array of shape
        (1, state_dim+1) with row [x1, ..., xn, V(0)] if the condition fails.
        """
        origin = torch.zeros(1, self.state_dim, device=self.device)
        v0 = float(self.nn_lyapunov(origin).item())
        if np.isclose(v0, 0.0):
            return None
        # Return 2-D array (1, state_dim+1) for consistency with other checks
        return np.array([[*([0.0] * self.state_dim), v0]])

    def check_positive(self, H: list[HalfspaceIntersection]) -> np.ndarray | None:
        """Check that V(x) > 0 for all x in region, except the origin (second Lyapunov condition).

        Only checks convex hull vertices. Vertices within self.hole of the origin are skipped.

        Returns None if the condition holds, or a 2-D array with each row
        [x1, ..., xn, V(x)] indicating a violation.
        """
        all_verts = np.concatenate([cell.intersections for cell in H], axis=0)
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

    def check_decrease(
        self,
        cells: list[HalfspaceIntersection],
        callback=None,
    ) -> np.ndarray | None:
        """Check that ∇V(x)·f(x) < 0 everywhere except the origin (third Lyapunov condition).

        Args:
            cells:    Cell decomposition from enumerate_cells().
            callback: Optional callable invoked after each cell with signature
                        callback(cell_idx, total_cells, info)
                      where *info* is a dict with keys:
                        "bounds"     — np.ndarray shape (state_dim, 2), [min, max] per dim
                        "grad_norm"  — float, ‖∇V‖ at the interior point
                        "outcome"    — str: "zero_grad" | "pre-check_violation" |
                                       "shgo_violation" | "shgo_certified"
                        "v_dot_max"  — float, max V_dot found (positive = violation)
                        "elapsed"    — float, seconds spent on this cell

        Returns None if the condition holds, or a 2-D array with each row
        [x1, x2, V_dot] indicating a violation.
        """
        from .hyperplane import (
            align_basis,
            analytic_gradient,
            extract_weights,
        )
        from scipy.optimize import shgo, OptimizeResult

        W_matrix, B_vector, W_out_vec = extract_weights(self.nn_lyapunov)
        violations: list[np.ndarray] = []
        n_cells = len(cells)
        n_certified = 0
        n_violated = 0

        for cell_idx, c in enumerate(cells):
            t0 = time.perf_counter()
            verts = c.intersections
            bounds = np.column_stack([verts.min(axis=0), verts.max(axis=0)])
            bounds_str = "  ".join(
                f"x{i}:[{bounds[i, 0]:.3g},{bounds[i, 1]:.3g}]"
                for i in range(bounds.shape[0])
            )

            # Calculate cell's gradient at interior point
            grad = analytic_gradient(W_matrix, B_vector, W_out_vec, c.interior_point)
            grad_norm = float(np.linalg.norm(grad))

            # If gradient is zero, V_dot = 0 everywhere in cell, means counterexample
            if grad_norm < 1e-10:
                elapsed = time.perf_counter() - t0
                n_violated += 1
                print(
                    f"[{cell_idx + 1}/{n_cells}] zero-grad violation  {bounds_str}"
                    f"  ({elapsed:.3f}s)"
                )
                violations.append(np.append(c.interior_point, 0.0))
                if callback is not None:
                    callback(
                        cell_idx,
                        n_cells,
                        {
                            "bounds": bounds,
                            "grad_norm": 0.0,
                            "outcome": "zero_grad",
                            "v_dot_max": 0.0,
                            "elapsed": elapsed,
                        },
                    )
                continue

            # Align polytope with the basis vector
            q1 = align_basis(grad)

            # --- Batched vertex + interior point pre-check ---
            check_pts = np.vstack([verts, c.interior_point[np.newaxis]])
            x_t = torch.tensor(check_pts, dtype=torch.float32)
            with torch.no_grad():
                f_check = self.dynamics(x_t).numpy()
            v_i_check = f_check @ q1
            max_idx = int(np.argmax(v_i_check))
            if v_i_check[max_idx] >= 0:
                elapsed = time.perf_counter() - t0
                v_dot_max = float(v_i_check[max_idx])
                n_violated += 1
                print(
                    f"[{cell_idx + 1}/{n_cells}] pre-check violation  {bounds_str}"
                    f"  v_dot_max={v_dot_max:.4g}  ({elapsed:.3f}s)"
                )
                violations.append(np.append(check_pts[max_idx], v_dot_max))
                if callback is not None:
                    callback(
                        cell_idx,
                        n_cells,
                        {
                            "bounds": bounds,
                            "grad_norm": grad_norm,
                            "outcome": "pre-check_violation",
                            "v_dot_max": v_dot_max,
                            "elapsed": elapsed,
                        },
                    )
                continue

            # --- shgo for cells where vertices are all negative ---
            # Halfspace constraints: each row of cell.halfspaces is [A_i | b_i]
            # with A_i @ x + b_i <= 0.  scipy convention: g(x) >= 0.
            A_hs = np.asarray(c.halfspaces[:, :-1], dtype=np.float64)
            b_hs = np.asarray(c.halfspaces[:, -1], dtype=np.float64)

            # SLSQP constraints: -A_hs @ x - b_hs >= 0 (vectorized)
            constraints = {
                "type": "ineq",
                "fun": lambda x: -A_hs @ x - b_hs,
                "jac": lambda _x: -A_hs,
            }

            q1_tensor = torch.tensor(q1, dtype=torch.float32)

            def v_i_1(x_np: np.ndarray) -> tuple[float, np.ndarray]:
                """Negated v_i_1 with gradient via autograd.
                Returns (value, grad), as expected by shgo with jac=True.
                """
                x_t = torch.tensor(
                    np.atleast_1d(x_np), dtype=torch.float32
                ).requires_grad_(True)
                val = -(self.dynamics(x_t.unsqueeze(0)).squeeze(0) @ q1_tensor)
                val.backward()
                return val.detach().item(), np.asarray(x_t.grad, dtype=np.float64)  # type: ignore[union-attr]

            # shgo minimizes v_i_1; f_min=0 stops early on finding a violation.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="A much lower value", category=UserWarning
                )
                result: OptimizeResult = shgo(
                    v_i_1,
                    bounds=bounds.tolist(),
                    constraints=constraints,
                    sampling_method="simplicial",
                    options={"f_min": 0.0, "minimize_every_iter": True, "maxiter": 20},
                    minimizer_kwargs={
                        "method": "SLSQP",
                        "jac": True,
                    },
                )
            if not result.success:
                raise RuntimeError(f"shgo failed on cell: {result.message}")

            elapsed = time.perf_counter() - t0
            # result.fun = min of -(q1·f(x)), so violation when result.fun < 0
            if result.fun < 0.0:
                v_dot_max = float(-result.fun)
                n_violated += 1
                print(
                    f"[{cell_idx + 1}/{n_cells}] shgo violation  {bounds_str}"
                    f"  v_dot_max={v_dot_max:.4g}  ({elapsed:.3f}s)"
                )
                violations.append(np.append(result.x, v_dot_max))
                if callback is not None:
                    callback(
                        cell_idx,
                        n_cells,
                        {
                            "bounds": bounds,
                            "grad_norm": grad_norm,
                            "outcome": "shgo_violation",
                            "v_dot_max": v_dot_max,
                            "elapsed": elapsed,
                        },
                    )
                continue
            else:
                n_certified += 1
                print(
                    f"[{cell_idx + 1}/{n_cells}] certified  {bounds_str}"
                    f"  ({elapsed:.3f}s)"
                )
                if callback is not None:
                    callback(
                        cell_idx,
                        n_cells,
                        {
                            "bounds": bounds,
                            "grad_norm": grad_norm,
                            "outcome": "shgo_certified",
                            "v_dot_max": float(-result.fun),
                            "elapsed": elapsed,
                        },
                    )

        print(
            f"check_decrease done: {n_certified}/{n_cells} certified, "
            f"{n_violated}/{n_cells} violated"
        )
        return None if not violations else np.vstack(violations)

    def enumerate_cells(self) -> list[HalfspaceIntersection]:
        """Build the ReLU activation-pattern cell decomposition for this network and region."""
        from .hyperplane import (
            extract_weights,
            build_bbox_halfspaces,
            enumerate_cells_bfs,
        )

        region_np = self.region.numpy()
        print("Extracting weights and building bounding-box halfspaces...")
        t0 = time.perf_counter()
        W_matrix, B_vector, _ = extract_weights(self.nn_lyapunov)
        bbox_hs = build_bbox_halfspaces(region_np)
        print(f"  done ({time.perf_counter() - t0:.3f}s)")

        print("Enumerating cells via BFS...")
        t0 = time.perf_counter()
        cells = enumerate_cells_bfs(W_matrix, B_vector, bbox_hs, region_np)
        print(f"  done ({time.perf_counter() - t0:.3f}s)  — {len(cells)} cells")
        return cells

    def verify(self) -> dict[str, np.ndarray | list[HalfspaceIntersection] | None]:
        """Run all three Lyapunov checks.

        Returns a dict with keys:
            "origin"   — None if V(0)=0 holds, else ndarray [x1,...,xn, V(0)]
            "positive" — None if V(x)>0 holds, else ndarray of violation rows
            "decrease" — None if V_dot<0 holds, else ndarray of violation rows
            "cells"    — list[HalfspaceIntersection], the set of convex polytopes
        """

        results: dict[str, np.ndarray | list[HalfspaceIntersection] | None] = {
            "origin": None,
            "positive": None,
            "decrease": None,
        }

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

        # Build the cell decomposition for the next two checks.
        cells = self.enumerate_cells()

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

        # Return even if all checks pass; good for plots or analysis.
        results["cells"] = cells

        print("Verification complete.")
        print("Summary of results:")
        for key in ["origin", "positive", "decrease"]:
            if results[key] is None:
                print(f"  {key}: PASSED")
            else:
                print(f"  {key}: FAILED with {len(results[key])} counterexamples")  # type: ignore
        print(f"  cells: {len(results['cells'])} returned.")
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
