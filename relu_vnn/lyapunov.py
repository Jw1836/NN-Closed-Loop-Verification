"""Defines the LyapunovProblem dataclass — the common interface
that verifiers and the Lyapunov network are handed."""

import multiprocessing as mp
import time
import warnings

import numpy as np
import torch
from torch import nn, Tensor
from scipy.spatial import HalfspaceIntersection

from relu_vnn.hyperplane import EPS


def _check_cell_lie(args) -> dict[str, object]:
    """Module-level worker for per-cell Lie derivative check.

    Must be top-level for pickling with multiprocessing.
    Each worker gets its own copy of the dynamics model so autograd
    is thread/process-safe.
    """
    from .hyperplane import align_basis, analytic_gradient
    from scipy.optimize import shgo, OptimizeResult

    (
        cell,
        cell_idx,
        n_cells,
        W_matrix,
        B_vector,
        W_out_vec,
        dynamics,
        hole,
        early_exit,
    ) = args

    t0 = time.perf_counter()
    verts = cell.intersections
    bounds = np.column_stack([verts.min(axis=0), verts.max(axis=0)])
    bounds_str = "  ".join(
        f"x{i}:[{bounds[i, 0]:.3g},{bounds[i, 1]:.3g}]" for i in range(bounds.shape[0])
    )

    ## Gradient alignment and quick check

    grad = analytic_gradient(W_matrix, B_vector, W_out_vec, cell.interior_point)
    grad_norm = float(np.linalg.norm(grad))

    if grad_norm < EPS:
        # Zero gradient is a violation
        elapsed = time.perf_counter() - t0
        return {
            "cell_idx": cell_idx,
            "outcome": "zero_grad",
            "violations": [np.append(cell.interior_point, 0.0)],
            "bounds": bounds,
            "grad_norm": 0.0,
            "v_dot_max": 0.0,
            "elapsed": elapsed,
            "bounds_str": bounds_str,
            "n_pts": 1,
        }

    q1 = align_basis(grad)

    # Fast and easy pre-check just at vertices and interior point before solver
    check_pts = np.vstack([verts, cell.interior_point[np.newaxis]])
    # Remove points near origin (norm < hole)
    norms = np.linalg.norm(check_pts, axis=1)
    check_pts = check_pts[norms >= hole]
    if len(check_pts) == 0:
        # Cell is essentially at the origin; decrease condition is only required
        # away from the origin (origin itself is handled by check_origin).
        elapsed = time.perf_counter() - t0
        return {
            "cell_idx": cell_idx,
            "outcome": "zero_norm",
            "violations": [],
            "bounds": bounds,
            "grad_norm": grad_norm,
            "v_dot_max": 0.0,
            "elapsed": elapsed,
            "bounds_str": bounds_str,
            "n_pts": 0,
        }
    x_t = torch.tensor(check_pts, dtype=torch.float32)
    with torch.no_grad():
        f_check = dynamics(x_t).numpy()
    v_i_check = f_check @ q1
    violating_mask = v_i_check >= 0
    if np.any(violating_mask):
        elapsed = time.perf_counter() - t0
        v_dot_max = float(np.max(v_i_check[violating_mask]))
        n_pts = int(np.sum(violating_mask))
        viols = [
            np.append(check_pts[idx], float(v_i_check[idx]))
            for idx in np.where(violating_mask)[0]
        ]
        return {
            "cell_idx": cell_idx,
            "outcome": "pre-check_violation",
            "violations": viols,
            "bounds": bounds,
            "grad_norm": grad_norm,
            "v_dot_max": v_dot_max,
            "elapsed": elapsed,
            "bounds_str": bounds_str,
            "n_pts": n_pts,
        }

    ## shgo global optimization

    # Pass convex polytope constraints to minimizer
    A_hs = np.asarray(cell.halfspaces[:, :-1], dtype=np.float64)
    b_hs = np.asarray(cell.halfspaces[:, -1], dtype=np.float64)
    hole_sq = float(hole) ** 2
    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: -A_hs @ x - b_hs,
            "jac": lambda _x: -A_hs,
        },
        {
            # Exclude the ball of radius hole: ||x||^2 - hole^2 >= 0
            "type": "ineq",
            "fun": lambda x: float(np.dot(x, x)) - hole_sq,
            "jac": lambda x: 2.0 * np.asarray(x, dtype=np.float64),
        },
    ]

    q1_tensor = torch.tensor(q1, dtype=torch.float32)

    def v_i_1(x_np):
        """The indicator vector is the function to MAX and check for positivity;
        keep in mind that shgo solves the MIN."""
        x = torch.tensor(np.atleast_1d(x_np), dtype=torch.float32).requires_grad_(True)
        val = -(dynamics(x.unsqueeze(0)).squeeze(0) @ q1_tensor)
        val.backward()
        return val.detach().item(), np.asarray(x.grad, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="A much lower value", category=UserWarning
        )
        result: OptimizeResult = shgo(
            v_i_1,
            bounds=bounds.tolist(),
            constraints=constraints,
            sampling_method="simplicial",
            options={"minimize_every_iter": True, "maxiter": 20}
            | ({"f_min": 0.0} if early_exit else {}),
            minimizer_kwargs={"method": "SLSQP", "jac": True},
        )

    if not result.success and "feasible minimizer" not in (result.message or ""):
        raise RuntimeError(f"shgo failed on cell: {result.message}")

    elapsed = time.perf_counter() - t0

    if result.success and result.fun < 0.0:
        v_dot_max = float(-result.fun)
        viols = []
        xl = getattr(result, "xl", None)
        funl = getattr(result, "funl", None)
        if xl is not None and funl is not None:
            for xi, fi in zip(np.atleast_2d(xl), np.atleast_1d(funl)):
                if fi < 0.0:
                    viols.append(np.append(xi, float(-fi)))
        else:
            viols.append(np.append(result.x, v_dot_max))
        return {
            "cell_idx": cell_idx,
            "outcome": "shgo_violation",
            "violations": viols,
            "bounds": bounds,
            "grad_norm": grad_norm,
            "v_dot_max": v_dot_max,
            "elapsed": elapsed,
            "bounds_str": bounds_str,
            "n_pts": len(viols),
        }
    else:
        return {
            "cell_idx": cell_idx,
            "outcome": "shgo_certified",
            "violations": [],
            "bounds": bounds,
            "grad_norm": grad_norm,
            "v_dot_max": float(-result.fun) if result.fun is not None else 0.0,
            "elapsed": elapsed,
            "bounds_str": bounds_str,
            "n_pts": 0,
        }


class LyapunovProblem:
    """A NN Lyapunov function, dynamics, and region of interest.
    This is collectively what is verified.
    """

    def __init__(
        self,
        nn_lyapunov: nn.Module,
        dynamics: nn.Module,
        region: Tensor,
        max_workers: int = 1,
    ) -> None:
        self.nn_lyapunov = nn_lyapunov
        self.dynamics = dynamics
        self.region = region
        self.hole: float = EPS  # Hole around origin for numerical stability
        self.early_exit: bool = False
        self.max_workers: int = max_workers

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

    def check_origin(self) -> dict:
        """Check that V(0) = 0 (first Lyapunov condition).

        Returns a dict with keys:
            "passed"          — True if V(0) ≈ 0
            "v0"              — float, the actual value V(0)
            "counterexamples" — None if passed, else [[0, ..., 0, v0]]
        """
        origin = torch.zeros(1, self.state_dim, device=self.device)
        v0 = float(self.nn_lyapunov(origin).item())
        passed = bool(np.isclose(v0, 0.0))
        return {
            "passed": passed,
            "v0": v0,
            "counterexamples": None if passed else [[*([0.0] * self.state_dim), v0]],
        }

    def check_positive(self, H: list[HalfspaceIntersection]) -> dict:
        """Check that V(x) > 0 for all x in region, except the origin (second Lyapunov condition).

        Only checks convex hull vertices. Vertices within self.hole of the origin are skipped.

        Returns a dict with keys:
            "passed"          — True if no violations found
            "n_violations"    — int, number of violating vertices
            "counterexamples" — None if passed, else list of rows [x1, ..., xn, V(x)]
        """
        all_verts = np.concatenate([cell.intersections for cell in H], axis=0)
        norms = np.linalg.norm(all_verts, axis=1)
        all_verts = all_verts[norms >= self.hole]

        if len(all_verts) == 0:
            print("Warning: all vertices at origin. Skipping.")
            return {"passed": True, "n_violations": 0, "counterexamples": None}

        x = torch.tensor(all_verts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.nn_lyapunov(x).squeeze(1).cpu().numpy()  # (N,)

        violations_mask = v <= 0.0
        n_violations = int(np.sum(violations_mask))
        if n_violations == 0:
            return {"passed": True, "n_violations": 0, "counterexamples": None}

        violating_verts = all_verts[violations_mask]
        violating_v = v[violations_mask]
        cex = np.concatenate([violating_verts, violating_v[:, None]], axis=1)

        return {
            "passed": False,
            "n_violations": n_violations,
            "counterexamples": cex.tolist(),
        }

    def check_decrease(
        self,
        cells: list[HalfspaceIntersection],
        callback=None,
    ) -> dict:
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
        from .hyperplane import extract_weights

        W_matrix, B_vector, W_out_vec = extract_weights(self.nn_lyapunov)
        n_cells = len(cells)

        # Build args for the worker function.  Dynamics is moved to CPU
        # and each worker gets its own copy (autograd needs per-process state).
        dynamics_cpu = self.dynamics.cpu()
        args_list = [
            (
                c,
                i,
                n_cells,
                W_matrix,
                B_vector,
                W_out_vec,
                dynamics_cpu,
                self.hole,
                self.early_exit,
            )
            for i, c in enumerate(cells)
        ]

        # Each worker needs at least 4 cells to be worth the overhead
        n_workers = min(n_cells // 4, self.max_workers)
        print(
            f"Checking decrease condition on {n_cells} cells with {n_workers} workers..."
        )
        if n_workers > 1:
            ctx = mp.get_context("forkserver")
            with ctx.Pool(n_workers) as pool:
                results = pool.map(_check_cell_lie, args_list)
        else:
            results = [_check_cell_lie(a) for a in args_list]

        # Move dynamics back to the original device
        self.dynamics.to(self.device)

        # Aggregate results and print (preserves original output order)
        violations: list[np.ndarray] = []
        n_certified = 0
        n_violated = 0

        for r in results:
            cell_idx = r["cell_idx"]
            outcome = r["outcome"]
            bounds_str = r["bounds_str"]
            elapsed = r["elapsed"]

            if outcome in ("shgo_certified", "zero_norm"):
                n_certified += 1
                print(
                    f"[{cell_idx + 1}/{n_cells}] certified  {bounds_str}"
                    f"  ({elapsed:.3f}s)"
                )
            else:
                n_violated += 1
                if outcome == "zero_grad":
                    print(
                        f"[{cell_idx + 1}/{n_cells}] zero-grad violation  {bounds_str}"
                        f"  ({elapsed:.3f}s)"
                    )
                elif outcome == "pre-check_violation":
                    print(
                        f"[{cell_idx + 1}/{n_cells}] pre-check violation  {bounds_str}"
                        f"  v_dot_max={r['v_dot_max']:.4g}  ({r['n_pts']} pts)  ({elapsed:.3f}s)"
                    )
                elif outcome == "shgo_violation":
                    print(
                        f"[{cell_idx + 1}/{n_cells}] shgo violation  {bounds_str}"
                        f"  v_dot_max={r['v_dot_max']:.4g}  ({r['n_pts']} pts)  ({elapsed:.3f}s)"
                    )

            violations.extend(r["violations"])

            if callback is not None:
                callback(
                    cell_idx,
                    n_cells,
                    {
                        "bounds": r["bounds"],
                        "grad_norm": r["grad_norm"],
                        "outcome": outcome,
                        "v_dot_max": r["v_dot_max"],
                        "elapsed": elapsed,
                    },
                )

        print(
            f"check_decrease done: {n_certified}/{n_cells} certified, "
            f"{n_violated}/{n_cells} violated"
        )
        cex_array = np.vstack(violations) if violations else None
        return {
            "passed": cex_array is None,
            "n_cells": n_cells,
            "n_certified": n_certified,
            "n_violations": n_violated,
            "counterexamples": cex_array.tolist() if cex_array is not None else None,
        }

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

    def verify(self) -> dict:
        """Run all three Lyapunov checks.

        Returns a dict with keys:
            "n_cells"  — int, number of cells in the ReLU partition (0 if early exit)
            "cells"    — list[HalfspaceIntersection] (empty if early exit)
            "origin"   — result dict from check_origin() (always present)
            "positive" — result dict from check_positive(), or None if skipped
            "decrease" — result dict from check_decrease(), or None if skipped
        """

        results: dict = {
            "n_cells": 0,
            "cells": [],
            "origin": None,
            "positive": None,
            "decrease": None,
        }

        # Criteria 1: V(0) = 0
        print("Checking origin condition...")
        t0 = time.perf_counter()
        results["origin"] = self.check_origin()
        print(f"  done ({time.perf_counter() - t0:.3f}s)")
        if not results["origin"]["passed"]:
            print(f"Origin violation: V(0) = {results['origin']['v0']:.6g}")
            if self.early_exit:
                print("Violation found! Exiting early.")
                return results
        else:
            print("Check passed!")

        # Build the cell decomposition for the next two checks.
        cells = self.enumerate_cells()
        results["n_cells"] = len(cells)
        results["cells"] = cells

        # Criteria 2: V(x) > 0
        print("Checking positive condition...")
        t0 = time.perf_counter()
        results["positive"] = self.check_positive(cells)
        print(f"  done ({time.perf_counter() - t0:.3f}s)")
        if not results["positive"]["passed"]:
            print(f"Found {results['positive']['n_violations']} counterexamples.")
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
        if not results["decrease"]["passed"]:
            print(f"Found {results['decrease']['n_violations']} counterexamples.")
        else:
            print("Check passed!")

        print("Verification complete.")
        print("Summary of results:")
        for key in ["origin", "positive", "decrease"]:
            r = results[key]
            if r is None:
                print(f"  {key}: skipped")
            elif r["passed"]:
                print(f"  {key}: PASSED")
            elif key == "origin":
                print(f"  {key}: FAILED (V(0) = {r['v0']:.6g})")
            else:
                print(f"  {key}: FAILED with {r['n_violations']} counterexamples")
        print(f"  cells: {results['n_cells']} returned.")
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
