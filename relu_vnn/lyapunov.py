"""Defines the LyapunovProblem dataclass — the common interface
that verifiers and the Lyapunov network are handed."""

import logging
import multiprocessing as mp
import time
import warnings

import numpy as np
import torch
from torch import nn, Tensor
from scipy.spatial import HalfspaceIntersection

from relu_vnn.hyperplane import EPS

logger = logging.getLogger(__name__)


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
        device: str | torch.device = "cpu",
    ) -> None:
        self.nn_lyapunov = nn_lyapunov
        self.dynamics = dynamics
        self.region = region
        self.device = device
        self.hole: float = 0.0  # Set by CLI command handlers
        self.early_exit: bool = False
        self.max_workers: int = max_workers

    @property
    def state_dim(self) -> int:
        return self.region.shape[0]

    def to(self, device: str | torch.device) -> "LyapunovProblem":
        """Move nn_lyapunov and dynamics to *device*.

        Region stays on CPU — it only holds float bounds accessed via .item()
        and never participates in GPU computation.
        """
        self.nn_lyapunov.to(device)
        self.dynamics.to(device)
        self.device = device
        return self

    def update_shift(self) -> None:
        """Recompute the output shift so that V(0) = 0 exactly.

        Sets nn_lyapunov.shift = network(0), so forward(0) = network(0) - shift = 0.
        Call before each training session (not each epoch).

        No-op if the network does not have a ``shift`` buffer.
        """
        if not hasattr(self.nn_lyapunov, "shift"):
            return
        with torch.no_grad():
            origin = torch.zeros(1, self.state_dim, device=self.device)
            raw_v0 = self.nn_lyapunov.network(origin).squeeze()
            self.nn_lyapunov.shift.copy_(raw_v0)

    def check_origin(self) -> dict:
        """Check that V(0) = 0 (first Lyapunov condition).

        Returns a dict with keys:
            "passed"          — True if V(0) ≈ 0
            "v0"              — float, the actual value V(0)
            "counterexamples" — None if passed, else [[0, ..., 0, v0]]
        """
        origin = torch.zeros(1, self.state_dim, device=self.device)
        v0 = float(self.nn_lyapunov(origin).item())
        passed = bool(np.isclose(v0, 0.0, atol=EPS))
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
        # Round to FP32 precision before deduplication — adjacent cells share vertices
        all_verts = np.unique(np.round(all_verts, 6), axis=0)
        norms = np.linalg.norm(all_verts, axis=1)
        all_verts = all_verts[norms >= self.hole]

        if len(all_verts) == 0:
            logger.warning("All vertices at origin — skipping positivity check.")
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
        logger.info(
            "Checking decrease condition on %d cells with %d workers...",
            n_cells,
            n_workers,
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
            cell_idx = int(r["cell_idx"])  # type: ignore[arg-type]
            outcome = str(r["outcome"])
            bounds_str = str(r["bounds_str"])
            elapsed = float(r["elapsed"])  # type: ignore[arg-type]

            if outcome in ("shgo_certified", "zero_norm"):
                n_certified += 1
                logger.debug(
                    "[%d/%d] certified  %s  (%.3fs)",
                    cell_idx + 1,
                    n_cells,
                    bounds_str,
                    elapsed,
                )
            else:
                n_violated += 1
                if outcome == "zero_grad":
                    logger.debug(
                        "[%d/%d] zero-grad violation  %s  (%.3fs)",
                        cell_idx + 1,
                        n_cells,
                        bounds_str,
                        elapsed,
                    )
                elif outcome == "pre-check_violation":
                    logger.debug(
                        "[%d/%d] pre-check violation  %s  v_dot_max=%.4g  (%d pts)  (%.3fs)",
                        cell_idx + 1,
                        n_cells,
                        bounds_str,
                        r["v_dot_max"],
                        r["n_pts"],
                        elapsed,
                    )
                elif outcome == "shgo_violation":
                    logger.debug(
                        "[%d/%d] shgo violation  %s  v_dot_max=%.4g  (%d pts)  (%.3fs)",
                        cell_idx + 1,
                        n_cells,
                        bounds_str,
                        r["v_dot_max"],
                        r["n_pts"],
                        elapsed,
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

        logger.info(
            "check_decrease done: %d/%d certified, %d/%d violated",
            n_certified,
            n_cells,
            n_violated,
            n_cells,
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
        logger.info("Extracting weights and building bounding-box halfspaces...")
        t0 = time.perf_counter()
        W_matrix, B_vector, _ = extract_weights(self.nn_lyapunov)
        bbox_hs = build_bbox_halfspaces(region_np)
        logger.info("  done (%.3fs)", time.perf_counter() - t0)

        logger.info("Enumerating cells via BFS...")
        t0 = time.perf_counter()
        cells = enumerate_cells_bfs(
            W_matrix, B_vector, bbox_hs, region_np, n_workers=self.max_workers
        )
        logger.info("  done (%.3fs)  — %d cells", time.perf_counter() - t0, len(cells))
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
        logger.info("Checking origin condition...")
        t0 = time.perf_counter()
        results["origin"] = self.check_origin()
        logger.info("  done (%.3fs)", time.perf_counter() - t0)
        if not results["origin"]["passed"]:
            logger.info("Origin violation: V(0) = %.6g", results["origin"]["v0"])
            if self.early_exit:
                logger.info("Violation found! Exiting early.")
                return results
        else:
            logger.info("Check passed!")

        # Build the cell decomposition for the next two checks.
        cells = self.enumerate_cells()
        results["n_cells"] = len(cells)
        results["cells"] = cells

        # Criteria 2: V(x) > 0
        logger.info("Checking positive condition...")
        t0 = time.perf_counter()
        results["positive"] = self.check_positive(cells)
        logger.info("  done (%.3fs)", time.perf_counter() - t0)
        if not results["positive"]["passed"]:
            logger.info(
                "Found %d counterexamples.", results["positive"]["n_violations"]
            )
            if self.early_exit:
                logger.info("Violation found! Exiting early.")
                return results
        else:
            logger.info("Check passed!")

        # Criteria 3: V_dot < 0
        logger.info("Checking decrease condition...")
        t0 = time.perf_counter()
        results["decrease"] = self.check_decrease(cells)
        logger.info("  done (%.3fs)", time.perf_counter() - t0)
        if not results["decrease"]["passed"]:
            logger.info(
                "Found %d counterexamples.", results["decrease"]["n_violations"]
            )
        else:
            logger.info("Check passed!")

        logger.info("Verification complete.")
        logger.info("Summary of results:")
        for key in ["origin", "positive", "decrease"]:
            r = results[key]
            if r is None:
                logger.info("  %s: skipped", key)
            elif r["passed"]:
                logger.info("  %s: PASSED", key)
            elif key == "origin":
                logger.info("  %s: FAILED (V(0) = %.6g)", key, r["v0"])
            else:
                logger.info(
                    "  %s: FAILED with %d counterexamples", key, r["n_violations"]
                )
        logger.info("  cells: %d returned.", results["n_cells"])
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
