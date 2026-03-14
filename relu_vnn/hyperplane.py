"""
hyperplane_verifier.py

Hyperplane arrangement verification for neural Lyapunov functions.

Pipeline:
  1. enumerate_cells_bfs           - BFS over activation patterns to find all cells
  2. Polygon.check_gradient        - rotate, count sign regions, check dynamics
  3. full_method                   - top-level orchestrator; returns counterexamples
"""

import time
from collections import deque

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from multiprocessing import Pool
from typing import Callable, Sequence, cast
from torch import nn
from .lyapunov import LyapunovProblem

# Type aliases
ActivationPattern = tuple[bool, ...]
Vertex2D = Sequence[float] | np.ndarray  # (x1, x2)
Counterexample = tuple[float, float, float]  # (x1, x2, V_dot)


# ── Cell enumeration via BFS over activation patterns ─────────────────────────


def compute_activation_pattern(
    W_matrix: np.ndarray, B_vector: np.ndarray, x: np.ndarray
) -> ActivationPattern:
    """Return the activation pattern (tuple of bools) at point x."""
    # W_matrix.T @ x + B_vector gives pre-activations for all neurons
    pre = W_matrix.T @ x + B_vector
    # Did the ReLU fire or not?
    return tuple(bool(v >= 0) for v in pre)


def build_bbox_halfspaces(
    x1_min: float, x1_max: float, x2_min: float, x2_max: float
) -> np.ndarray:
    """Build bounding box halfspaces in the form [a1, a2, b] where a·x + b <= 0."""
    return np.array(
        [
            [-1.0, 0.0, x1_min],  # x1 >= x1_min  =>  -x1 + x1_min <= 0
            [1.0, 0.0, -x1_max],  # x1 <= x1_max  =>   x1 - x1_max <= 0
            [0.0, -1.0, x2_min],  # x2 >= x2_min  =>  -x2 + x2_min <= 0
            [0.0, 1.0, -x2_max],  # x2 <= x2_max  =>   x2 - x2_max <= 0
        ]
    )


def build_cell_halfspaces(
    W_matrix: np.ndarray,
    B_vector: np.ndarray,
    bbox_hs: np.ndarray,
    pattern: ActivationPattern,
) -> np.ndarray:
    """Build halfspaces for a polyhedral cell with the given activation pattern.

    Each neuron i contributes: if active, -w_i·x - b_i <= 0; if inactive, w_i·x + b_i <= 0.
    Combined with the 4 bounding-box halfspaces.
    """
    n_hidden = W_matrix.shape[1]
    hs = np.empty((n_hidden + 4, 3))  # each neuron + 4 bounds from original region
    hs[:4] = bbox_hs
    for i in range(n_hidden):
        w = W_matrix[:, i]
        b = B_vector[i]
        if pattern[i]:  # active: w·x + b >= 0  =>  -w·x - b <= 0
            hs[4 + i] = [-w[0], -w[1], -b]
        else:  # inactive: w·x + b <= 0
            hs[4 + i] = [w[0], w[1], b]
    return hs


def find_chebyshev_center(halfspaces: np.ndarray) -> np.ndarray | None:
    """Find a strictly interior point via the Chebyshev center LP.

    Maximizes the inscribed ball radius r subject to a_j·x + b_j + ||a_j||·r <= 0.
    Each row of *halfspaces* is ``[a1, a2, b]`` defining ``a·x + b <= 0``.
    Returns shape ``(2,)`` or ``None`` if infeasible.

    This is a slightly more expensive but robust way to find an interior point.
    It has the advantage of returning ``None`` for infeasible solutions,
    meaning they have effectively zero area.
    """
    A_hs = halfspaces[:, :2]
    b_hs = halfspaces[:, 2]
    norms = np.linalg.norm(A_hs, axis=1, keepdims=True)

    # LP: minimize -r  subject to  A_hs @ x + norms * r <= -b_hs, r >= 0
    A_lp = np.hstack([A_hs, norms])
    b_lp = -b_hs
    c = np.array([0.0, 0.0, -1.0])

    result = linprog(
        c,
        A_ub=A_lp,
        b_ub=b_lp,
        bounds=[(None, None), (None, None), (0, None)],
        method="highs",
    )
    if result.success and result.x[2] > 1e-10:
        return result.x[:2]
    return None


def enumerate_cells_bfs(
    W_matrix: np.ndarray,
    B_vector: np.ndarray,
    bbox_hs: np.ndarray,
    x1_min: float,
    x1_max: float,
    x2_min: float,
    x2_max: float,
) -> list[np.ndarray]:
    """Enumerate all non-empty cells of the ReLU hyperplane arrangement via BFS.

    Each hidden neuron defines a hyperplane ``w_i · x + b_i = 0`` that splits
    the domain into two half-spaces.  A *cell* is the intersection of one
    half-space per neuron (the "activation pattern") with the bounding box —
    i.e. the region where a specific set of ReLUs are on/off.  Within each
    cell the network is affine, so ``∇V`` is constant.

    In 2-D with *n* hyperplanes there are at most ``O(n²)`` non-empty cells
    (arrangement complexity theorem), so BFS is tractable even though the
    number of possible activation patterns is ``2ⁿ``.

    The algorithm:
      1. Seed: evaluate the activation pattern at the domain centre.
      2. For each queued pattern, build its halfspace system and compute the
         cell vertices via ``scipy.spatial.HalfspaceIntersection``.
      3. Determine which ReLU hyperplanes are *tight* (touch a vertex of the
         cell).  Each tight hyperplane has an adjacent cell on the other side
         — flip that bit to get the neighbor's activation pattern.
      4. BFS until no unvisited neighbors remain.

    Some cells leverage ``ConvexHull`` to check for degenerate intersections;
    however, downstream methods only use vertices so ``list[np.ndarray]``
    is the more appropriate return type.
    """
    # Seed the BFS from the domain centre
    center = np.array([(x1_min + x1_max) / 2, (x2_min + x2_max) / 2])
    sigma_0 = compute_activation_pattern(W_matrix, B_vector, center)

    visited: set[ActivationPattern] = {sigma_0}
    queue: deque[tuple[ActivationPattern, np.ndarray]] = deque([(sigma_0, center)])
    cells: list[np.ndarray] = []

    while queue:
        sigma, hint = queue.popleft()

        # Build the halfspace system: n ReLU constraints + 4 bounding-box
        # walls.  Each row is [a1, a2, b] meaning a·x + b ≤ 0.
        hs = build_cell_halfspaces(W_matrix, B_vector, bbox_hs, sigma)

        # --- Find a strictly interior point (required by HalfspaceIntersection) ---
        # Fast path: the hint (reflected interior of the parent cell) is often
        # already strictly feasible, saving an LP solve.
        hint_violations = hs[:, :2] @ hint + hs[:, 2]
        if np.all(hint_violations < -1e-10):
            interior = hint
        else:
            # Slow path: solve the Chebyshev-center LP for the largest
            # inscribed ball.  Returns None when the cell is infeasible
            # (activation pattern has no realizable region in the domain).
            interior = find_chebyshev_center(hs)
            if interior is None:
                continue

        # --- Compute cell vertices ---
        try:
            hs_obj = HalfspaceIntersection(hs, interior)
        except Exception:
            continue  # degenerate (e.g. numerically flat cell)

        raw_verts = hs_obj.intersections

        # HalfspaceIntersection can return duplicate or near-collinear points.
        # ConvexHull filters these and gives vertices in CCW order.
        if len(raw_verts) < 3:
            continue
        try:
            hull = ConvexHull(raw_verts)
        except Exception:
            continue  # degenerate (collinear points, etc.)
        ordered_verts = raw_verts[hull.vertices]

        if len(ordered_verts) < 3:
            continue

        cells.append(ordered_verts)

        # --- Discover neighboring cells ---
        # A ReLU hyperplane is "tight" if at least one vertex of this cell
        # lies on it (|w_i · v + b_i| < tol).  Flipping that neuron's
        # activation bit gives the adjacent cell on the other side.
        # Vectorized: evaluate all n hyperplanes at all k vertices at once.
        all_values = ordered_verts @ W_matrix + B_vector  # (k, n_hidden)
        tight_mask = np.any(np.abs(all_values) < 1e-6, axis=0)  # (n_hidden,)

        for i in np.where(tight_mask)[0]:
            # Flip bit i to get the neighbor's activation pattern
            neighbor = list(sigma)
            neighbor[i] = not neighbor[i]
            neighbor_pat = tuple(neighbor)
            if neighbor_pat not in visited:
                visited.add(neighbor_pat)
                # Reflect interior across hyperplane i as a hint for the
                # neighbor's interior point, avoiding an LP solve if it
                # lands strictly inside the neighbor cell.
                w = W_matrix[:, i]
                b = B_vector[i]
                dist = (w @ interior + b) / (w @ w)
                neighbor_hint = interior - 2 * dist * w
                queue.append((neighbor_pat, neighbor_hint))

    return cells


# ── Neural network utilities ───────────────────────────────────────────────────


def _step(x: float) -> int:
    """Heaviside step function."""
    return 1 if x >= 0 else 0


def extract_weights(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract weight arrays from a [Linear, ReLU, Linear] nn.Sequential.

    Returns
    -------
    W_matrix : ndarray, shape (input_dim, hidden_dim)
    B_vector : ndarray, shape (hidden_dim,)
    W_out_vec : ndarray, shape (hidden_dim,)
    """
    try:
        network = cast(nn.Sequential, model.network)
        layer0 = cast(nn.Linear, network[0])
        layer2 = cast(nn.Linear, network[2])
    except (AttributeError, IndexError) as e:
        raise TypeError(
            "model.network must be an nn.Sequential with indexable layers "
            "[Linear, ReLU, Linear]"
        ) from e
    if not isinstance(layer0, nn.Linear) or not isinstance(layer2, nn.Linear):
        raise TypeError(
            "model.network[0] and model.network[2] must both be nn.Linear; "
            f"got {type(layer0).__name__} and {type(layer2).__name__}"
        )
    W_matrix = layer0.weight.detach().numpy().T  # (input_dim, hidden_dim)
    B_vector = layer0.bias.detach().numpy()  # (hidden_dim,)
    W_out_vec = layer2.weight.detach().numpy().flatten()  # (hidden_dim,)
    return W_matrix, B_vector, W_out_vec


def analytic_gradient(
    W_matrix: np.ndarray,
    B_vector: np.ndarray,
    W_out_vec: np.ndarray,
    x_state: np.ndarray,
) -> np.ndarray:
    """Compute the exact gradient of the ReLU network at *x_state*.

    Exploits the fact that activation patterns are constant within each
    linear region, so the gradient is a fixed vector there.
    """
    W_out_mat = np.diag(W_out_vec)
    input_num = W_matrix.shape[0]
    hidden_num = W_matrix.shape[1]

    # Flatten to 1-D so dot products always return scalars, regardless of
    # whether x_state was passed as (n,) or (n, 1).
    x_flat = np.asarray(x_state).flatten()

    # Activation pattern at x_state
    U = np.array(
        [
            [_step(np.dot(W_matrix[:, i], x_flat) + B_vector[i])]
            for i in range(hidden_num)
        ]
    )  # (hidden_dim, 1)

    grad = np.zeros((input_num, 1))
    for i in range(input_num):
        W_prime = W_out_mat @ W_matrix[i, :].reshape(-1, 1)
        grad[i] = (U.T @ W_prime).item()  # (1,1) → scalar
    return grad


def _dynamics_numpy(dynamics: nn.Module, x: np.ndarray) -> tuple[float, float]:
    """Evaluate dynamics at a (2, 1) numpy column vector; returns (f1, f2) floats."""
    t = torch.tensor([[x[0, 0], x[1, 0]]], dtype=torch.float32)
    out = dynamics(t)
    return float(out[0, 0]), float(out[0, 1])


def verify_point(
    W_matrix: np.ndarray,
    B_vector: np.ndarray,
    W_out_vec: np.ndarray,
    x1_val: float,
    x2_val: float,
    dynamics: nn.Module,
) -> bool:
    """Return True if the Lie derivative V_dot < 0 at (x1_val, x2_val)."""
    x = np.array([[x1_val], [x2_val]])
    grad = analytic_gradient(W_matrix, B_vector, W_out_vec, x)
    f1, f2 = _dynamics_numpy(dynamics, x)
    return float(grad[0] * f1 + grad[1] * f2) < 0


# ── Polygon geometry helpers ───────────────────────────────────────────────────


def is_point_in_polygon(
    x: float, y: float, polygon_vertices: Sequence[Vertex2D]
) -> bool:
    """Ray-casting test: True if (x, y) is strictly inside the polygon."""
    intersections = 0
    n = len(polygon_vertices)
    for i in range(n):
        xi, yi = polygon_vertices[i]
        xj, yj = polygon_vertices[(i + 1) % n]
        if (yi > y) != (yj > y):
            xinters = xi + (y - yi) * (xj - xi) / (yj - yi)
            if abs(xinters - x) < 1e-12:
                return False
            if xinters > x:
                intersections += 1
    return (intersections % 2) == 1


def zero_level_set_crosses_edge(
    v1: Vertex2D, v2: Vertex2D, f_function: Callable[[np.ndarray], float]
) -> bool:
    """Return True if the zero level set of f_function crosses the edge v1→v2."""
    x1_min_e = min(v1[0], v2[0])
    x1_max_e = max(v1[0], v2[0])
    x2_min_e = min(v1[1], v2[1])
    x2_max_e = max(v1[1], v2[1])

    # Quick check on endpoints
    f_v1 = f_function(np.array([[v1[0]], [v1[1]]]))
    f_v2 = f_function(np.array([[v2[0]], [v2[1]]]))
    if f_v1 * f_v2 < 0:
        return True

    # Refine along the edge (lightweight sampling + early exit)
    x1_diff = x1_max_e - x1_min_e
    x2_diff = x2_max_e - x2_min_e
    steps = max(int(max(x1_diff, x2_diff) * 8), 64)

    if abs(v2[0] - v1[0]) < 1e-10:  # vertical edge
        x1_on_edge = np.full(steps, v1[0])
        x2_on_edge = np.linspace(x2_min_e, x2_max_e, steps)
    else:
        t = np.linspace(0, 1, steps)
        x1_on_edge = v1[0] + t * (v2[0] - v1[0])
        x2_on_edge = v1[1] + t * (v2[1] - v1[1])

    prev = float(f_function(np.array([[x1_on_edge[0]], [x2_on_edge[0]]])))
    prev_sign = np.sign(prev)
    for x1, x2 in zip(x1_on_edge[1:], x2_on_edge[1:]):
        cur = float(f_function(np.array([[x1], [x2]])))
        cur_sign = np.sign(cur)
        if prev_sign * cur_sign < 0:
            return True
        prev_sign = cur_sign
    return False


# ── Polygon class ──────────────────────────────────────────────────────────────


class Polygon:
    """A single face of the ReLU hyperplane arrangement.

    Parameters
    ----------
    vertex_coords : list of array-like
        Ordered 2-D coordinates of the polygon vertices.
    """

    def __init__(self, vertex_coords: Sequence[Vertex2D]) -> None:
        self.vertex_coords = [np.asarray(c, dtype=float) for c in vertex_coords]

    @property
    def centroid(self) -> tuple[float, float]:
        xs = [c[0] for c in self.vertex_coords]
        ys = [c[1] for c in self.vertex_coords]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def rotate(self, gradient_vec: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the rotation that aligns gradient_vec with E_1 = [1, 0]^T.

        Parameters
        ----------
        gradient_vec : ndarray, shape (2, 1)

        Returns
        -------
        theta : float
            Rotation angle (radians).
        R : ndarray, shape (2, 2)
            Rotation matrix.
        """
        grad_flat = np.asarray(gradient_vec, dtype=float).reshape(-1)
        if grad_flat.size < 2:
            raise ValueError(
                f"gradient_vec must have at least 2 entries, got shape {np.asarray(gradient_vec).shape}"
            )
        gx, gy = float(grad_flat[0]), float(grad_flat[1])
        theta = -np.arctan2(gy, gx)
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=float,
        )
        return theta, R

    def check_gradient(
        self,
        W_matrix: np.ndarray,
        B_vector: np.ndarray,
        W_out_vec: np.ndarray,
        dynamics: nn.Module,
        max_refinements: int | None = None,
    ) -> list[Counterexample]:
        """Detect a polygon counterexample using rf1 sign-region logic.

        Rules:
        1) If rf1 intersects any polygon edge, target region count is 2; else 1.
        2) For target=1, do not discretize: test centroid only.
        3) For target=2, discretize polygon bounds and return the first interior,
           non-edge point with rf1 >= 0.
        """
        centroid_x = float(self.centroid[0])
        centroid_y = float(self.centroid[1])
        x_col = np.array([[centroid_x], [centroid_y]])
        grad = analytic_gradient(W_matrix, B_vector, W_out_vec, x_col)
        g1, g2 = float(grad[0, 0]), float(grad[1, 0])

        # ── Zero-gradient early exit ──────────────────────────────────────────
        # If ∇V = 0 then V is constant in this polygon, so V_dot = ∇V · f = 0
        # everywhere — V is not strictly decreasing.  The rotation framework
        # below is undefined (no direction to align with E_1), and would
        # fall through to checking rf1 = f1, which is unrelated to V_dot.
        # Return the centroid as a counterexample immediately.
        if np.linalg.norm(grad) < 1e-10:
            return [(float(self.centroid[0]), float(self.centroid[1]), 0.0)]

        theta = -np.arctan2(g2, g1)
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))

        def rf1(x):
            f1, f2 = _dynamics_numpy(dynamics, x)
            return cos_t * f1 - sin_t * f2

        xs_i = [float(c[0]) for c in self.vertex_coords] + [
            float(self.vertex_coords[0][0])
        ]
        ys_i = [float(c[1]) for c in self.vertex_coords] + [
            float(self.vertex_coords[0][1])
        ]

        # Step 1: edge-intersection test for rf1 determines target region count.
        rf1_crosses_edge = False
        for j in range(len(xs_i) - 1):
            x1_start, x2_start = xs_i[j], ys_i[j]
            x1_end, x2_end = xs_i[j + 1], ys_i[j + 1]
            if zero_level_set_crosses_edge((x1_start, x2_start), (x1_end, x2_end), rf1):
                rf1_crosses_edge = True
                break
        num_regions = 2 if rf1_crosses_edge else 1

        # Step 2: one-region case -> centroid-only classification.
        if num_regions == 1:
            x_c = np.array([[centroid_x], [centroid_y]])
            centroid_val = float(rf1(x_c))
            if centroid_val >= 0.0:
                f1_c, f2_c = _dynamics_numpy(dynamics, x_c)
                vdot = g1 * float(f1_c) + g2 * float(f2_c)
                return [(centroid_x, centroid_y, vdot)]
            return []

        # Step 3: two-region case -> discretize bounds and stop at first CEX.
        del max_refinements  # intentionally unused in discretized variant

        def _is_on_edge(px: float, py: float, tol: float = 1e-10) -> bool:
            for j in range(len(xs_i) - 1):
                x1_start, y1_start = xs_i[j], ys_i[j]
                x1_end, y1_end = xs_i[j + 1], ys_i[j + 1]
                dx = x1_end - x1_start
                dy = y1_end - y1_start
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq <= tol:
                    continue
                t = ((px - x1_start) * dx + (py - y1_start) * dy) / seg_len_sq
                if t < -tol or t > 1.0 + tol:
                    continue
                proj_x = x1_start + t * dx
                proj_y = y1_start + t * dy
                dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
                if dist_sq <= tol * tol:
                    return True
            return False

        x1_poly = np.linspace(min(xs_i), max(xs_i), 100)
        x2_poly = np.linspace(min(ys_i), max(ys_i), 100)
        for x1 in x1_poly:
            for x2 in x2_poly:
                x1f = float(x1)
                x2f = float(x2)
                if _is_on_edge(x1f, x2f):
                    continue
                if not is_point_in_polygon(x1f, x2f, self.vertex_coords):
                    continue
                x_point = np.array([[x1f], [x2f]])
                if float(rf1(x_point)) >= 0.0:
                    f1_p, f2_p = _dynamics_numpy(dynamics, x_point)
                    vdot_p = g1 * float(f1_p) + g2 * float(f2_p)
                    return [(x1f, x2f, vdot_p)]

        return []


# ── Main pipeline ──────────────────────────────────────────────────────────────


def _check_poly_worker(args: tuple) -> list[Counterexample]:
    vertex_coords, W_matrix, B_vector, W_out_vec, dynamics = args
    poly = Polygon(vertex_coords)
    return poly.check_gradient(W_matrix, B_vector, W_out_vec, dynamics)


def full_method(
    problem: LyapunovProblem,
) -> tuple[list[tuple[float, float, float]], list[list[np.ndarray]], None]:
    """End-to-end hyperplane verification pipeline.

    Parameters
    ----------
    problem: LyapunovProblem
        Includes
            nn_lyapunov: nn.Module, a single-hidden-layer ReLU network
                (network[0]=Linear, network[1]=ReLU, network[2]=Linear).
            dynamics : nn.Module, system dynamics
            region : Tensor, shape (2, 2), domain bounds [[x1_min, x1_max], [x2_min, x2_max]].

    Returns
    -------
    counterexamples : list of (float, float, float)  — (x1, x2, V_dot)
    cells : list of list[ndarray]  — ordered vertex coordinates per cell
    """
    x1_min, x1_max = problem.region[0, 0].item(), problem.region[0, 1].item()
    x2_min, x2_max = problem.region[1, 0].item(), problem.region[1, 1].item()

    W_matrix, B_vector, W_out_vec = extract_weights(problem.nn_lyapunov)

    bbox_hs = build_bbox_halfspaces(x1_min, x1_max, x2_min, x2_max)

    print("Enumerating cells via BFS...")
    t0 = time.perf_counter()
    cells = enumerate_cells_bfs(
        W_matrix, B_vector, bbox_hs, x1_min, x1_max, x2_min, x2_max
    )
    print(f"  done ({time.perf_counter() - t0:.3f}s)  — {len(cells)} cells")

    print("Running verification...")
    t0 = time.perf_counter()
    dynamics_cpu = problem.dynamics.cpu()
    work_items = [
        (cell.tolist(), W_matrix, B_vector, W_out_vec, dynamics_cpu) for cell in cells
    ]
    with Pool() as pool:
        results = pool.map(_check_poly_worker, work_items)
    counterexamples = [cex for batch in results for cex in batch]
    print(
        f"  done ({time.perf_counter() - t0:.3f}s)  "
        f"— {len(counterexamples)} counterexamples"
    )

    cell_coords = [cell.tolist() for cell in cells]
    return counterexamples, cell_coords, None
