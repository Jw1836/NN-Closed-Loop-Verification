"""
hyperplane_verifier.py

Hyperplane arrangement verification for neural Lyapunov functions.

Pipeline:
  1. enumerate_cells_bfs           - BFS over activation patterns to find all cells
  2. Polygon.check_gradient        - rotate, count sign regions, check dynamics
  3. full_method                   - top-level orchestrator; returns counterexamples
"""

from collections import deque

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
from typing import Sequence, cast
from torch import nn

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


def build_bbox_halfspaces(region: np.ndarray) -> np.ndarray:
    """Build bounding box halfspaces in the form [a1, ..., an, b] where a·x + b <= 0.

    Parameters
    ----------
    region : ndarray, shape (state_dim, 2)
        Each row is [min, max] for that dimension.

    Returns
    -------
    ndarray, shape (2*state_dim, state_dim+1)
    """
    state_dim = region.shape[0]
    hs = np.zeros((2 * state_dim, state_dim + 1))
    for d in range(state_dim):
        # x_d >= region[d, 0]  =>  -x_d + region[d, 0] <= 0
        hs[2 * d, d] = -1.0
        hs[2 * d, -1] = region[d, 0]
        # x_d <= region[d, 1]  =>   x_d - region[d, 1] <= 0
        hs[2 * d + 1, d] = 1.0
        hs[2 * d + 1, -1] = -region[d, 1]
    return hs


def build_cell_halfspaces(
    W_matrix: np.ndarray,
    B_vector: np.ndarray,
    bbox_hs: np.ndarray,
    pattern: ActivationPattern,
) -> np.ndarray:
    """Build halfspaces for a polyhedral cell with the given activation pattern.

    Each neuron i contributes: if active, -w_i·x - b_i <= 0; if inactive, w_i·x + b_i <= 0.
    Combined with the bounding-box halfspaces.
    """
    n_hidden = W_matrix.shape[1]
    n_bbox = bbox_hs.shape[0]
    state_dim = W_matrix.shape[0]
    hs = np.empty((n_hidden + n_bbox, state_dim + 1))
    hs[:n_bbox] = bbox_hs
    for i in range(n_hidden):
        w = W_matrix[:, i]
        b = B_vector[i]
        if pattern[i]:  # active: w·x + b >= 0  =>  -w·x - b <= 0
            hs[n_bbox + i] = np.concatenate([-w, [-b]])
        else:  # inactive: w·x + b <= 0
            hs[n_bbox + i] = np.concatenate([w, [b]])
    return hs


def find_chebyshev_center(halfspaces: np.ndarray) -> np.ndarray | None:
    """Find a strictly interior point via the Chebyshev center LP.

    Maximizes the inscribed ball radius r subject to a_j·x + b_j + ||a_j||·r <= 0.
    Each row of *halfspaces* is ``[a1, ..., an, b]`` defining ``a·x + b <= 0``.
    Returns shape ``(state_dim,)`` or ``None`` if infeasible.

    This is a slightly more expensive but robust way to find an interior point.
    It has the advantage of returning ``None`` for infeasible solutions,
    meaning they have effectively zero area.
    """
    state_dim = halfspaces.shape[1] - 1
    A_hs = halfspaces[:, :-1]
    b_hs = halfspaces[:, -1]
    norms = np.linalg.norm(A_hs, axis=1, keepdims=True)

    # LP: minimize -r  subject to  A_hs @ x + norms * r <= -b_hs, r >= 0
    A_lp = np.hstack([A_hs, norms])
    b_lp = -b_hs
    c = np.zeros(state_dim + 1)
    c[-1] = -1.0

    result = linprog(
        c,
        A_ub=A_lp,
        b_ub=b_lp,
        bounds=[(None, None)] * state_dim + [(0, None)],
        method="highs",
    )
    if result.success and result.x[-1] > 1e-10:
        return result.x[:state_dim]
    return None


def enumerate_cells_bfs(
    W_matrix: np.ndarray,
    B_vector: np.ndarray,
    bbox_hs: np.ndarray,
    region: np.ndarray,
) -> list[ConvexHull]:
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

    Returns ``ConvexHull`` objects so callers can access both vertices
    (``hull.points[hull.vertices]``) and halfplane equations
    (``hull.equations``) for point-in-cell tests.
    """
    # Seed the BFS from the domain center
    state_dim = region.shape[0]
    center = region.mean(axis=1)
    sigma_0 = compute_activation_pattern(W_matrix, B_vector, center)

    visited: set[ActivationPattern] = {sigma_0}
    queue: deque[tuple[ActivationPattern, np.ndarray]] = deque([(sigma_0, center)])
    cells: list[ConvexHull] = []

    while queue:
        sigma, hint = queue.popleft()

        # Build the halfspace system: n ReLU constraints + 2*state_dim
        # bounding-box walls.  Each row is [a1, ..., an, b] meaning a·x + b ≤ 0.
        hs = build_cell_halfspaces(W_matrix, B_vector, bbox_hs, sigma)

        # --- Find a strictly interior point (required by HalfspaceIntersection) ---
        # Fast path: the hint (reflected interior of the parent cell) is often
        # already strictly feasible, saving an LP solve.
        hint_violations = hs[:, :-1] @ hint + hs[:, -1]
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
            hs_obj = HalfspaceIntersection(
                hs,
                interior,
                incremental=False,
            )
        except QhullError:
            continue  # degenerate (e.g. numerically flat cell)

        raw_verts = hs_obj.intersections

        # HalfspaceIntersection can return duplicate or near-collinear points.
        # ConvexHull filters these and gives vertices in CCW order.
        if len(raw_verts) < state_dim + 1:
            continue
        try:
            hull = ConvexHull(raw_verts, incremental=False)
        except QhullError:
            continue  # degenerate (collinear points, etc.)
        if len(hull.vertices) < state_dim + 1:
            continue

        cells.append(hull)

        # --- Discover neighboring cells ---
        # A ReLU hyperplane is "tight" if at least one vertex of this cell
        # lies on it (|w_i · v + b_i| < tol).  Flipping that neuron's
        # activation bit gives the adjacent cell on the other side.
        # Vectorized: evaluate all n hyperplanes at all k vertices at once.
        ordered_verts = raw_verts[hull.vertices]
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


def _step(x: float) -> int:
    """Heaviside step function."""
    return 1 if x >= 0 else 0


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
    hidden_num = W_matrix.shape[1]

    # Flatten to 1-D so dot products always return scalars, regardless of
    # whether x_state was passed as (n,) or (n, 1).
    x_flat = np.asarray(x_state).flatten()

    # Activation pattern at x_state: (hidden_dim,) binary vector
    U = np.array(
        [_step(np.dot(W_matrix[:, i], x_flat) + B_vector[i]) for i in range(hidden_num)]
    )

    return (W_matrix @ (U * W_out_vec)).reshape(-1, 1)


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


# ── ConvexHull functions ───────────────────────────────────────────────────────


def align_basis(
    cell: ConvexHull,
    gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate cell vertices so that *gradient* aligns with E_1, matrix-free.

    Uses a Householder reflection Q = I - 2 v v^T / (v^T v) with
    v = g_hat - e_1, giving Q @ g_hat = +e_1.  The full matrix is never
    formed; vertices are rotated via the rank-1 update and Q's first row
    is returned as a vector for downstream rf1 computation.

    Parameters
    ----------
    cell : ConvexHull
        Polytope whose vertices live in R^n.
    gradient : ndarray, shape (n,) or (n, 1)
        Non-zero gradient of the Lyapunov function in this cell.

    Returns
    -------
    q1 : ndarray, shape (n,)
        First row of Q.  ``q1 @ f`` gives the rotated first component of f.
    rotated_vertices : ndarray, shape (num_vertices, n)
        Cell vertices after applying Q.
    """
    g = np.asarray(gradient, dtype=float).flatten()
    n = g.shape[0]
    g_hat = g / np.linalg.norm(g)

    e1 = np.zeros(n)
    e1[0] = 1.0

    v = g_hat - e1
    vtv = np.dot(v, v)

    vertices = cell.points[cell.vertices]

    if vtv < 1e-14:
        # gradient already aligned with e_1
        return e1.copy(), vertices.copy()

    # Matrix-free rotation: Q @ x = x - 2 v (v^T x) / (v^T v)
    # For rows of `vertices` (shape m x n): proj has shape (m,)
    proj = (vertices @ v) / vtv
    rotated_vertices = vertices - 2.0 * np.outer(proj, v)

    # First row of Q: e_1 - 2 v[0] v / (v^T v)
    q1 = e1 - 2.0 * v[0] * v / vtv

    return q1, rotated_vertices
