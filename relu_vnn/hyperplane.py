"""
hyperplane_verifier.py

Hyperplane arrangement verification for neural Lyapunov functions.

Pipeline:
  1. enumerate_cells_bfs           - BFS over activation patterns to find all cells
  2. align_basis / check_decrease  - rotate cell, verify Lie derivative per cell
"""

import itertools
import logging
from collections import deque
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from typing import Sequence, cast
from torch import nn

logger = logging.getLogger(__name__)

# float32 numerical tolerance (~10× machine epsilon)
EPS: float = 1e-6

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
    if result.success and result.x[-1] > EPS:
        return result.x[:state_dim]
    return None


def enumerate_cells_bfs(
    W_matrix: np.ndarray,
    B_vector: np.ndarray,
    bbox_hs: np.ndarray,
    region: np.ndarray,
) -> list[HalfspaceIntersection]:
    """Enumerate all non-empty cells of the ReLU hyperplane arrangement via BFS.

    Each hidden neuron defines a hyperplane ``w_i · x + b_i = 0`` that splits
    the domain into two half-spaces.  A *cell* is the convex polytope formed
    by the intersection of one half-space per neuron (the "activation pattern")
    with the bounding box — i.e. the region where a specific set of ReLUs are on/off.
    Within each cell the network is affine, so ``∇V`` is constant.

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

    Returns ``HalfspaceIntersection`` objects so callers can access both
    * vertices (``cell.intersections``) and
    * halfplane equations (``cell.halfspaces``) for point-in-cell tests.
    """
    from scipy.spatial import ConvexHull, QhullError

    # Seed the BFS from the domain center plus offset points at ±25% of each
    # dimension's range.  Multiple seeds guard against the center landing in a
    # near-degenerate cell (e.g. when many hyperplanes cluster near the origin
    # after training), which would stall BFS with an empty queue.
    state_dim = region.shape[0]
    center = region.mean(axis=1)
    quarter = (region[:, 1] - region[:, 0]) * 0.25
    seed_points = [center] + [
        center + quarter * np.array(signs)
        for signs in itertools.product([-1.0, 1.0], repeat=state_dim)
    ]

    # BFS invariant: `visited` contains every activation pattern that has ever
    # been enqueued, so each pattern is processed at most once.  Completeness
    # relies on the hyperplane arrangement graph being connected within the
    # bounding box — every non-empty cell shares at least one facet (a tight
    # ReLU hyperplane) with another cell, so BFS from any seed reaches all of
    # them.
    visited: set[ActivationPattern] = set()
    queue: deque[tuple[ActivationPattern, np.ndarray]] = deque()
    for pt in seed_points:
        sigma = compute_activation_pattern(W_matrix, B_vector, pt)
        if sigma not in visited:
            visited.add(sigma)
            queue.append((sigma, pt))
    cells: list[HalfspaceIntersection] = []
    total_volume = 0.0  # accumulated for the sanity check at the end

    # `is_seed` enables diagnostic printing only for the very first cell.
    # If the seed cell is skipped, something is likely wrong with the network
    # or domain configuration, so we emit a warning.  After the seed, skipped
    # cells are expected (BFS explores candidate neighbors that may be empty).
    is_seed = True
    while queue:
        sigma, hint = queue.popleft()

        # Build the halfspace system: n ReLU constraints + 2*state_dim
        # bounding-box walls.  Each row is [a1, ..., an, b] meaning a·x + b ≤ 0.
        hs = build_cell_halfspaces(W_matrix, B_vector, bbox_hs, sigma)

        # --- Find a strictly interior point (required by HalfspaceIntersection) ---
        # Fast path: the hint (reflected interior of the parent cell) is often
        # already strictly feasible, saving an LP solve.
        hint_violations = hs[:, :-1] @ hint + hs[:, -1]
        if np.all(hint_violations < -EPS):
            interior = hint
        else:
            # Slow path: solve the Chebyshev-center LP for the largest
            # inscribed ball.  Returns None when the cell is infeasible
            # (activation pattern has no realizable region in the domain).
            interior = find_chebyshev_center(hs)
            if interior is None:
                # Skipping infeasible / degenerate cells does not break
                # completeness: an infeasible pattern has no realizable region
                # in the domain (it was a candidate neighbor that turned out
                # empty).  A zero-volume cell has no interior and therefore no
                # tight hyperplanes that could lead to undiscovered neighbors.
                if is_seed:
                    logger.warning(
                        "[BFS seed] Chebyshev center LP infeasible — seed cell skipped"
                    )
                is_seed = False
                continue

        # --- Compute cell vertices ---
        try:
            hs_obj = HalfspaceIntersection(
                hs,
                interior,
                incremental=False,
            )
        except QhullError as e:
            if is_seed:
                logger.warning("[BFS seed] QhullError in HalfspaceIntersection: %s", e)
            is_seed = False
            continue  # degenerate (e.g. numerically flat cell)

        raw_verts = hs_obj.intersections

        # HalfspaceIntersection can return duplicate or near-collinear points.
        # ConvexHull filters these and gives vertices in CCW order.
        if len(raw_verts) < state_dim + 1:
            if is_seed:
                logger.warning(
                    "[BFS seed] too few raw vertices: %d < %d",
                    len(raw_verts),
                    state_dim + 1,
                )
            is_seed = False
            continue
        try:
            hull = ConvexHull(raw_verts, incremental=False)
        except QhullError as e:
            if is_seed:
                logger.warning("[BFS seed] QhullError in ConvexHull: %s", e)
            is_seed = False
            continue  # degenerate (collinear points, etc.)
        if len(hull.vertices) < state_dim + 1:
            if is_seed:
                logger.warning(
                    "[BFS seed] too few hull vertices: %d < %d",
                    len(hull.vertices),
                    state_dim + 1,
                )
            is_seed = False
            continue

        cells.append(hs_obj)
        total_volume += hull.volume

        # --- Discover neighboring cells ---
        # Evaluate all neuron pre-activations w_i · v + b_i at every vertex.
        # A hyperplane is "tight" if any vertex has |pre-activation| < EPS,
        # meaning the cell shares a facet with the cell on the other side of
        # that hyperplane — this is the adjacency condition in the arrangement
        # graph.  Vectorized: all n hyperplanes × all k vertices at once.
        ordered_verts = raw_verts[hull.vertices]
        all_values = ordered_verts @ W_matrix + B_vector  # (k, n_hidden)
        tight_mask = np.any(np.abs(all_values) < EPS, axis=0)  # (n_hidden,)

        for i in np.where(tight_mask)[0]:
            # Flip bit i to get the neighbor's activation pattern
            neighbor = list(sigma)
            neighbor[i] = not neighbor[i]
            neighbor_pat = tuple(neighbor)
            if neighbor_pat not in visited:
                visited.add(neighbor_pat)
                # Reflect interior across hyperplane i: x' = x - 2·((w·x+b)/(w·w))·w.
                # Since `interior` is strictly inside the current cell, the
                # reflected point typically lands strictly inside the neighbor
                # cell on the other side of hyperplane i.  This provides a
                # "free" interior point, avoiding the expensive Chebyshev LP.
                w = W_matrix[:, i]
                b = B_vector[i]
                dist = (w @ interior + b) / (w @ w)
                neighbor_hint = interior - 2 * dist * w
                queue.append((neighbor_pat, neighbor_hint))

    # ── Sanity check: cells must tile the bounding box exactly ──────────────
    # If the cells partition the domain correctly, their volumes must sum to
    # the bounding-box volume.  A shortfall means cells were missed; an excess
    # means cells overlap.  Either way, verification results would be unsound.
    bbox_volume = float(np.prod(region[:, 1] - region[:, 0]))
    rel_error = (
        abs(total_volume - bbox_volume) / bbox_volume if bbox_volume > 0 else 0.0
    )
    if rel_error > 1e-6:
        raise RuntimeError(
            f"Cell enumeration volume mismatch: "
            f"sum(cell volumes)={total_volume:.8g} vs "
            f"bbox volume={bbox_volume:.8g} "
            f"(relative error={rel_error:.2e}). "
            f"This indicates missing or overlapping cells."
        )

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


# ── Cell functions ───────────────────────────────────────────────


def align_basis(gradient: np.ndarray) -> np.ndarray:
    """First row of the Householder reflection mapping *gradient* to basis e1: ``+||g||·e₁``, matrix-free.

    Algorithm 5.1.1 from Golub & Van Loan, *Matrix Computations* 4th ed,
    adapted for a single reflection (no ``v(1)=1`` normalization or packed storage).
    """
    g = np.asarray(gradient, dtype=float).flatten()
    n = g.shape[0]
    mu = np.linalg.norm(g)  # textbook: mu = ||x||
    if mu < EPS:
        raise ValueError("Gradient is zero; cannot align basis.")

    e_1 = np.zeros(n)
    e_1[0] = 1.0

    sigma = np.dot(g[1:], g[1:])  # textbook: sigma = x(2:m)^T x(2:m)

    # Householder vector v: only v[0] differs from g, v[1:] = g[1:]
    v = g.copy()
    if sigma < EPS:
        if g[0] >= 0:
            return e_1.copy()  # already along +e_1, Q = I
        else:
            v[0] = g[0] - mu  # along -e_1, no cancellation
    elif g[0] <= 0:
        v[0] = g[0] - mu  # no cancellation: both terms same sign
    else:
        v[0] = -sigma / (g[0] + mu)  # avoids cancellation when g[0] > 0

    # beta = 2/(v^T v); textbook folds this into a normalized v(1)=1 form we don't need
    vtv = np.dot(v, v)
    beta = 2.0 / vtv

    # Only the first row of Q = I - beta*v*v^T is needed downstream
    q1 = e_1 - (beta * v[0]) * v

    return q1


def contains_point(cell: HalfspaceIntersection, point: np.ndarray) -> bool:
    """Uses the matrix inequality Ax <= b of the halfspaces to test if
    a point lies inside a convex polytope."""
    # Each row of cell.halfspaces is [A_row | b] with constraint A_row @ x + b <= 0
    return bool(np.all(cell.halfspaces[:, :-1] @ point + cell.halfspaces[:, -1] <= 0))
