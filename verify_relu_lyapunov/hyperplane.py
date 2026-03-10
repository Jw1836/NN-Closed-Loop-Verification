"""
hyperplane_verifier.py

Hyperplane arrangement verification for neural Lyapunov functions.

Pipeline:
  1. find_all_intersection_points  - ReLU hyperplane x hyperplane + domain edges
  2. get_planar_graph              - build planar graph from intersection points
  3. polygons_from_igraph_mcb      - extract polygon faces via minimum cycle basis
  4. Polygon.check_gradient        - rotate, count sign regions, run Amsden-Hirt grid
  5. full_method                   - top-level orchestrator; returns counterexamples
"""

import time
import numpy as np
import networkx as nx
import igraph as ig
import torch
from multiprocessing import Pool
from typing import cast
from torch import nn, Tensor
from .lyapunov import LyapunovProblem


# ── Geometry helpers ───────────────────────────────────────────────────────────


def is_within_rect(point, x1_min, x1_max, x2_min, x2_max, tol=1e-8):
    """Return True if point lies inside (or on) the axis-aligned rectangle."""
    return (x1_min - tol <= point[0] <= x1_max + tol) and (
        x2_min - tol <= point[1] <= x2_max + tol
    )


def find_intersection(plane_1_normal, plane_1_bias, plane_2_normal, plane_2_bias):
    """Solve for the intersection of two lines given in normal/bias form.

    Line equation: normal . x + bias = 0
    Returns None if the lines are parallel (det < tol).
    """
    A = np.array([plane_1_normal, plane_2_normal])
    det = np.linalg.det(A)
    tol = 1e-6
    if abs(det) < tol:
        return None
    b = np.array([-plane_1_bias, -plane_2_bias])
    return np.linalg.solve(A, b)


def find_all_intersection_points(W_matrix, B_vector, x1_min, x1_max, x2_min, x2_max):
    """Find all pairwise intersections of ReLU hyperplanes, plus their
    intersections with the four domain edges, clipped to the rectangle.

    Parameters
    ----------
    W_matrix : ndarray, shape (input_dim, hidden_dim)
        Columns are weight vectors into each hidden neuron.
    B_vector : ndarray, shape (hidden_dim,)
        Biases of the hidden layer.
    x1_min, x1_max, x2_min, x2_max : float
        Domain bounds.
    """
    intersection_points = []

    # Pairwise intersections of hyperplanes
    for i in range(W_matrix.shape[1]):
        w_i = W_matrix[:, i]
        b_i = B_vector[i]
        for j in range(i + 1, W_matrix.shape[1]):
            w_j = W_matrix[:, j]
            b_j = B_vector[j]
            pt = find_intersection(w_i, b_i, w_j, b_j)
            if pt is not None:
                intersection_points.append(pt)

    # Intersections of each hyperplane with the four rectangle edges
    for i in range(W_matrix.shape[1]):
        w_i = W_matrix[:, i]
        b_i = B_vector[i]

        # Bottom edge: x2 = x2_min  →  [0,1]·x + (-x2_min) = 0
        pt = find_intersection(w_i, b_i, np.array([0, 1]), -x2_min)
        if pt is not None and is_within_rect(pt, x1_min, x1_max, x2_min, x2_max):
            intersection_points.append(pt)

        # Top edge: x2 = x2_max
        pt = find_intersection(w_i, b_i, np.array([0, 1]), -x2_max)
        if pt is not None and is_within_rect(pt, x1_min, x1_max, x2_min, x2_max):
            intersection_points.append(pt)

        # Left edge: x1 = x1_min  →  [1,0]·x + (-x1_min) = 0
        pt = find_intersection(w_i, b_i, np.array([1, 0]), -x1_min)
        if pt is not None and is_within_rect(pt, x1_min, x1_max, x2_min, x2_max):
            intersection_points.append(pt)

        # Right edge: x1 = x1_max
        pt = find_intersection(w_i, b_i, np.array([1, 0]), -x1_max)
        if pt is not None and is_within_rect(pt, x1_min, x1_max, x2_min, x2_max):
            intersection_points.append(pt)

    # Add the four rectangle corners
    for corner in [
        (x1_min, x2_min),
        (x1_min, x2_max),
        (x1_max, x2_min),
        (x1_max, x2_max),
    ]:
        intersection_points.append(corner)

    # Discard any points that fell outside the rectangle
    in_rect = [
        p
        for p in intersection_points
        if is_within_rect(p, x1_min, x1_max, x2_min, x2_max)
    ]

    # De-duplicate near-coincident points (tol=1e-6).
    # Many hidden neurons have biases driven near zero by the V(0)=0 loss term,
    # so all their decision boundaries intersect within a tiny neighbourhood of
    # the origin.  Without dedup these micro-points form hundreds of zero-area
    # "polygons" that produce false-positive counterexamples.
    dedup = []
    for pt in in_rect:
        pt_arr = np.asarray(pt)
        if not any(np.linalg.norm(pt_arr - np.asarray(kept)) < 1e-6 for kept in dedup):
            dedup.append(pt)
    return dedup


def _connect_consecutive_on_edge(points_on_edge, edge_name, edge_list):
    """Sort points along one rectangle edge and connect consecutive pairs."""
    if len(points_on_edge) < 2:
        return edge_list
    if edge_name in ("left", "right"):
        points_on_edge.sort(key=lambda p: p[1][1])  # sort by x2
    else:
        points_on_edge.sort(key=lambda p: p[1][0])  # sort by x1
    for k in range(len(points_on_edge) - 1):
        edge_list.append((points_on_edge[k][0], points_on_edge[k + 1][0]))
    return edge_list


def get_planar_graph(
    intersection_points, W_matrix, B_vector, x1_min, x1_max, x2_min, x2_max, tol=1e-6
):
    """Build the planar graph whose vertices are the intersection points.

    Returns
    -------
    vertex_dict : dict[str, array-like]
        Maps vertex label 'v0', 'v1', ... to 2-D coordinates.
    edge_list : list of (str, str) tuples
        Edges between vertex labels.
    """
    vertex_dict = {
        f"v{i}": intersection_points[i] for i in range(len(intersection_points))
    }
    edge_list = []

    # Collect points on each rectangle edge for boundary connectivity
    left_pts, right_pts, bottom_pts, top_pts = [], [], [], []
    for i, pt in enumerate(intersection_points):
        x1, x2 = pt[0], pt[1]
        label = (f"v{i}", pt)
        if abs(x1 - x1_min) < tol:
            left_pts.append(label)
        if abs(x1 - x1_max) < tol:
            right_pts.append(label)
        if abs(x2 - x2_min) < tol:
            bottom_pts.append(label)
        if abs(x2 - x2_max) < tol:
            top_pts.append(label)

    edge_list = _connect_consecutive_on_edge(left_pts, "left", edge_list)
    edge_list = _connect_consecutive_on_edge(right_pts, "right", edge_list)
    edge_list = _connect_consecutive_on_edge(bottom_pts, "bottom", edge_list)
    edge_list = _connect_consecutive_on_edge(top_pts, "top", edge_list)

    # Connect points that lie on the same ReLU hyperplane
    for plane_idx in range(W_matrix.shape[1]):
        w_i = W_matrix[:, plane_idx]
        b_i = B_vector[plane_idx]

        on_plane = [
            (i, pt)
            for i, pt in enumerate(intersection_points)
            if abs(np.dot(pt, w_i) + b_i) < tol
        ]

        # Sort along the plane: if the normal is more horizontal, sort by x2
        if abs(w_i[0]) > abs(w_i[1]):
            on_plane.sort(key=lambda p: p[1][1])
        else:
            on_plane.sort(key=lambda p: p[1][0])

        for k in range(len(on_plane) - 1):
            edge_list.append((f"v{on_plane[k][0]}", f"v{on_plane[k + 1][0]}"))

    return vertex_dict, edge_list


# ── Graph / polygon extraction ─────────────────────────────────────────────────


def ordered_nodes_from_cycle_edges(g_ig, cycle_edge_indices):
    """Convert a cycle (given as igraph edge indices) to an ordered node list."""
    cycle_edges = []
    for idx in cycle_edge_indices:
        e = g_ig.es[idx]
        u = g_ig.vs[e.source]["_nx_name"]
        v = g_ig.vs[e.target]["_nx_name"]
        cycle_edges.append((u, v))

    adjacency = {}
    for u, v in cycle_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    if len(adjacency) < 3:
        return []

    start = next(iter(adjacency))
    ordered = [start]
    prev, current = None, start

    while True:
        neighbors = adjacency[current]
        candidates = [n for n in neighbors if n != prev]
        if not candidates:
            break
        next_node = candidates[0]
        if next_node == start:
            break
        ordered.append(next_node)
        prev, current = current, next_node
        if len(ordered) > len(adjacency):  # safety guard
            break

    if len(set(ordered)) != len(adjacency):
        return list(adjacency.keys())
    return ordered


def polygons_from_igraph_mcb(g_ig):
    """Extract all polygon faces from the planar graph via minimum cycle basis."""
    mcb_edges = g_ig.minimum_cycle_basis()
    polygons = []
    for cycle_edge_indices in mcb_edges:
        ordered = ordered_nodes_from_cycle_edges(g_ig, cycle_edge_indices)
        if len(ordered) >= 3:
            polygons.append(ordered)
    return polygons


# ── Neural network utilities ───────────────────────────────────────────────────


def _step(x):
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


def analytic_gradient(W_matrix, B_vector, W_out_vec, x_state):
    """Compute the exact gradient of the ReLU network at x_state.

    Exploits the fact that activation patterns are constant within each
    linear region, so the gradient is a fixed vector there.

    Parameters
    ----------
    W_matrix : ndarray, shape (input_dim, hidden_dim)
    B_vector : ndarray, shape (hidden_dim,)
    W_out_vec : ndarray, shape (hidden_dim,)
    x_state : array-like, shape (input_dim, 1)
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


def _dynamics_numpy(dynamics: nn.Module, x: np.ndarray):
    """Evaluate dynamics at a (2, 1) numpy column vector; returns (f1, f2) floats."""
    t = torch.tensor([[x[0, 0], x[1, 0]]], dtype=torch.float32)
    out = dynamics(t)
    return float(out[0, 0]), float(out[0, 1])


def verify_point(
    W_matrix, B_vector, W_out_vec, x1_val, x2_val, dynamics: nn.Module
) -> bool:
    """Return True if the Lie derivative V_dot < 0 at (x1_val, x2_val)."""
    x = np.array([[x1_val], [x2_val]])
    grad = analytic_gradient(W_matrix, B_vector, W_out_vec, x)
    f1, f2 = _dynamics_numpy(dynamics, x)
    return float(grad[0] * f1 + grad[1] * f2) < 0


# ── Polygon geometry helpers ───────────────────────────────────────────────────


def is_point_in_polygon(x, y, polygon_vertices):
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


def zero_level_set_crosses_edge(v1, v2, f_function) -> bool:
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


def amsden_hirt_grid(
    polygon_vertices: list[float],
    N1: int,
    N2: int,
    max_iter: int = 250,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a boundary-fitted interior grid for a polygon via SOR (Amsden-Hirt).

    Maps the polygon boundary onto the perimeter of an (N1 x N2) structured grid,
    distributing points proportionally to edge length, then relaxes the interior
    to a smooth (harmonic) mapping using successive over-relaxation (SOR).

    Parameters
    ----------
    polygon_vertices : list of (float, float)
        Ordered polygon vertices (open — the closing vertex is NOT repeated).
        This list is not modified.
    N1, N2 : int
        Grid dimensions. Total boundary slots = 2*N1 + 2*N2 - 4.
    max_iter : int
        Maximum SOR iterations.
    tol : float
        Convergence threshold on the maximum point displacement per iteration.

    Returns
    -------
    X, Y : ndarray, shape (N1, N2)
        Physical coordinates of grid nodes.
    """
    # Close the polygon without mutating the caller's list.
    verts = list(polygon_vertices) + [polygon_vertices[0]]
    n_edges = len(verts) - 1

    # Optimal SOR relaxation factor for a Laplace problem on a rectangular grid.
    # Formula: ω = 4 / (2 + √(4 - (cos(π/N1) + cos(π/N2))²))
    omega = 4 / (2 + np.sqrt(4 - (np.cos(np.pi / N1) + np.cos(np.pi / N2)) ** 2))

    # Number of boundary slots on the structured grid perimeter.
    total_pts = 2 * N1 + 2 * N2 - 4

    # ── Boundary point distribution ────────────────────────────────────────────
    # Allocate boundary slots proportionally to each polygon edge's arc length.
    lengths = [
        np.linalg.norm(np.array(verts[i + 1]) - np.array(verts[i]))
        for i in range(n_edges)
    ]
    perimeter = sum(lengths)

    # Assign integer slot counts to each edge via floor.  Do NOT clamp to 1 —
    # short edges legitimately get 0 slots, and clamping inflates the total
    # above total_pts before the remainder correction, causing a mismatch.
    # The remainder (always in [-n_edges+1, n_edges-1]) goes to the last edge.
    n_per_edge = [int((edge_len / perimeter) * total_pts) for edge_len in lengths]
    remainder = total_pts - sum(n_per_edge)
    n_per_edge[-1] += remainder  # guaranteed non-negative after pure floor

    boundary_pts = []
    for i in range(n_edges):
        n = n_per_edge[i]
        if n <= 0:
            continue
        p0 = np.array(verts[i])
        p1 = np.array(verts[i + 1])
        xs = np.linspace(p0[0], p1[0], n, endpoint=False)
        ys = np.linspace(p0[1], p1[1], n, endpoint=False)
        boundary_pts.extend(zip(xs.tolist(), ys.tolist()))

    if len(boundary_pts) != total_pts:
        raise ValueError(
            f"Boundary point count mismatch: expected {total_pts}, got {len(boundary_pts)}"
        )

    # ── Assign boundary points to the structured grid perimeter ───────────────
    # Walk the perimeter counter-clockwise:
    #   bottom row  (j=0):      i = 0 … N1-1          (N1 pts)
    #   right column (i=N1-1):  j = 1 … N2-2          (N2-2 pts)
    #   top row     (j=N2-1):   i = N1-1 … 0          (N1 pts)
    #   left column  (i=0):     j = N2-2 … 1          (N2-2 pts)
    X = np.zeros((N1, N2), dtype=float)
    Y = np.zeros((N1, N2), dtype=float)
    k = 0
    for row in range(N1):  # bottom (j=0)
        X[row, 0], Y[row, 0] = boundary_pts[k]
        k += 1
    for col in range(1, N2 - 1):  # right (i=N1-1)
        X[N1 - 1, col], Y[N1 - 1, col] = boundary_pts[k]
        k += 1
    for row in range(N1 - 1, -1, -1):  # top (j=N2-1)
        X[row, N2 - 1], Y[row, N2 - 1] = boundary_pts[k]
        k += 1
    for col in range(N2 - 2, 0, -1):  # left (i=0)
        X[0, col], Y[0, col] = boundary_pts[k]
        k += 1

    # ── Initial interior guess ─────────────────────────────────────────────────
    # Bilinear interpolation between the left (i=0) and right (i=N2-1) columns.
    # This gives the SOR a reasonable starting point.
    for col in range(1, N2 - 1):
        for row in range(1, N1 - 1):
            s = row / (N1 - 1)
            X[row, col] = (1 - s) * X[0, col] + s * X[N1 - 1, col]
            Y[row, col] = (1 - s) * Y[0, col] + s * Y[N1 - 1, col]

    # ── SOR relaxation (Gauss-Seidel sweep with over-relaxation) ──────────────
    # Each interior node is updated to the average of its four cardinal
    # neighbours (discretised Laplace equation ∇²X = 0, ∇²Y = 0), scaled
    # by the over-relaxation factor ω.  Because we write back to X/Y in
    # place, already-updated neighbors feed into later updates in the same
    # sweep — this is Gauss-Seidel order, which is what SOR requires.
    for _ in range(max_iter):
        max_diff = 0.0
        for row in range(1, N1 - 1):
            for col in range(1, N2 - 1):
                x_gs = np.mean(
                    [X[row + 1, col], X[row - 1, col], X[row, col + 1], X[row, col - 1]]
                )
                y_gs = np.mean(
                    [Y[row + 1, col], Y[row - 1, col], Y[row, col + 1], Y[row, col - 1]]
                )
                x_new = omega * x_gs + (1 - omega) * X[row, col]
                y_new = omega * y_gs + (1 - omega) * Y[row, col]
                diff = max(abs(x_new - X[row, col]), abs(y_new - Y[row, col]))
                if diff > max_diff:
                    max_diff = diff
                X[row, col] = x_new
                Y[row, col] = y_new
        if max_diff < tol:
            break

    return X, Y


# ── Polygon class ──────────────────────────────────────────────────────────────


class Polygon:
    """A single face of the ReLU hyperplane arrangement.

    Parameters
    ----------
    vertex_names : list[str]
        Ordered vertex labels (e.g. ['v0', 'v3', 'v7']).
    vertex_dict : dict[str, array-like]
        Mapping from label to 2-D coordinate.
    """

    def __init__(self, vertex_names, vertex_dict):
        self.vertex_names = vertex_names
        self.vertex_coords = [vertex_dict[v] for v in vertex_names]

    @property
    def centroid(self):
        xs = [c[0] for c in self.vertex_coords]
        ys = [c[1] for c in self.vertex_coords]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def rotate(self, gradient_vec):
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
    ):
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


def _check_poly_worker(args):
    node_list, vertex_dict, W_matrix, B_vector, W_out_vec, dynamics = args
    poly = Polygon(node_list, vertex_dict)
    return poly.check_gradient(W_matrix, B_vector, W_out_vec, dynamics)


def full_method(
    problem: LyapunovProblem,
) -> tuple[list[tuple[float, float, float]], list[str], dict[str, np.ndarray]]:
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
    polygons : list of list[str]   (ordered vertex-name lists)
    vertex_dict : dict[str, array-like]
    """
    x1_min, x1_max = problem.region[0, 0].item(), problem.region[0, 1].item()
    x2_min, x2_max = problem.region[1, 0].item(), problem.region[1, 1].item()

    W_matrix, B_vector, W_out_vec = extract_weights(problem.nn_lyapunov)

    print("Getting intersection points...")
    t0 = time.perf_counter()
    intersection_points = find_all_intersection_points(
        W_matrix,
        B_vector,
        x1_min,
        x1_max,
        x2_min,
        x2_max,
    )
    print(
        f"  done ({time.perf_counter() - t0:.3f}s)  — {len(intersection_points)} points"
    )

    print("Building planar graph...")
    t0 = time.perf_counter()
    vertex_dict, edge_list = get_planar_graph(
        intersection_points,
        W_matrix,
        B_vector,
        x1_min,
        x1_max,
        x2_min,
        x2_max,
    )
    G = nx.Graph()
    G.add_nodes_from(vertex_dict.keys())
    G.add_edges_from(edge_list)
    print(f"  done ({time.perf_counter() - t0:.3f}s)")

    print("Finding polygons...")
    t0 = time.perf_counter()
    g_ig = ig.Graph.from_networkx(G)
    polygon_node_lists = polygons_from_igraph_mcb(g_ig)
    print(
        f"  done ({time.perf_counter() - t0:.3f}s)  "
        f"— {len(polygon_node_lists)} polygons"
    )

    # Filter out near-zero-area polygons before verification.
    # These arise from residual near-duplicate vertices after dedup and from
    # boundary-edge slivers; their centroids land in degenerate regions that
    # produce false-positive counterexamples.
    domain_area = (x1_max - x1_min) * (x2_max - x2_min)
    area_tol = domain_area * 1e-10  # absolute area threshold
    filtered_node_lists = []
    n_zero_area = 0
    for node_list in polygon_node_lists:
        coords = [np.asarray(vertex_dict[v], dtype=float) for v in node_list]
        if len(coords) < 3:
            n_zero_area += 1
            continue
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        n = len(xs)
        area = (
            abs(
                sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
            )
            / 2.0
        )
        if area < area_tol:
            n_zero_area += 1
        else:
            filtered_node_lists.append(node_list)
    if n_zero_area:
        print(
            f"  Filtered {n_zero_area} near-zero-area polygons "
            f"({len(filtered_node_lists)} remain)"
        )

    print("Running verification...")
    t0 = time.perf_counter()
    dynamics_cpu = problem.dynamics.cpu()
    work_items = [
        (node_list, vertex_dict, W_matrix, B_vector, W_out_vec, dynamics_cpu)
        for node_list in filtered_node_lists
    ]
    with Pool() as pool:
        print(f"Using {pool._processes} parallel workers...")
        results = pool.map(_check_poly_worker, work_items)
    counterexamples = [cex for batch in results for cex in batch]
    print(
        f"  done ({time.perf_counter() - t0:.3f}s)  "
        f"— {len(counterexamples)} counterexamples"
    )

    return counterexamples, polygon_node_lists, vertex_dict
