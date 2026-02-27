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
from torch import nn, Tensor


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
    return [
        p
        for p in intersection_points
        if is_within_rect(p, x1_min, x1_max, x2_min, x2_max)
    ]


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


def analytic_gradient(model, input_num, hidden_num, x_state):
    """Compute the exact gradient of the ReLU network at x_state.

    Exploits the fact that activation patterns are constant within each
    linear region, so the gradient is a fixed vector there.

    Parameters
    ----------
    model : nn.Module with a .network Sequential containing [Linear, ReLU, Linear]
    input_num : int
    hidden_num : int
    x_state : array-like, shape (input_num, 1)
    """
    W_matrix = model.network[0].weight.detach().numpy().T  # (input_dim, hidden_dim)
    B_vector = model.network[0].bias.detach().numpy()  # (hidden_dim,)
    W_out_vec = model.network[2].weight.detach().numpy().flatten()  # (hidden_dim,)
    W_out_mat = np.diag(W_out_vec)

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
    model, input_dim, hidden_dim, x1_val, x2_val, dynamics: nn.Module
) -> bool:
    """Return True if the Lie derivative V_dot < 0 at (x1_val, x2_val)."""
    x = np.array([[x1_val], [x2_val]])
    grad = analytic_gradient(model, input_dim, hidden_dim, x)
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

    # Refine along the edge
    x1_diff = x1_max_e - x1_min_e
    x2_diff = x2_max_e - x2_min_e
    steps = max(int(max(x1_diff, x2_diff) * 10), 100)

    if abs(v2[0] - v1[0]) < 1e-10:  # vertical edge
        x1_on_edge = np.full(steps, v1[0])
        x2_on_edge = np.linspace(x2_min_e, x2_max_e, steps)
    else:
        t = np.linspace(0, 1, steps)
        x1_on_edge = v1[0] + t * (v2[0] - v1[0])
        x2_on_edge = v1[1] + t * (v2[1] - v1[1])

    f_on_edge = np.array(
        [f_function(np.array([[x1], [x2]])) for x1, x2 in zip(x1_on_edge, x2_on_edge)]
    )
    return len(np.where(np.diff(np.sign(f_on_edge)) != 0)[0]) > 0


def amsden_hirt_grid(polygon_vertices, N1, N2, max_iter=250, tol=1e-6):
    """Generate a boundary-fitted interior grid for a polygon (Amsden-Hirt / SOR).

    Note: mutates polygon_vertices in place (appends the closing vertex).
    Pass a copy if the caller needs the original list unchanged.
    """
    polygon_vertices.append(polygon_vertices[0])  # close the polygon

    omega = 4 / (2 + np.sqrt(4 - (np.cos(np.pi / N1) + np.cos(np.pi / N2)) ** 2))
    total_pts = 2 * N1 + 2 * N2 - 4

    # Compute per-edge arc lengths
    lengths = [
        np.linalg.norm(
            np.array(polygon_vertices[i + 1]) - np.array(polygon_vertices[i])
        )
        for i in range(len(polygon_vertices) - 1)
    ]
    perimeter = sum(lengths)

    # Distribute boundary points proportionally to edge length
    boundary_pts = []
    pts_left = total_pts
    for i in range(len(lengths) - 1):
        n_edge = int((lengths[i] / perimeter) * total_pts)
        if n_edge <= 0:
            continue
        p0 = np.array(polygon_vertices[i])
        p1 = np.array(polygon_vertices[i + 1])
        xs = np.linspace(p0[0], p1[0], n_edge, endpoint=False)
        ys = np.linspace(p0[1], p1[1], n_edge, endpoint=False)
        for x_new, y_new in zip(xs, ys):
            boundary_pts.append((float(x_new), float(y_new)))
        pts_left -= n_edge

    # Remaining points go on the last edge
    if pts_left > 0:
        p0 = np.array(polygon_vertices[-2])
        p1 = np.array(polygon_vertices[-1])
        xs = np.linspace(p0[0], p1[0], pts_left, endpoint=False)
        ys = np.linspace(p0[1], p1[1], pts_left, endpoint=False)
        for x_new, y_new in zip(xs, ys):
            boundary_pts.append((float(x_new), float(y_new)))

    # Assign boundary points to the structured grid boundary
    X = np.zeros((N1, N2), dtype=float)
    Y = np.zeros((N1, N2), dtype=float)
    k = 0
    for i in range(N1):  # bottom (j=0)
        X[i, 0], Y[i, 0] = boundary_pts[k]
        k += 1
    for j in range(1, N1 - 1):  # right (i=N1-1)
        X[N1 - 1, j], Y[N1 - 1, j] = boundary_pts[k]
        k += 1
    for i in range(N1 - 1, -1, -1):  # top (j=N2-1)
        X[i, N2 - 1], Y[i, N2 - 1] = boundary_pts[k]
        k += 1
    for j in range(N2 - 2, 0, -1):  # left (i=0)
        X[0, j], Y[0, j] = boundary_pts[k]
        k += 1

    # Initial interior guess: bilinear interpolation between left/right boundaries
    for j in range(1, N2 - 1):
        for i in range(1, N1 - 1):
            s = i / (N1 - 1)
            xL, yL = X[0, j], Y[0, j]
            xR, yR = X[N1 - 1, j], Y[N1 - 1, j]
            X[i, j] = (1 - s) * xL + s * xR
            Y[i, j] = (1 - s) * yL + s * yR

    # SOR relaxation
    prev_X = X.copy()
    prev_Y = Y.copy()
    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(1, N1 - 1):
            for j in range(1, N2 - 1):
                x_new = 0.25 * (
                    prev_X[i + 1, j] + X[i - 1, j] + prev_X[i, j + 1] + X[i, j - 1]
                )
                y_new = 0.25 * (
                    prev_Y[i + 1, j] + Y[i - 1, j] + prev_Y[i, j + 1] + Y[i, j - 1]
                )
                X[i, j] = omega * x_new + (1 - omega) * prev_X[i, j]
                Y[i, j] = omega * y_new + (1 - omega) * prev_Y[i, j]
                diff = max(abs(prev_X[i, j] - X[i, j]), abs(prev_Y[i, j] - Y[i, j]))
                if diff > max_diff:
                    max_diff = diff
        if max_diff < tol:
            break
        prev_X = X.copy()
        prev_Y = Y.copy()

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
        theta = -np.arctan2(gradient_vec[1][0], gradient_vec[0][0])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return theta, R

    def check_gradient(self, model, input_dim, hidden_dim, dynamics: nn.Module):
        """Find counterexample points inside this polygon.

        Computes the analytic gradient at the centroid, rotates the coordinate
        frame so the gradient aligns with E_1, counts how many distinct sign
        regions of the rotated dynamics exist in the polygon, then uses an
        Amsden-Hirt grid to find one representative point per region. Points
        where the rotated f1 component is positive (V_dot >= 0) are returned
        as counterexamples.

        Parameters
        ----------
        model : nn.Module
        input_dim, hidden_dim : int
        dynamics : nn.Module

        Returns
        -------
        counterexamples : list of (float, float) tuples
        """
        E_2 = np.array([[0.0], [1.0]])
        centroid_pt = np.array([[self.centroid[0]], [self.centroid[1]]])

        U = analytic_gradient(model, input_dim, hidden_dim, centroid_pt)
        theta, R = self.rotate(U)
        rotated_U = R @ U

        # Sanity check: rotated gradient should be close to E_1
        if np.dot(rotated_U.T, E_2) > 1e-3 and rotated_U[0][0] < 0:
            print("Warning: rotated gradient is not close to E_1")

        # Component callables from dynamics module
        def f1(x):
            return _dynamics_numpy(dynamics, x)[0]

        def f2(x):
            return _dynamics_numpy(dynamics, x)[1]

        # Rotated dynamics
        def rf1(x):
            return np.cos(theta) * f1(x) - np.sin(theta) * f2(x)

        def rf2(x):
            return np.sin(theta) * f1(x) + np.cos(theta) * f2(x)

        # Count the number of distinct sign regions inside this polygon
        # TODO: This is specific to Duffing dynamics; needs to be checked differently for generic dynamics
        origin_inside = is_point_in_polygon(0, 0, self.vertex_coords)
        num_regions = 4 if origin_inside else 1

        f1_flag = f2_flag = False
        n = len(self.vertex_coords)
        for j in range(n):
            v1 = self.vertex_coords[j]
            v2 = self.vertex_coords[(j + 1) % n]
            if not f1_flag and zero_level_set_crosses_edge(v1, v2, rf1):
                f1_flag = True
            if not f2_flag and zero_level_set_crosses_edge(v1, v2, rf2):
                f2_flag = True
            if f1_flag and f2_flag:
                break

        if not origin_inside:
            if f1_flag and f2_flag:
                num_regions = 3
            elif f1_flag != f2_flag:
                num_regions = 2

        # Search via Amsden-Hirt grid for one representative per sign region
        N1 = N2 = 15
        found_count = 0
        counter = 0
        sign_list = []
        representative_pts = []

        while found_count < num_regions:
            # Pass a copy so amsden_hirt_grid's append doesn't corrupt our list
            X_grid, Y_grid = amsden_hirt_grid(
                list(self.vertex_coords),
                N1 * (counter + 1),
                N2 * (counter + 1),
            )
            for j in range(1, X_grid.shape[0] - 1):
                for k in range(1, X_grid.shape[1] - 1):
                    x1_pt = X_grid[j, k]
                    x2_pt = Y_grid[j, k]
                    f1_val = rf1(np.array([[x1_pt], [x2_pt]]))
                    f2_val = rf2(np.array([[x1_pt], [x2_pt]]))
                    sign_pair = (np.sign(f1_val), np.sign(f2_val))
                    if sign_pair not in sign_list:
                        sign_list.append(sign_pair)
                        representative_pts.append((x1_pt, x2_pt))
                        found_count += 1
                    if found_count == num_regions:
                        break
                if found_count == num_regions:
                    break
            counter += 1

        # Counterexamples: regions where rotated f1 > 0  (V_dot >= 0)
        return [
            representative_pts[j] for j, signs in enumerate(sign_list) if signs[0] > 0
        ]


# ── Main pipeline ──────────────────────────────────────────────────────────────


def full_method(
    net: nn.Module,
    dynamics: nn.Module,
    region: Tensor,
) -> tuple[list, list, dict]:
    """End-to-end hyperplane verification pipeline.

    Parameters
    ----------
    net : nn.Module
        Single-hidden-layer ReLU network (network[0]=Linear, network[1]=ReLU,
        network[2]=Linear).
    dynamics : nn.Module
        System dynamics; forward(x) returns dx with shape (N, 2).
    region : Tensor, shape (2, 2)
        Domain bounds [[x1_min, x1_max], [x2_min, x2_max]].

    Returns
    -------
    counterexamples : list of (float, float)
    polygons : list of list[str]   (ordered vertex-name lists)
    vertex_dict : dict[str, array-like]
    """
    x1_min, x1_max = region[0, 0].item(), region[0, 1].item()
    x2_min, x2_max = region[1, 0].item(), region[1, 1].item()

    W_matrix = net.network[0].weight.detach().numpy().T  # (input_dim, hidden_dim)
    B_vector = net.network[0].bias.detach().numpy()
    input_dim, hidden_layer_size = W_matrix.shape

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

    print("Running verification...")
    t0 = time.perf_counter()
    counterexamples = []
    for node_list in polygon_node_lists:
        poly = Polygon(node_list, vertex_dict)
        cex = poly.check_gradient(
            net,
            input_dim,
            hidden_layer_size,
            dynamics,
        )
        counterexamples.extend(cex)
    print(
        f"  done ({time.perf_counter() - t0:.3f}s)  "
        f"— {len(counterexamples)} counterexamples"
    )

    return counterexamples, polygon_node_lists, vertex_dict
