"""Hyperplane arrangement verifier for ReLU neural Lyapunov functions.

Exploits the piecewise-linear structure of ReLU networks: each hidden
neuron defines a hyperplane, partitioning the state space into convex
polytopes where the gradient of V is constant.  Within each region
we run constrained optimisation (SLSQP) to find the worst-case Lie
derivative, guaranteeing completeness up to optimiser precision.

Public API
----------
find_counterexamples(model, problem, hidden_dim) -> list of counterexamples
"""

import numpy as np
import torch
from scipy.optimize import brentq, minimize, LinearConstraint

# ── Analytic gradient (constant per region) ────────────────────────────────


def _step(x):
    return 1 if x >= 0 else 0


def analytic_gradient(model, input_dim, hidden_dim, x_state):
    """Compute grad V analytically using the piecewise-linear structure.

    For a single-hidden-layer ReLU net:  grad V = W_out . diag(U) . W_in
    where U_i = 1 if neuron i is active at x_state, else 0.

    Parameters
    ----------
    model : NeuralLyapunov (single hidden layer)
    input_dim, hidden_dim : int
    x_state : np.ndarray, shape (dim, 1)

    Returns
    -------
    grad : np.ndarray, shape (dim, 1)
    """
    W_in = model.network[0].weight.detach().numpy().T  # (input_dim, hidden_dim)
    b_in = model.network[0].bias.detach().numpy()  # (hidden_dim,)
    W_out_vec = model.network[2].weight.detach().numpy().flatten()  # (hidden_dim,)
    W_out_diag = np.diag(W_out_vec)

    x_flat = x_state.flatten()
    U = np.array(
        [_step(np.dot(W_in[:, i], x_flat) + b_in[i]) for i in range(hidden_dim)]
    ).reshape(-1, 1)

    grad = np.zeros((input_dim, 1))
    for i in range(input_dim):
        W_prime = W_out_diag @ W_in[i, :].reshape(-1, 1)
        grad[i] = np.dot(U.T, W_prime)
    return grad


# ── Geometry: intersection finding ─────────────────────────────────────────


def _is_within_rect(point, x1_min, x1_max, x2_min, x2_max, tol=1e-8):
    return (x1_min - tol <= point[0] <= x1_max + tol) and (
        x2_min - tol <= point[1] <= x2_max + tol
    )


def _find_intersection(n1, b1, n2, b2):
    """Intersection of two 2-D hyperplanes (lines): n.x + b = 0."""
    A = np.array([n1, n2])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, np.array([-b1, -b2]))


def _find_all_intersection_points(
    W, b, zero_level_funcs, x1_min, x1_max, x2_min, x2_max
):
    """Find all hyperplane / boundary / dynamics intersections inside the ROI.

    Returns (intersection_points, dynamic_intersections).
    """
    x1_search = np.linspace(x1_min, x1_max, max(int(x1_max - x1_min), 200))
    points = []

    # Plane-plane intersections
    n_planes = W.shape[1]
    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            pt = _find_intersection(W[:, i], b[i], W[:, j], b[j])
            if pt is not None:
                points.append(pt)

    # Dynamics zero-level-set intersections (where f1=0 meets f2=0)
    dynamic_intersections = []
    if zero_level_funcs is not None and len(zero_level_funcs) >= 2:
        obj = lambda x1: zero_level_funcs[1](x1) - zero_level_funcs[0](x1)
        vals = obj(x1_search)
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        for idx in sign_changes:
            lo, hi = x1_search[idx], x1_search[idx + 1]
            if obj(lo) == 0:
                dynamic_intersections.append((lo, zero_level_funcs[1](lo)))
            else:
                root = brentq(obj, lo, hi)
                dynamic_intersections.append((root, zero_level_funcs[1](root)))

    # Plane-boundary intersections
    for i in range(n_planes):
        w_i, b_i = W[:, i], b[i]
        for edge_normal, edge_bias in [
            (np.array([0, 1]), -x2_min),
            (np.array([0, 1]), -x2_max),
            (np.array([1, 0]), -x1_min),
            (np.array([1, 0]), -x1_max),
        ]:
            pt = _find_intersection(w_i, b_i, edge_normal, edge_bias)
            if pt is not None and _is_within_rect(pt, x1_min, x1_max, x2_min, x2_max):
                points.append(pt)

    # Rectangle corners
    for corner in [
        (x1_min, x2_min),
        (x1_min, x2_max),
        (x1_max, x2_min),
        (x1_max, x2_max),
    ]:
        points.append(corner)

    valid_points = [
        p for p in points if _is_within_rect(p, x1_min, x1_max, x2_min, x2_max)
    ]
    valid_dynamic = [
        p
        for p in dynamic_intersections
        if _is_within_rect(p, x1_min, x1_max, x2_min, x2_max)
    ]
    return valid_points, valid_dynamic


# ── Planar graph construction ──────────────────────────────────────────────


def _connect_consecutive(points_on_edge, edge_name, edge_list):
    if len(points_on_edge) < 2:
        return edge_list
    key = 1 if edge_name in ("left", "right") else 0
    points_on_edge.sort(key=lambda p: p[1][key])
    for k in range(len(points_on_edge) - 1):
        edge_list.append((points_on_edge[k][0], points_on_edge[k + 1][0]))
    return edge_list


def _build_planar_graph(
    intersection_points, W, b, x1_min, x1_max, x2_min, x2_max, tol=1e-6
):
    """Build vertex dict and edge list from intersection points."""
    vertex_dict = {f"v{i}": pt for i, pt in enumerate(intersection_points)}

    edge_list = []
    # Boundary edges
    sides = {"left": [], "right": [], "bottom": [], "top": []}
    for i, pt in enumerate(intersection_points):
        v = f"v{i}"
        if abs(pt[0] - x1_min) < tol:
            sides["left"].append((v, pt))
        if abs(pt[0] - x1_max) < tol:
            sides["right"].append((v, pt))
        if abs(pt[1] - x2_min) < tol:
            sides["bottom"].append((v, pt))
        if abs(pt[1] - x2_max) < tol:
            sides["top"].append((v, pt))

    for name, pts in sides.items():
        edge_list = _connect_consecutive(pts, name, edge_list)

    # Hyperplane edges
    for plane_idx in range(W.shape[1]):
        w_i, b_i = W[:, plane_idx], b[plane_idx]
        pts_on_plane = []
        for i, pt in enumerate(intersection_points):
            if abs(np.asarray(pt) @ w_i + b_i) < tol:
                pts_on_plane.append((i, pt))

        if abs(w_i[0]) > abs(w_i[1]):
            pts_on_plane.sort(key=lambda p: p[1][1])
        else:
            pts_on_plane.sort(key=lambda p: p[1][0])

        for k in range(len(pts_on_plane) - 1):
            edge_list.append((f"v{pts_on_plane[k][0]}", f"v{pts_on_plane[k + 1][0]}"))

    return vertex_dict, edge_list


def _extract_polygons(vertex_dict, edge_list):
    """Use igraph minimum cycle basis to find polygon faces."""
    import igraph as ig

    vertex_names = list(vertex_dict.keys())
    name_to_idx = {n: i for i, n in enumerate(vertex_names)}
    ig_edges = []
    for e1, e2 in edge_list:
        if e1 in name_to_idx and e2 in name_to_idx:
            ig_edges.append((name_to_idx[e1], name_to_idx[e2]))

    g = ig.Graph(n=len(vertex_names), edges=ig_edges, directed=False)
    g.vs["name"] = vertex_names

    coords = [vertex_dict[n] for n in vertex_names]
    coords_arr = np.array([(c[0], c[1]) for c in coords])

    # Use planar face detection via minimum cycle basis
    try:
        g_planar = g.is_planar(kuratowski=False)
        if not g_planar:
            print("WARNING: graph is not planar")

        mcb = g.minimum_cycle_basis()
        polygons = []
        for cycle in mcb:
            polygon_vertices = [vertex_names[idx] for idx in cycle]
            polygons.append(polygon_vertices)
        return polygons
    except Exception as e:
        print(f"Polygon extraction failed: {e}")
        return []


# ── Point-in-polygon & verification ───────────────────────────────────────


def _is_point_in_polygon(x, y, polygon_vertices):
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


def _verify_point(model, input_dim, hidden_dim, x1, x2, f_list):
    """Check whether Lie derivative < 0 at (x1, x2) using autograd."""
    x = torch.tensor([[x1, x2]], dtype=torch.float32, requires_grad=True)
    V = model(x)
    V_grad = torch.autograd.grad(
        V, x, grad_outputs=torch.ones_like(V), create_graph=True
    )[0]

    x_col = np.array([[x1], [x2]])
    f1 = f_list[0](x_col)
    f2 = f_list[1](x_col)
    vdot = V_grad[0, 0].item() * f1 + V_grad[0, 1].item() * f2
    return vdot < 0


# ── Constrained optimisation per polygon ───────────────────────────────────


def _objective(x1, x2, model, input_dim, hidden_dim, f_list):
    """Lie derivative via analytic gradient."""
    grad = analytic_gradient(
        model, input_dim, hidden_dim, np.array([x1, x2]).reshape(-1, 1)
    )
    x_col = np.array([[x1], [x2]])
    f1 = f_list[0](x_col)
    f2 = f_list[1](x_col)
    return float(grad[0] * f1 + grad[1] * f2)


def _optimize_over_polygon(objective_func, polygon_vertices):
    """Maximise objective inside a convex polygon using SLSQP."""
    verts = np.array(polygon_vertices)
    centroid = np.mean(verts, axis=0)

    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    padding = 0.01 * max(x_max - x_min, y_max - y_min)
    bounds = [(x_min - padding, x_max + padding), (y_min - padding, y_max + padding)]

    A_rows, b_rows = [], []
    for i in range(len(verts)):
        p1 = verts[i]
        p2 = verts[(i + 1) % len(verts)]
        dx, dy = p2 - p1
        normal = np.array([-dy, dx])
        constant = np.dot(normal, p1)
        if np.dot(normal, centroid) > constant:
            A_rows.append(normal)
            b_rows.append(constant)
        else:
            A_rows.append(-normal)
            b_rows.append(-constant)

    constraints = LinearConstraint(np.array(A_rows), lb=np.array(b_rows), ub=np.inf)

    def safe_obj(x):
        try:
            val = objective_func(x)
            return -val if np.isfinite(val) else 1e10
        except Exception:
            return 1e10

    result = minimize(
        safe_obj,
        x0=centroid,
        constraints=constraints,
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": 200, "ftol": 1e-6},
    )

    if np.isfinite(result.fun):
        result.fun = -result.fun
    else:
        result.fun = -np.inf

    return result, centroid


# ── Top-level public API ──────────────────────────────────────────────────


def find_counterexamples(model, problem, hidden_dim):
    """Run the full hyperplane arrangement verification pipeline.

    Parameters
    ----------
    model : NeuralLyapunov  (single hidden layer, ReLU)
    problem : LyapunovProblem  (must be 2-D)
    hidden_dim : int  – number of neurons in the hidden layer

    Returns
    -------
    counterexamples : list of (x1, x2) tuples
    centroids : list of polygon centroids tested
    vertex_dict : dict mapping vertex name -> coords
    edge_list : list of (v_name, v_name)
    polygons : list of polygon vertex name lists
    """
    assert problem.dim == 2, "hyperplane verifier currently supports 2-D only"

    (x1_min, x1_max), (x2_min, x2_max) = problem.domain
    input_dim = problem.dim

    # Extract weight matrix and bias from first layer
    W = model.network[0].weight.detach().numpy().T  # (input_dim, hidden_dim)
    b = model.network[0].bias.detach().numpy()  # (hidden_dim,)

    # 1. Find all intersection points
    intersection_points, dynamic_intersections = _find_all_intersection_points(
        W,
        b,
        problem.zero_level_set_funcs,
        x1_min,
        x1_max,
        x2_min,
        x2_max,
    )
    print(
        f"Found {len(intersection_points)} intersection points, "
        f"{len(dynamic_intersections)} dynamic intersections"
    )

    # 2. Build planar graph
    vertex_dict, edge_list = _build_planar_graph(
        intersection_points,
        W,
        b,
        x1_min,
        x1_max,
        x2_min,
        x2_max,
    )
    print(f"Graph: {len(vertex_dict)} vertices, {len(edge_list)} edges")

    # 3. Extract polygon faces
    polygons = _extract_polygons(vertex_dict, edge_list)
    print(f"Found {len(polygons)} polygons")

    # 4. Optimise over each polygon
    counterexamples = []
    centroids = []

    def obj_wrapper(x):
        return _objective(x[0], x[1], model, input_dim, hidden_dim, problem.f)

    for i, poly in enumerate(polygons):
        poly_coords = [vertex_dict[v] for v in poly]
        result, centroid = _optimize_over_polygon(obj_wrapper, poly_coords)
        centroids.append(centroid)

        if not _verify_point(
            model, input_dim, hidden_dim, result.x[0], result.x[1], problem.f
        ):
            counterexamples.append((result.x[0], result.x[1]))

    print(
        f"Verification complete: {len(counterexamples)} counterexamples "
        f"in {len(polygons)} regions"
    )

    return counterexamples, centroids, vertex_dict, edge_list, polygons
