"""Basic smoke tests for enumerate_cells_bfs."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull, HalfspaceIntersection

from relu_vnn.hyperplane import (
    build_bbox_halfspaces,
    build_cell_halfspaces,
    enumerate_cells_bfs,
    find_chebyshev_center,
)


# Simple fixture: 2 hidden neurons whose hyperplanes cross inside [-1, 1]^2.
# W_matrix shape: (input_dim=2, hidden_dim=2)
# Neuron 0: x1 = 0  (vertical line through origin)
# Neuron 1: x2 = 0  (horizontal line through origin)
# This should produce exactly 4 cells (quadrants).
W_MATRIX = np.array([[1.0, 0.0], [0.0, 1.0]])  # each column is a neuron's weights
B_VECTOR = np.array([0.0, 0.0])
REGION = np.array([[-1.0, 1.0], [-1.0, 1.0]])
BBOX_HS = build_bbox_halfspaces(REGION)


@pytest.fixture
def cells():
    return enumerate_cells_bfs(W_MATRIX, B_VECTOR, BBOX_HS, REGION)


def test_returns_list_of_halfspace_intersections(cells):
    assert isinstance(cells, list)
    assert len(cells) > 0
    assert all(isinstance(c, HalfspaceIntersection) for c in cells)


def test_four_quadrant_cells(cells):
    assert len(cells) == 4


def test_cells_have_vertices(cells):
    for cell in cells:
        verts = cell.intersections
        assert verts.shape[0] >= 3
        assert verts.shape[1] == 2


def test_cells_have_halfspaces(cells):
    for cell in cells:
        # halfspaces shape: (n_constraints, ndim+1) = (n_constraints, 3)
        assert cell.halfspaces.shape[1] == 3
        assert cell.halfspaces.shape[0] >= 3


def test_cells_tile_domain(cells):
    """Total area of cells should equal the bounding box area (4.0)."""
    total = sum(ConvexHull(cell.intersections).volume for cell in cells)
    assert total == pytest.approx(4.0, abs=1e-8)


def test_point_in_cell_via_halfspaces(cells):
    """A point known to be in a quadrant should satisfy that cell's halfspaces."""
    test_point = np.array([0.5, 0.5])  # first quadrant
    found = False
    for cell in cells:
        hs = cell.halfspaces
        if np.all(hs[:, :2] @ test_point + hs[:, -1] <= 1e-10):
            found = True
            break
    assert found


# ---------------------------------------------------------------------------
# 3-D tests
# ---------------------------------------------------------------------------

# 3 hidden neurons whose hyperplanes are the coordinate planes in [-1, 1]^3.
# This should produce 2^3 = 8 octant cells.
W_3D = np.eye(3)  # (input_dim=3, hidden_dim=3)
B_3D = np.zeros(3)
REGION_3D = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
BBOX_HS_3D = build_bbox_halfspaces(REGION_3D)


@pytest.fixture
def cells_3d():
    return enumerate_cells_bfs(W_3D, B_3D, BBOX_HS_3D, REGION_3D)


def test_3d_bbox_halfspaces_shape():
    assert BBOX_HS_3D.shape == (6, 4)


def test_3d_cell_halfspaces_shape():
    pattern = (True, False, True)
    hs = build_cell_halfspaces(W_3D, B_3D, BBOX_HS_3D, pattern)
    # 6 bbox + 3 neuron constraints, each row has 4 columns (3 coords + offset)
    assert hs.shape == (9, 4)


def test_3d_chebyshev_center():
    center = find_chebyshev_center(BBOX_HS_3D)
    assert center is not None
    assert center.shape == (3,)
    np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-6)


def test_3d_eight_octant_cells(cells_3d):
    assert len(cells_3d) == 8


def test_3d_cells_have_correct_dimension(cells_3d):
    for cell in cells_3d:
        verts = cell.intersections
        assert verts.shape[1] == 3
        assert verts.shape[0] >= 4  # state_dim + 1


def test_3d_cells_tile_domain(cells_3d):
    """Total volume of cells should equal the bounding box volume (8.0)."""
    total = sum(ConvexHull(cell.intersections).volume for cell in cells_3d)
    assert total == pytest.approx(8.0, abs=1e-6)


def test_3d_point_in_cell(cells_3d):
    """A point in a known octant should be inside exactly one cell."""
    test_point = np.array([0.5, 0.5, 0.5])
    found = 0
    for cell in cells_3d:
        hs = cell.halfspaces
        if np.all(hs[:, :3] @ test_point + hs[:, -1] <= 1e-10):
            found += 1
    assert found == 1
