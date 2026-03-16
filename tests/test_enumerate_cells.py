"""Basic smoke tests for enumerate_cells_bfs."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from relu_vnn.hyperplane import build_bbox_halfspaces, enumerate_cells_bfs


# Simple fixture: 2 hidden neurons whose hyperplanes cross inside [-1, 1]^2.
# W_matrix shape: (input_dim=2, hidden_dim=2)
# Neuron 0: x1 = 0  (vertical line through origin)
# Neuron 1: x2 = 0  (horizontal line through origin)
# This should produce exactly 4 cells (quadrants).
W_MATRIX = np.array([[1.0, 0.0], [0.0, 1.0]])  # each column is a neuron's weights
B_VECTOR = np.array([0.0, 0.0])
BBOX = (-1.0, 1.0, -1.0, 1.0)
BBOX_HS = build_bbox_halfspaces(*BBOX)


@pytest.fixture
def cells():
    return enumerate_cells_bfs(W_MATRIX, B_VECTOR, BBOX_HS, *BBOX)


def test_returns_list_of_convex_hulls(cells):
    assert isinstance(cells, list)
    assert len(cells) > 0
    assert all(isinstance(c, ConvexHull) for c in cells)


def test_four_quadrant_cells(cells):
    assert len(cells) == 4


def test_cells_have_vertices(cells):
    for hull in cells:
        verts = hull.points[hull.vertices]
        assert verts.shape[0] >= 3
        assert verts.shape[1] == 2


def test_cells_have_equations(cells):
    for hull in cells:
        # equations shape: (n_facets, ndim+1) = (n_facets, 3)
        assert hull.equations.shape[1] == 3
        assert hull.equations.shape[0] >= 3


def test_cells_tile_domain(cells):
    """Total area of cells should equal the bounding box area (4.0)."""
    total = sum(hull.volume for hull in cells)  # .volume is area in 2D
    assert total == pytest.approx(4.0, abs=1e-8)


def test_point_in_cell_via_equations(cells):
    """A point known to be in a quadrant should satisfy that cell's equations."""
    test_point = np.array([0.5, 0.5])  # first quadrant
    found = False
    for hull in cells:
        eq = hull.equations
        if np.all(eq[:, :2] @ test_point + eq[:, -1] <= 1e-10):
            found = True
            break
    assert found
