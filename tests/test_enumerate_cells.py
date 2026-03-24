"""Tests for enumerate_cells_bfs: correctness of ReLU hyperplane cell enumeration."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull, HalfspaceIntersection

from relu_vnn.hyperplane import (
    EPS,
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


# ---------------------------------------------------------------------------
# Helper: n-dimensional test infrastructure
# ---------------------------------------------------------------------------


def _make_region(dim: int) -> np.ndarray:
    """Return [-1, 1]^dim region."""
    return np.tile([-1.0, 1.0], (dim, 1))


def _bbox_volume(region: np.ndarray) -> float:
    return float(np.prod(region[:, 1] - region[:, 0]))


def _cell_volumes(cells: list) -> list[float]:
    return [ConvexHull(c.intersections).volume for c in cells]


def _count_containing_cells(cells: list, point: np.ndarray) -> int:
    """Count how many cells contain a point (with EPS tolerance)."""
    count = 0
    for c in cells:
        hs = c.halfspaces
        if np.all(hs[:, :-1] @ point + hs[:, -1] <= EPS):
            count += 1
    return count


# ---------------------------------------------------------------------------
# A. Diagonal hyperplane (non-axis-aligned)
# ---------------------------------------------------------------------------


class TestDiagonalHyperplane:
    """Single neuron w = [1,...,1], b = 0 cuts the cube along the diagonal."""

    @pytest.fixture(params=[2, 3, 5])
    def setup(self, request):
        dim = request.param
        W = np.ones((dim, 1))
        B = np.array([0.0])
        region = _make_region(dim)
        bbox_hs = build_bbox_halfspaces(region)
        cells = enumerate_cells_bfs(W, B, bbox_hs, region)
        return dim, region, cells

    def test_two_cells(self, setup):
        _, _, cells = setup
        assert len(cells) == 2

    def test_tiles_domain(self, setup):
        _, region, cells = setup
        total = sum(_cell_volumes(cells))
        assert total == pytest.approx(_bbox_volume(region), abs=1e-6)

    def test_cells_have_enough_vertices(self, setup):
        dim, _, cells = setup
        for c in cells:
            hull = ConvexHull(c.intersections)
            assert len(hull.vertices) >= dim + 1


# ---------------------------------------------------------------------------
# B. Shifted hyperplane (non-zero bias, unequal volumes)
# ---------------------------------------------------------------------------


class TestShiftedHyperplane:
    """Single neuron along x1, shifted to x1 = 0.5."""

    @pytest.fixture(params=[2, 3, 5])
    def setup(self, request):
        dim = request.param
        W = np.zeros((dim, 1))
        W[0, 0] = 1.0
        B = np.array([-0.5])  # hyperplane at x1 = 0.5
        region = _make_region(dim)
        bbox_hs = build_bbox_halfspaces(region)
        cells = enumerate_cells_bfs(W, B, bbox_hs, region)
        return dim, region, cells

    def test_two_cells(self, setup):
        _, _, cells = setup
        assert len(cells) == 2

    def test_tiles_domain(self, setup):
        _, region, cells = setup
        total = sum(_cell_volumes(cells))
        assert total == pytest.approx(_bbox_volume(region), abs=1e-6)

    def test_unequal_volumes(self, setup):
        dim, _, cells = setup
        vols = sorted(_cell_volumes(cells))
        # x1 in [-1, 0.5] has width 1.5, x1 in [0.5, 1] has width 0.5
        # other dims have width 2 each → ratio is 3:1
        cross_section = 2.0 ** (dim - 1)
        assert vols[0] == pytest.approx(0.5 * cross_section, abs=1e-6)
        assert vols[1] == pytest.approx(1.5 * cross_section, abs=1e-6)


# ---------------------------------------------------------------------------
# C. Hyperplane outside domain
# ---------------------------------------------------------------------------


class TestOutsideDomain:
    """Single neuron whose hyperplane x1 = 5 lies entirely outside [-1,1]^n."""

    @pytest.fixture(params=[2, 3, 5])
    def setup(self, request):
        dim = request.param
        W = np.zeros((dim, 1))
        W[0, 0] = 1.0
        B = np.array([-5.0])
        region = _make_region(dim)
        bbox_hs = build_bbox_halfspaces(region)
        cells = enumerate_cells_bfs(W, B, bbox_hs, region)
        return dim, region, cells

    def test_one_cell(self, setup):
        _, _, cells = setup
        assert len(cells) == 1

    def test_full_volume(self, setup):
        _, region, cells = setup
        total = sum(_cell_volumes(cells))
        assert total == pytest.approx(_bbox_volume(region), abs=1e-6)


# ---------------------------------------------------------------------------
# D. Hyperplane on domain boundary
# ---------------------------------------------------------------------------


class TestBoundaryHyperplane:
    """Hyperplane x1 = 1.0, exactly on the edge of [-1,1]^n."""

    @pytest.fixture(params=[2, 3, 5])
    def setup(self, request):
        dim = request.param
        W = np.zeros((dim, 1))
        W[0, 0] = 1.0
        B = np.array([-1.0])
        region = _make_region(dim)
        bbox_hs = build_bbox_halfspaces(region)
        cells = enumerate_cells_bfs(W, B, bbox_hs, region)
        return dim, region, cells

    def test_tiles_domain(self, setup):
        _, region, cells = setup
        total = sum(_cell_volumes(cells))
        assert total == pytest.approx(_bbox_volume(region), abs=1e-6)

    def test_at_most_two_cells(self, setup):
        _, _, cells = setup
        assert 1 <= len(cells) <= 2


# ---------------------------------------------------------------------------
# E. Mixed inside/outside neurons
# ---------------------------------------------------------------------------


class TestMixedNeurons:
    """n neurons at origin + n neurons at x=5 (outside domain)."""

    @pytest.fixture(params=[2, 3, 5])
    def setup(self, request):
        dim = request.param
        # n "inside" neurons (coordinate hyperplanes) + n "outside" neurons
        n_inside = dim
        n_outside = dim
        W = np.zeros((dim, n_inside + n_outside))
        W[:, :n_inside] = np.eye(dim)  # coordinate hyperplanes at origin
        W[:, n_inside:] = np.eye(dim)  # same directions ...
        B = np.zeros(n_inside + n_outside)
        B[n_inside:] = -5.0  # ... but shifted to x_i = 5
        region = _make_region(dim)
        bbox_hs = build_bbox_halfspaces(region)
        cells = enumerate_cells_bfs(W, B, bbox_hs, region)
        return dim, region, cells

    def test_cell_count(self, setup):
        dim, _, cells = setup
        # Only the n inside hyperplanes create cells: 2^n cells
        assert len(cells) == 2**dim

    def test_tiles_domain(self, setup):
        _, region, cells = setup
        total = sum(_cell_volumes(cells))
        assert total == pytest.approx(_bbox_volume(region), abs=1e-6)


# ---------------------------------------------------------------------------
# F. Single neuron (simplest nontrivial case)
# ---------------------------------------------------------------------------


class TestSingleNeuron:
    """One neuron with x1 = 0 → exactly 2 cells."""

    @pytest.fixture(params=[2, 3, 5])
    def setup(self, request):
        dim = request.param
        W = np.zeros((dim, 1))
        W[0, 0] = 1.0
        B = np.array([0.0])
        region = _make_region(dim)
        bbox_hs = build_bbox_halfspaces(region)
        cells = enumerate_cells_bfs(W, B, bbox_hs, region)
        return dim, region, cells

    def test_two_cells(self, setup):
        _, _, cells = setup
        assert len(cells) == 2

    def test_tiles_domain(self, setup):
        _, region, cells = setup
        total = sum(_cell_volumes(cells))
        assert total == pytest.approx(_bbox_volume(region), abs=1e-6)


# ---------------------------------------------------------------------------
# G. Non-overlapping: random points each in exactly one cell
# ---------------------------------------------------------------------------


class TestNonOverlapping:
    """Random interior points must belong to exactly one cell."""

    @pytest.fixture(
        params=[
            (W_MATRIX, B_VECTOR, REGION),
            (W_3D, B_3D, REGION_3D),
            (np.eye(5), np.zeros(5), _make_region(5)),
            (
                np.array([[1.0, 1.0], [1.0, -1.0]]),
                np.zeros(2),
                REGION,
            ),
        ],
        ids=["2d-quadrants", "3d-octants", "5d-orthants", "2d-oblique"],
    )
    def setup(self, request):
        W, B, region = request.param
        bbox_hs = build_bbox_halfspaces(region)
        cells = enumerate_cells_bfs(W, B, bbox_hs, region)
        return region, cells

    def test_each_point_in_exactly_one_cell(self, setup):
        region, cells = setup
        dim = region.shape[0]
        n_points = 200 if dim >= 5 else 100
        rng = np.random.default_rng(42)
        # Sample interior points (shrink by 1% to avoid exact boundaries)
        lo = region[:, 0] * 0.99
        hi = region[:, 1] * 0.99
        for _ in range(n_points):
            pt = rng.uniform(lo, hi)
            count = _count_containing_cells(cells, pt)
            assert count == 1, f"Point {pt} in {count} cells (expected 1)"


# ---------------------------------------------------------------------------
# H. Oblique arrangement (non-axis-aligned hyperplanes)
# ---------------------------------------------------------------------------


class TestObliqueArrangement:
    """Two diagonal hyperplanes x+y=0 and x-y=0 create 4 diamond-shaped cells."""

    @pytest.fixture
    def cells(self):
        W = np.array([[1.0, 1.0], [1.0, -1.0]])
        B = np.zeros(2)
        bbox_hs = build_bbox_halfspaces(REGION)
        return enumerate_cells_bfs(W, B, bbox_hs, REGION)

    def test_four_cells(self, cells):
        assert len(cells) == 4

    def test_tiles_domain(self, cells):
        total = sum(ConvexHull(c.intersections).volume for c in cells)
        assert total == pytest.approx(4.0, abs=1e-8)
