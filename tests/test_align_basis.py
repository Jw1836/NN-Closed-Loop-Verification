"""Tests for align_basis: Householder rotation aligning gradient with e_1."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from relu_vnn.hyperplane import align_basis


def _make_hull(vertices: np.ndarray) -> ConvexHull:
    """Create a ConvexHull from an array of vertices."""
    return ConvexHull(vertices)


# ── 3-D ──────────────────────────────────────────────────────────────────────


class TestAlignBasis3D:
    """3-D: tetrahedron with various gradient directions."""

    @pytest.fixture()
    def tetrahedron(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        return _make_hull(verts)

    def test_gradient_along_e3(self, tetrahedron):
        """Gradient along e_3 — q1 should pick out the z component."""
        q1, _ = align_basis(tetrahedron, np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(q1 @ np.array([0, 0, 1]), 1.0, atol=1e-12)
        np.testing.assert_allclose(q1 @ np.array([1, 0, 0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(q1 @ np.array([0, 1, 0]), 0.0, atol=1e-12)

    def test_negative_e1(self, tetrahedron):
        """Gradient along -e_1 — should flip sign."""
        q1, _ = align_basis(tetrahedron, np.array([-3.0, 0.0, 0.0]))
        np.testing.assert_allclose(q1, [-1, 0, 0], atol=1e-12)

    def test_preserves_volume(self, tetrahedron):
        grad = np.array([1.0, 2.0, -3.0])
        _, rv = align_basis(tetrahedron, grad)
        orig_hull = tetrahedron
        rot_hull = ConvexHull(rv)
        np.testing.assert_allclose(rot_hull.volume, orig_hull.volume, atol=1e-12)


# ── 2-D ──────────────────────────────────────────────────────────────────────


class TestAlignBasis2D:
    """2-D: compare against known rotation matrix."""

    @pytest.fixture()
    def square(self):
        return _make_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))

    def test_gradient_along_e1(self, square):
        """Gradient already along e_1 — identity rotation."""
        q1, rv = align_basis(square, np.array([2.0, 0.0]))
        np.testing.assert_allclose(q1, [1.0, 0.0], atol=1e-12)
        verts = square.points[square.vertices]
        np.testing.assert_allclose(rv, verts, atol=1e-12)

    def test_gradient_along_e2(self, square):
        """Gradient along e_2 — 90-degree rotation."""
        q1, rv = align_basis(square, np.array([0.0, 1.0]))
        # q1 should map e_2 direction to positive first component
        np.testing.assert_allclose(q1 @ np.array([0.0, 1.0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(q1 @ np.array([1.0, 0.0]), 0.0, atol=1e-12)

    def test_diagonal_gradient(self, square):
        grad = np.array([1.0, 1.0])
        q1, rv = align_basis(square, grad)
        # q1 @ grad_hat should equal 1 (mapped to e_1)
        g_hat = grad / np.linalg.norm(grad)
        np.testing.assert_allclose(q1 @ g_hat, 1.0, atol=1e-12)

    def test_rotated_vertices_preserve_distances(self, square):
        """Householder is orthogonal, so pairwise distances are preserved."""
        grad = np.array([3.0, -4.0])
        _, rv = align_basis(square, grad)
        verts = square.points[square.vertices]
        orig_dists = np.linalg.norm(verts[:, None] - verts[None, :], axis=-1)
        rot_dists = np.linalg.norm(rv[:, None] - rv[None, :], axis=-1)
        np.testing.assert_allclose(rot_dists, orig_dists, atol=1e-12)

    def test_column_input(self, square):
        """Gradient passed as (2, 1) column vector should work."""
        q1, _ = align_basis(square, np.array([[0.0], [1.0]]))
        np.testing.assert_allclose(q1 @ np.array([0.0, 1.0]), 1.0, atol=1e-12)


# ── 5-D ──────────────────────────────────────────────────────────────────────


class TestAlignBasis5D:
    """5-D: verify properties that must hold in any dimension."""

    @pytest.fixture()
    def simplex_5d(self):
        """A 5-D simplex (6 vertices)."""
        verts = np.eye(5) * 2.0
        verts = np.vstack([verts, np.zeros(5)])
        return _make_hull(verts)

    def test_q1_maps_gradient_to_e1(self, simplex_5d):
        rng = np.random.default_rng(42)
        grad = rng.standard_normal(5)
        q1, _ = align_basis(simplex_5d, grad)
        g_hat = grad / np.linalg.norm(grad)
        np.testing.assert_allclose(q1 @ g_hat, 1.0, atol=1e-12)

    def test_q1_is_unit_vector(self, simplex_5d):
        rng = np.random.default_rng(99)
        grad = rng.standard_normal(5)
        q1, _ = align_basis(simplex_5d, grad)
        np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-12)

    def test_preserves_pairwise_distances(self, simplex_5d):
        rng = np.random.default_rng(7)
        grad = rng.standard_normal(5)
        _, rv = align_basis(simplex_5d, grad)
        verts = simplex_5d.points[simplex_5d.vertices]
        orig_dists = np.linalg.norm(verts[:, None] - verts[None, :], axis=-1)
        rot_dists = np.linalg.norm(rv[:, None] - rv[None, :], axis=-1)
        np.testing.assert_allclose(rot_dists, orig_dists, atol=1e-12)

    def test_q1_orthogonal_to_complement(self, simplex_5d):
        """q1 · v_perp = 0 for any v perpendicular to gradient."""
        rng = np.random.default_rng(123)
        grad = rng.standard_normal(5)
        q1, _ = align_basis(simplex_5d, grad)
        g_hat = grad / np.linalg.norm(grad)
        # Build a random vector, project out gradient component
        w = rng.standard_normal(5)
        w_perp = w - np.dot(w, g_hat) * g_hat
        # Householder maps g_hat→e1, so vectors ⊥ g_hat stay ⊥ e1, meaning q1·w_perp = 0
        np.testing.assert_allclose(q1 @ w_perp, 0.0, atol=1e-12)

    def test_gradient_near_e1(self, simplex_5d):
        """Gradient nearly aligned with e_1 — should hit the early-return branch."""
        grad = np.array([1.0, 1e-16, 0.0, 0.0, 0.0])
        q1, rv = align_basis(simplex_5d, grad)
        np.testing.assert_allclose(q1, [1, 0, 0, 0, 0], atol=1e-10)
