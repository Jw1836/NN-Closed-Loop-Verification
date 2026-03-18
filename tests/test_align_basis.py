"""Tests for align_basis: Householder rotation aligning gradient with e_1."""

import numpy as np
import pytest

from relu_vnn.hyperplane import align_basis


def _reference_q1(gradient: np.ndarray) -> np.ndarray:
    """Reference first row via numpy QR on the gradient as a single column."""
    g = np.asarray(gradient, dtype=float).flatten().reshape(-1, 1)
    Q, R = np.linalg.qr(g, mode="complete")
    # QR may map g to -||g||*e1; correct sign so it maps to +||g||*e1
    if R[0, 0] < 0:
        Q = -Q
    return Q[0, :]


# ── 3-D ──────────────────────────────────────────────────────────────────────


class TestAlignBasis3D:
    """3-D: various gradient directions."""

    def test_gradient_along_e3(self):
        """Gradient along e_3 — q1 should pick out the z component."""
        q1 = align_basis(np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(q1 @ np.array([0, 0, 1]), 1.0, atol=1e-12)
        np.testing.assert_allclose(q1 @ np.array([1, 0, 0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(q1 @ np.array([0, 1, 0]), 0.0, atol=1e-12)

    def test_negative_e1(self):
        """Gradient along -e_1 — should flip sign."""
        q1 = align_basis(np.array([-3.0, 0.0, 0.0]))
        np.testing.assert_allclose(q1, [-1, 0, 0], atol=1e-12)

    def test_matches_qr_reference(self):
        grad = np.array([1.0, 2.0, -3.0])
        q1 = align_basis(grad)
        ref_q1 = _reference_q1(grad)
        np.testing.assert_allclose(q1, ref_q1, atol=1e-12)


# ── 2-D ──────────────────────────────────────────────────────────────────────


class TestAlignBasis2D:
    """2-D: compare against known rotation matrix."""

    def test_gradient_along_e1(self):
        """Gradient already along e_1 — identity rotation."""
        q1 = align_basis(np.array([2.0, 0.0]))
        np.testing.assert_allclose(q1, [1.0, 0.0], atol=1e-12)

    def test_gradient_along_e2(self):
        """Gradient along e_2 — 90-degree rotation."""
        q1 = align_basis(np.array([0.0, 1.0]))
        np.testing.assert_allclose(q1 @ np.array([0.0, 1.0]), 1.0, atol=1e-12)
        np.testing.assert_allclose(q1 @ np.array([1.0, 0.0]), 0.0, atol=1e-12)

    def test_diagonal_gradient(self):
        grad = np.array([1.0, 1.0])
        q1 = align_basis(grad)
        g_hat = grad / np.linalg.norm(grad)
        np.testing.assert_allclose(q1 @ g_hat, 1.0, atol=1e-12)

    def test_matches_qr_reference(self):
        """q1 should match the first row from numpy QR."""
        grad = np.array([3.0, -4.0])
        q1 = align_basis(grad)
        ref_q1 = _reference_q1(grad)
        np.testing.assert_allclose(q1, ref_q1, atol=1e-12)

    def test_column_input(self):
        """Gradient passed as (2, 1) column vector should work."""
        q1 = align_basis(np.array([[0.0], [1.0]]))
        np.testing.assert_allclose(q1 @ np.array([0.0, 1.0]), 1.0, atol=1e-12)


# ── 5-D ──────────────────────────────────────────────────────────────────────


class TestAlignBasis5D:
    """5-D: verify properties that must hold in any dimension."""

    def test_q1_maps_gradient_to_e1(self):
        rng = np.random.default_rng(42)
        grad = rng.standard_normal(5)
        q1 = align_basis(grad)
        g_hat = grad / np.linalg.norm(grad)
        np.testing.assert_allclose(q1 @ g_hat, 1.0, atol=1e-12)

    def test_q1_is_unit_vector(self):
        rng = np.random.default_rng(99)
        grad = rng.standard_normal(5)
        q1 = align_basis(grad)
        np.testing.assert_allclose(np.linalg.norm(q1), 1.0, atol=1e-12)

    def test_matches_qr_reference(self):
        """q1 should match numpy QR for random gradients."""
        rng = np.random.default_rng(7)
        grad = rng.standard_normal(5)
        q1 = align_basis(grad)
        ref_q1 = _reference_q1(grad)
        np.testing.assert_allclose(q1, ref_q1, atol=1e-12)

    def test_q1_orthogonal_to_complement(self):
        """q1 · v_perp = 0 for any v perpendicular to gradient."""
        rng = np.random.default_rng(123)
        grad = rng.standard_normal(5)
        q1 = align_basis(grad)
        g_hat = grad / np.linalg.norm(grad)
        w = rng.standard_normal(5)
        w_perp = w - np.dot(w, g_hat) * g_hat
        np.testing.assert_allclose(q1 @ w_perp, 0.0, atol=1e-12)

    def test_gradient_near_e1(self):
        """Gradient nearly aligned with e_1 — should hit the early-return branch."""
        grad = np.array([1.0, 1e-16, 0.0, 0.0, 0.0])
        q1 = align_basis(grad)
        np.testing.assert_allclose(q1, [1, 0, 0, 0, 0], atol=1e-10)
