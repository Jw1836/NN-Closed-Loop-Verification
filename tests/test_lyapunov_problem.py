"""Unit tests for LyapunovProblem.check_origin and check_positive."""

import numpy as np
import pytest
import torch
from torch import nn, Tensor
from scipy.spatial import ConvexHull, HalfspaceIntersection

from relu_vnn.lyapunov import LyapunovProblem


class SumSquares(nn.Module):
    """V(x) = sum(x**2). No learned parameters except a dummy for device tracking."""

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return (x**2).sum(dim=1, keepdim=True)


class ConstantNet(nn.Module):
    """V(x) = c (constant). Useful for forcing violations."""

    def __init__(self, c: float) -> None:
        super().__init__()
        self.c = c
        # Needs at least one parameter so LyapunovProblem.device works.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return torch.full((x.shape[0], 1), self.c, dtype=x.dtype, device=x.device)


def _make_problem(net: nn.Module) -> LyapunovProblem:
    region = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    dynamics = nn.Identity()  # unused by the two methods under test
    return LyapunovProblem(net, dynamics, region)


def _unit_square_hs() -> list[HalfspaceIntersection]:
    pts = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
    hull = ConvexHull(pts)
    interior = pts.mean(axis=0)
    return [HalfspaceIntersection(hull.equations, interior)]


# ---------------------------------------------------------------------------
# check_origin
# ---------------------------------------------------------------------------


class TestCheckOrigin:
    def test_passes_when_v0_is_zero(self):
        # SumSquares(0) = 0, so condition holds
        prob = _make_problem(SumSquares())
        assert prob.check_origin() is None

    def test_fails_when_v0_is_nonzero(self):
        # Constant net returning 1.0 — origin violation
        prob = _make_problem(ConstantNet(1.0))
        result = prob.check_origin()
        assert result is not None
        # Shape: (1, state_dim + 1) = (1, 3) for 2-D problem
        assert result.shape == (1, prob.state_dim + 1)
        # First state_dim entries are the origin
        assert np.allclose(result[0, :-1], 0.0)
        # Last entry is V(0)
        assert np.isclose(result[0, -1], 1.0)

    def test_negative_v0_is_also_a_violation(self):
        prob = _make_problem(ConstantNet(-0.5))
        result = prob.check_origin()
        assert result is not None
        assert np.isclose(result[0, -1], -0.5)


# ---------------------------------------------------------------------------
# check_positive
# ---------------------------------------------------------------------------


class TestCheckPositive:
    def test_passes_when_v_positive_everywhere(self):
        # SumSquares is strictly positive away from origin; hull vertices are
        # well outside the default hole radius so all should pass.
        prob = _make_problem(SumSquares())
        assert prob.check_positive(_unit_square_hs()) is None

    def test_detects_violation_when_v_nonpositive(self):
        # Constant 0 is not > 0, so all hull vertices should be returned.
        prob = _make_problem(ConstantNet(0.0))
        result = prob.check_positive(_unit_square_hs())
        assert result is not None
        assert result.ndim == 2
        assert result.shape[1] == prob.state_dim + 1
        # All returned V values should be ≤ 0
        assert np.all(result[:, -1] <= 0.0)

    def test_violation_rows_contain_correct_coordinates(self):
        prob = _make_problem(ConstantNet(-1.0))
        result = prob.check_positive(_unit_square_hs())
        assert result is not None
        # Verify each violating point actually evaluates to -1.0
        assert np.allclose(result[:, -1], -1.0)

    def test_skips_vertices_within_hole(self):
        # If all intersections are inside the hole radius, None is returned.
        pts = np.array([[1e-8, 0.0], [0.0, 1e-8], [-1e-8, 0.0], [0.0, -1e-8]])
        try:
            hull = ConvexHull(pts)
            interior = pts.mean(axis=0)
            hs = HalfspaceIntersection(hull.equations, interior)
        except Exception:
            pytest.skip("Degenerate hull — scipy version dependent")

        prob = _make_problem(ConstantNet(-1.0))
        prob.hole = 1.0  # large hole swallows all vertices
        result = prob.check_positive([hs])
        assert result is None
