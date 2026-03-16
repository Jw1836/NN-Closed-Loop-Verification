"""Tests for Polygon.check_gradient on simple cells with known dynamics."""

import numpy as np
import torch
from torch import nn

from relu_vnn.hyperplane import Polygon, analytic_gradient


# ---------------------------------------------------------------------------
# Helpers: minimal ReLU net + dynamics
# ---------------------------------------------------------------------------


def _make_relu_net(W_in, b_in, W_out, b_out=0.0):
    """Build a [Linear, ReLU, Linear] Sequential from numpy arrays.

    W_in:  (hidden, input)
    b_in:  (hidden,)
    W_out: (1, hidden)
    """
    hidden, input_dim = W_in.shape
    net = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(W_in, dtype=torch.float32))
        net[0].bias.copy_(torch.tensor(b_in, dtype=torch.float32))
        net[2].weight.copy_(torch.tensor(W_out, dtype=torch.float32))
        net[2].bias.fill_(b_out)
    return net


class StableDynamics(nn.Module):
    """f(x) = -x.  Globally stable — V_dot = ∇V · (-x) < 0 for good V."""

    def forward(self, x):
        return -x


class UnstableDynamics(nn.Module):
    """f(x) = +x.  Unstable — V_dot = ∇V · x > 0 away from origin."""

    def forward(self, x):
        return x


# A simple net: V(x) = relu(x1) + relu(x2).
# Hidden layer: identity (2 neurons, each picks one coordinate).
# In the first quadrant (x1>0, x2>0): V = x1 + x2, ∇V = [1, 1].
W_IN = np.array([[1.0, 0.0], [0.0, 1.0]])  # (hidden=2, input=2)
B_IN = np.array([0.0, 0.0])
W_OUT = np.array([[1.0, 1.0]])  # (1, hidden=2)

# Weight arrays in the shape analytic_gradient expects:
W_MATRIX = W_IN.T  # (input=2, hidden=2)
B_VECTOR = B_IN  # (hidden=2,)
W_OUT_VEC = W_OUT.flatten()  # (hidden=2,)

# A polygon fully inside the first quadrant — both neurons active.
QUAD1_VERTS = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]


class TestCheckGradientStable:
    """With f(x)=-x and ∇V=[1,1], V_dot = [1,1]·(-x) = -(x1+x2) < 0 in Q1."""

    def test_no_counterexamples(self):
        poly = Polygon(QUAD1_VERTS)
        cexs = poly.check_gradient(W_MATRIX, B_VECTOR, W_OUT_VEC, StableDynamics())
        assert cexs == []

    def test_gradient_is_correct(self):
        """Sanity: analytic_gradient at (0.5, 0.5) should be [1, 1]."""
        grad = analytic_gradient(W_MATRIX, B_VECTOR, W_OUT_VEC, np.array([0.5, 0.5]))
        np.testing.assert_allclose(grad.flatten(), [1.0, 1.0])


class TestCheckGradientUnstable:
    """With f(x)=+x and ∇V=[1,1], V_dot = [1,1]·x = x1+x2 > 0 in Q1."""

    def test_finds_counterexample(self):
        poly = Polygon(QUAD1_VERTS)
        cexs = poly.check_gradient(W_MATRIX, B_VECTOR, W_OUT_VEC, UnstableDynamics())
        assert len(cexs) > 0

    def test_counterexample_has_positive_vdot(self):
        poly = Polygon(QUAD1_VERTS)
        cexs = poly.check_gradient(W_MATRIX, B_VECTOR, W_OUT_VEC, UnstableDynamics())
        for x1, x2, vdot in cexs:
            assert vdot > 0, f"Expected V_dot > 0 at ({x1}, {x2}), got {vdot}"

    def test_counterexample_is_inside_polygon(self):
        poly = Polygon(QUAD1_VERTS)
        cexs = poly.check_gradient(W_MATRIX, B_VECTOR, W_OUT_VEC, UnstableDynamics())
        for x1, x2, _ in cexs:
            assert 0.2 <= x1 <= 0.8 and 0.2 <= x2 <= 0.8


class TestCheckGradientZeroGrad:
    """When ∇V = 0, the polygon should always be flagged (V_dot = 0 everywhere)."""

    def test_zero_gradient_returns_counterexample(self):
        # All output weights zero → ∇V = 0 everywhere
        W_out_zero = np.array([0.0, 0.0])
        poly = Polygon(QUAD1_VERTS)
        cexs = poly.check_gradient(W_MATRIX, B_VECTOR, W_out_zero, StableDynamics())
        assert len(cexs) > 0
        # V_dot should be 0.0 (since ∇V = 0)
        assert cexs[0][2] == 0.0
