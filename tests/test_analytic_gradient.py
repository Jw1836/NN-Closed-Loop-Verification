"""Sanity-check analytic_gradient against torch.autograd."""

import numpy as np
import pytest
import torch
from torch import nn

from relu_vnn.hyperplane import analytic_gradient


def _make_relu_net(W_in, b_in, W_out):
    """Build a [Linear, ReLU, Linear] net from numpy arrays.

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
        net[2].bias.zero_()
    return net


def _autograd_gradient(net, x_np):
    """Compute ∇V(x) via torch.autograd."""
    x = torch.tensor(x_np.flatten(), dtype=torch.float32, requires_grad=True)
    y = net(x)
    y.backward()
    assert x.grad is not None
    return x.grad.detach().numpy().reshape(-1, 1)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Weights for 2-D input, 3-unit hidden layer
W_IN = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])  # (3, 2)
B_IN = np.array([0.5, -0.5, 0.0])  # (3,)
W_OUT = np.array([[2.0, -1.0, 0.5]])  # (1, 3)

# Weights for 7-D input, 5-unit hidden layer (non-power-of-2 stress test)
rng = np.random.default_rng(42)
W_IN_7 = rng.standard_normal((5, 7)).astype(np.float32)  # (hidden=5, input=7)
B_IN_7 = rng.standard_normal(5).astype(np.float32)
W_OUT_7 = rng.standard_normal((1, 5)).astype(np.float32)


@pytest.fixture
def net():
    return _make_relu_net(W_IN, B_IN, W_OUT)


# Points chosen to avoid all activation hyperplanes for W_IN/B_IN above.
# Activation boundaries: x[0]=-0.5 (neuron 0), x[1]=0.5 (neuron 1),
# x[0]-x[1]=0 (neuron 2).  At a boundary _step (>=0) and PyTorch ReLU (>0)
# disagree, so we deliberately stay away from them.
@pytest.mark.parametrize(
    "x",
    [
        np.array([[1.0], [2.0]]),  # neurons 0,1 active; 2 inactive
        np.array([[-1.0], [-2.0]]),  # all neurons inactive
        np.array([[0.5], [-0.5]]),  # neurons 0,2 active; 1 inactive
        np.array([[2.0], [-3.0]]),  # neurons 0,2 active; 1 inactive
        np.array([[-0.3], [0.1]]),  # neuron 0 active; 1,2 inactive
    ],
)
def test_matches_autograd(net, x):
    """analytic_gradient must agree with torch.autograd to float32 precision."""
    # analytic_gradient takes W_matrix as (input_dim, hidden_dim) — transpose of W_in
    W_matrix = W_IN.T  # (2, 3)
    B_vector = B_IN  # (3,)
    W_out_vec = W_OUT.flatten()  # (3,)

    analytic = analytic_gradient(W_matrix, B_vector, W_out_vec, x)
    autograd = _autograd_gradient(net, x)

    np.testing.assert_allclose(
        analytic,
        autograd,
        rtol=1e-5,
        err_msg=f"Mismatch at x={x.flatten()}",
    )


def test_matches_autograd_7d():
    """analytic_gradient agrees with autograd for a 7-D input, 5-unit hidden net."""
    net_7 = _make_relu_net(W_IN_7, B_IN_7, W_OUT_7)
    W_matrix = W_IN_7.T  # (7, 5)

    # Use a fixed point unlikely to land on any activation hyperplane
    x = np.array([[0.3], [-0.7], [1.2], [-0.1], [0.9], [-0.5], [0.4]], dtype=np.float32)

    analytic = analytic_gradient(W_matrix, B_IN_7, W_OUT_7.flatten(), x)
    autograd = _autograd_gradient(net_7, x)

    np.testing.assert_allclose(analytic, autograd, rtol=1e-5)
