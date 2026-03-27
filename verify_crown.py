"""alpha-beta-CROWN verification of Neural Lyapunov Networks.
For comparison to hyperplane method."""

from relu_vnn.lyapunov import LyapunovProblem
import torch
from torch import nn
from numpy import isclose
from abcrown import (
    VerificationSpec,
    ABCrownSolver,
    ConfigBuilder,
    input_vars,
    output_vars,
)
from abcrown.api import SolveResult
from abcrown.auto_LiRPA.jacobian import JacobianOP
from typing import Any, cast


def check_origin(problem: LyapunovProblem, device: torch.device) -> tuple[str, float]:
    zero_input = torch.zeros(problem.state_dim, device=device)
    zero_output = problem.nn_lyapunov(zero_input).item()
    if isclose(zero_output, 0.0):
        return ("safe", zero_output)
    else:
        return ("unsafe", zero_output)


def _build_slabs(
    region: torch.Tensor, hole: float, overlap: float = 1e-6
) -> list[torch.Tensor]:
    """Build 2*N slabs tiling the region minus a hole around origin.

    Returns list of (state_dim, 2) tensors with [min, max] per dimension.
    """
    slabs = []
    for d in range(region.shape[0]):
        d_min = region[d, 0].item()
        d_max = region[d, 1].item()
        h = (d_max - d_min) * hole

        # Positive slab: x_d in [h - overlap, d_max]
        pos_slab = region.clone()
        pos_slab[d, 0] = h - overlap
        slabs.append(pos_slab)

        # Negative slab: x_d in [d_min, -h + overlap]
        neg_slab = region.clone()
        neg_slab[d, 1] = -h + overlap
        slabs.append(neg_slab)

    return slabs


def _build_input_constraint(x, slab: torch.Tensor):
    """Build ab-CROWN input constraint from symbolic vars and a slab bounds tensor."""
    constraint = (x[0] > slab[0, 0].item()) & (x[0] < slab[0, 1].item())
    for d in range(1, slab.shape[0]):
        constraint = (
            constraint & (x[d] > slab[d, 0].item()) & (x[d] < slab[d, 1].item())
        )
    return constraint


def check_positive(
    problem: LyapunovProblem, config, hole: float
) -> dict[int, tuple[SolveResult, SolveResult]]:
    """V(x) > 0 for all x in region, x != 0.

    In practice, check outside of rectangular hole with tessellated rectangles.
    A good default for the hole is 0.1% of the region size.
    """
    config.set(solver__bound_prop_method="forward+backward")  # propagate both ways

    result = {}
    x = input_vars(problem.state_dim)
    y = output_vars(1)  # Lyapunov outputs scalar
    slabs = _build_slabs(problem.region, hole)

    for slab in slabs:
        input_constraint = _build_input_constraint(x, slab)
        output_constraint = y[0] > 0.0
        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        )
        result[slab] = ABCrownSolver(spec, problem.nn_lyapunov, config=config).solve()

    return result


class DecreaseNetwork(nn.Module):
    """This network must be negative for all points in region.

    See Verified-Intelligence/alpha-beta-CROWN computation_graph.py#L122-L139
    """

    def __init__(self, dynamics: nn.Module, nn_lyapunov: nn.Module) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.nn_lyapunov = nn_lyapunov

    def forward(self, x):
        x = x.clone().requires_grad_(True)
        V = self.nn_lyapunov(x)
        # auto_LiRPA function to compute Jacobian, maintaining computation graph
        dVdx = JacobianOP.apply(V, x).squeeze(1)  # (batch, state_dim)
        f_x = self.dynamics(x)
        return torch.sum(dVdx * f_x, dim=1, keepdim=True)


def check_decrease(
    problem: LyapunovProblem, config, hole: float
) -> dict[int, tuple[SolveResult, SolveResult]]:
    """dot{V}(x) = DV(x)f(X) < 0 for all x in region, x != 0.

    In practice, check outside of rectangular hole with tessellated rectangles.
    A good default for the hole is 0.1% of the region size.
    """
    config.set(model__with_jacobian=True)  # Needed for decrease condition
    config.set(solver__bound_prop_method="backward")  # no "forward" with JacobianOP

    result = {}
    x = input_vars(problem.state_dim)
    y = output_vars(1)  # Lyapunov outputs scalar
    slabs = _build_slabs(problem.region, hole)

    for slab in slabs:
        input_constraint = _build_input_constraint(x, slab)
        output_constraint = y[0] < 0.0
        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        )
        model = DecreaseNetwork(problem.dynamics, problem.nn_lyapunov)
        result[slab] = ABCrownSolver(spec, model, config=config).solve()

    return result


def verify_lyapunov_nn(
    problem: LyapunovProblem,
    device: torch.device,
    hole: float = 0.001,
) -> dict[str, Any]:

    problem.to(device)

    # Dict to store verification result
    verification_result = {}

    # V(0) = 0
    verification_result["origin"] = check_origin(problem, device)

    # Build config for abcrown
    config = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(attack__pgd_order="skip")  # Prevent early exit with false counterexample
        .set(
            general__enable_incomplete_verification=False
        )  # Force complete (BaB) verification
        .set(general__complete_verifier="bab")
        .set(bab__branching__method="sb")  # Split the input space since low dimension
        .set(bab__branching__input_split__enable=True)
        .set(bab__timeout=3600)  # 1 hour per quadrant
    )

    # V(x) > 0 for x != 0, for all x in region
    verification_result["positive"] = check_positive(problem, config, hole)

    # dot{V(x)} = DV(x)F(X) < 0, for all x in region
    verification_result["decrease"] = check_decrease(problem, config, hole)

    return verification_result


def _set_origin_shift(net: nn.Module):
    """Perform a forward pass to find the true zero point of the network."""
    shift = cast(torch.Tensor, net.shift)  # type: ignore[attr-defined]
    shift.zero_()
    state_dim = cast(nn.Linear, cast(nn.Sequential, net.network)[0]).in_features  # type: ignore[attr-defined]
    zero_input = torch.zeros(state_dim)
    with torch.no_grad():
        value = net(zero_input)
    print(f"Setting origin shift to {value.item()} based on forward pass.")
    shift.copy_(value.squeeze())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="alpha-beta-CROWN Lyapunov verification"
    )
    parser.add_argument("problem_file", help="Path to a problem .py file")
    parser.add_argument(
        "checkpoint", nargs="?", default=None, help="Path to a .pt checkpoint"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None, help="Hidden layer size"
    )
    args = parser.parse_args()

    problem_path = args.problem_file
    checkpoint_path = args.checkpoint

    from relu_vnn.__main__ import load_problem_module, _load_model_state

    mod = load_problem_module(problem_path)

    hidden_size = args.hidden_size
    if checkpoint_path is not None and hidden_size is None:
        ckpt_tmp = torch.load(checkpoint_path, weights_only=False)
        w = ckpt_tmp["model_state"].get("network.0.weight")
        if w is not None:
            hidden_size = w.shape[0]

    kwargs = {} if hidden_size is None else {"hidden_size": hidden_size}
    problem = mod.make_problem(**kwargs)

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for verification.")
        device = torch.device("cuda")
    else:
        print("Warning: CUDA not available, trying to use CPU.")
        device = torch.device("cpu")

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, weights_only=False)
        _load_model_state(problem.nn_lyapunov, ckpt["model_state"])
        print(f"Checkpoint loaded: {checkpoint_path}")

    # Some networks have a torch register_buffer to exactly set the origin to be zero.
    # By convention, they are named "shift". Must run after loading checkpoint so the
    # shift is computed from the actual trained weights, not random initialization.
    if hasattr(problem.nn_lyapunov, "shift"):
        _set_origin_shift(problem.nn_lyapunov)

    result = verify_lyapunov_nn(problem, device)

    print("==============================================")
    print(f"Origin result: {result['origin']}")
    print("==============================================")
    print(f"Positive result: {result['positive']}\n\n")
    print("==============================================")
    print(f"Decrease result: {result['decrease']}\n\n")
    print("==============================================")
    origin_valid = result["origin"][0] == "safe"
    positive_valid = all(sr.status == "safe" for sr in result["positive"].values())
    decrease_valid = all(sr.status == "safe" for sr in result["decrease"].values())
    print(f"Origin:   {'valid' if origin_valid else 'violation'}")
    print(f"Positive: {'valid' if positive_valid else 'violation'}")
    print(f"Decrease: {'valid' if decrease_valid else 'violation'}")
