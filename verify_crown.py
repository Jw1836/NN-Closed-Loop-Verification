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
from typing import Any


def check_origin(problem: LyapunovProblem, device: torch.device) -> tuple[bool, float]:
    zero_input = torch.zeros(problem.state_dim, device=device)
    zero_output = problem.nn_lyapunov(zero_input).item()
    if isclose(zero_output, 0.0):
        return (True, zero_output)
    else:
        return (False, zero_output)


def check_positive(
    problem: LyapunovProblem, config, hole: float = 0.0001
) -> dict[int, tuple[SolveResult, SolveResult]]:
    """V(x) > 0 for all x in region, x != 0.

    In practice, check outside of rectangular hole with tesselated rectangles.
    A good default for the hole is 0.01% of the region size.
    """
    result = {}
    # Symbolic variables
    x = input_vars(problem.state_dim)
    y = output_vars(1)  # Lyapunov outputs scalar

    # Hardcode to the 2D case, four checks
    x1_min = problem.region[0, 0].item()
    x1_max = problem.region[0, 1].item()
    x2_min = problem.region[1, 0].item()
    x2_max = problem.region[1, 1].item()
    h1 = (x1_max - x1_min) * hole
    h2 = (x2_max - x2_min) * hole
    overlap = 1e-6  # Handle the <= vs < issue with the hole boundary
    # Think of this as going clockwise around origin, with distance h over axis.
    # Trailing edge overlaps the previous quadrant to prevent discontinuities.
    quads = []
    quads.append(torch.tensor([[-h1 - overlap, x1_max], [h2, x2_max]]))
    quads.append(torch.tensor([[h1, x1_max], [x2_min, h2 + overlap]]))
    quads.append(torch.tensor([[x1_min, h1 + overlap], [x2_min, -h2]]))
    quads.append(torch.tensor([[x1_min, -h1], [-h2 - overlap, x2_max]]))

    # Check each quadrant
    for q in quads:
        # Convert bounds to symbolic vars and constraints
        x1_constraint = (x[0] > q[0][0].item()) & (x[0] < q[0][1].item())
        x2_constraint = (x[1] > q[1][0].item()) & (x[1] < q[1][1].item())
        input_constraint = x1_constraint & x2_constraint
        output_constraint = y[0] > 0.0
        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        )
        # Solve the spec
        result[q] = ABCrownSolver(spec, problem.nn_lyapunov, config=config).solve()

    # Complete for all dimensions
    return result


class DecreaseNetwork(nn.Module):
    """This network must be negative for all points in region.

    https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/examples_abcrown/neural_lyapunov_dependency/computation_graph.py#L122-L139
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
    problem: LyapunovProblem, config, hole: float = 0.0001
) -> dict[int, tuple[SolveResult, SolveResult]]:
    """dot{V}(x) = DV(x)f(X) < 0 for all x in region, x != 0.

    In practice, check outside of rectangular hole with tesselated rectangles.
    A good default for the hole is 0.01% of the region size.
    """
    result = {}
    # Symbolic variables
    x = input_vars(problem.state_dim)
    y = output_vars(1)  # Lyapunov outputs scalar

    # Hardcode to the 2D case, four checks
    x1_min = problem.region[0, 0].item()
    x1_max = problem.region[0, 1].item()
    x2_min = problem.region[1, 0].item()
    x2_max = problem.region[1, 1].item()
    h1 = (x1_max - x1_min) * hole
    h2 = (x2_max - x2_min) * hole
    overlap = 1e-6  # Handle the <= vs < issue with the hole boundary
    # Think of this as going clockwise around origin, with distance h over axis.
    # Trailing edge overlaps the previous quadrant to prevent discontinuities.
    quads = []
    quads.append(torch.tensor([[-h1 - overlap, x1_max], [h2, x2_max]]))
    quads.append(torch.tensor([[h1, x1_max], [x2_min, h2 + overlap]]))
    quads.append(torch.tensor([[x1_min, h1 + overlap], [x2_min, -h2]]))
    quads.append(torch.tensor([[x1_min, -h1], [-h2 - overlap, x2_max]]))

    # Check each quadrant
    for q in quads:
        # Convert bounds to symbolic vars and constraints
        x1_constraint = (x[0] > q[0][0].item()) & (x[0] < q[0][1].item())
        x2_constraint = (x[1] > q[1][0].item()) & (x[1] < q[1][1].item())
        input_constraint = x1_constraint & x2_constraint
        output_constraint = y[0] < 0.0
        spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=input_constraint,
            output_constraint=output_constraint,
        )
        # Solve the spec
        model = DecreaseNetwork(problem.dynamics, problem.nn_lyapunov)
        result[q] = ABCrownSolver(spec, model, config=config).solve()

    # Complete for all dimensions
    return result


def verify_lyapunov_nn(
    problem: LyapunovProblem,
    device: torch.device = torch.device("cpu"),
    hole: float = 0.0001,
) -> dict[str, Any]:
    # Dict to store verification result
    verification_result = {}

    # V(0) = 0
    verification_result["origin"] = check_origin(problem, device)

    # Build config for abcrown
    config = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(general__show_adv_example=True)
        .set(model__with_jacobian=True)
    )

    # V(x) > 0 for x != 0, for all x in region
    verification_result["positive"] = check_positive(problem, config, hole)

    # dot{V(x)} = DV(x)F(X) < 0, for all x in region
    verification_result["decrease"] = check_decrease(problem, config, hole)

    return verification_result


def _test_negative_dynamics():
    """In-file test: V(x) = |x1| + |x2| is a valid Lyapunov for f(x) = -x.

    dV/dt = sign(x)·(-x) = -|x1| - |x2| < 0.
    Represented exactly as relu(x1)+relu(-x1)+relu(x2)+relu(-x2).
    """

    class ReLULyapunov(nn.Module):
        def __init__(self):
            super().__init__()
            l1 = nn.Linear(2, 4, bias=False)
            l1.weight = nn.Parameter(
                torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
                requires_grad=False,
            )
            l2 = nn.Linear(4, 1, bias=False)
            l2.weight = nn.Parameter(
                torch.tensor([[1.0, 1.0, 1.0, 1.0]]), requires_grad=False
            )
            self.network = nn.Sequential(l1, nn.ReLU(), l2)

        def forward(self, x):
            return self.network(x)

    class NegativeDynamics(nn.Module):
        def forward(self, x):
            return -x

    nn_lyapunov = ReLULyapunov()
    print(nn_lyapunov)
    dynamics = NegativeDynamics()
    region = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

    problem = LyapunovProblem(nn_lyapunov=nn_lyapunov, dynamics=dynamics, region=region)

    result = verify_lyapunov_nn(problem)
    print("==============================================")
    print(f"Origin result: {result['origin']}")
    print("==============================================")
    print(f"Positive result: {result['positive']}\n\n")
    print("==============================================")
    print(f"Decrease result: {result['decrease']}\n\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(
            "Usage: python verify_crown.py <problem.py> [checkpoint.pt]",
            file=sys.stderr,
        )
        sys.exit(1)

    problem_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) == 3 else None

    from relu_vnn.__main__ import load_problem_module, _load_model_state

    mod = load_problem_module(problem_path)
    problem = mod.make_problem()

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, weights_only=False)
        _load_model_state(problem.nn_lyapunov, ckpt["model_state"])
        print(f"Checkpoint loaded: {checkpoint_path}")

    result = verify_lyapunov_nn(problem)
    print("==============================================")
    print(f"Origin result: {result['origin']}")
    print("==============================================")
    print(f"Positive result: {result['positive']}\n\n")
    print("==============================================")
    print(f"Decrease result: {result['decrease']}\n\n")
