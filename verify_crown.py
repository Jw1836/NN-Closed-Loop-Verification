"""alpha-beta-CROWN verification of Neural Lyapunov Networks.
For comparison to hyperplane method."""

from lyapunov import LyapunovProblem
import torch
from numpy import isclose
from abcrown import (
    VerificationSpec,
    ABCrownSolver,
    ConfigBuilder,
    input_vars,
    output_vars,
)
from abcrown.api import SolveResult
from typing import Any


def check_greater_zero(
    problem: LyapunovProblem, config
) -> dict[int, tuple[SolveResult, SolveResult]]:
    """V(x) > 0 for all x in region, x != 0.

    The region must contain the origin,
    so to not check zero does an independent half-plane check along each state dimension."""
    result = {}
    # Symbolic variables
    x = input_vars(problem.state_dim)
    y = output_vars(1)  # Lyapunov outputs scalar
    for d in range(problem.state_dim):
        # This loop does some redundant checking, since we really just need
        # to not check the origin. But this logic is cleaner for now.
        # For each dimension, check both half planes
        d_min = problem.region[d, 0].item()
        d_max = problem.region[d, 1].item()
        neg_halfplane = (x[d] > d_min) & (x[d] < 0.0)
        pos_halfplane = (x[d] > 0.0) & (x[d] < d_max)
        # Set other dimension bounds to be the full respective region
        for i in range(problem.state_dim):
            if i != d:
                i_min = problem.region[i, 0].item()
                i_max = problem.region[i, 1].item()
                neg_halfplane = neg_halfplane & (x[i] > i_min) & (x[i] < i_max)
                pos_halfplane = pos_halfplane & (x[i] > i_min) & (x[i] < i_max)
        # Output is simple scalar test
        output_greater_zero = y[0] > 0.0
        neg_spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=neg_halfplane,
            output_constraint=output_greater_zero,
        )
        pos_spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=pos_halfplane,
            output_constraint=output_greater_zero,
        )
        # Solve the two specs
        neg_result = ABCrownSolver(neg_spec, problem.nn_lyapunov, config=config).solve()
        pos_result = ABCrownSolver(pos_spec, problem.nn_lyapunov, config=config).solve()
        result[d] = [neg_result, pos_result]

    # Complete for all dimensions
    return result


def verify_lyapunov_nn(
    problem: LyapunovProblem, device: torch.device = torch.device("cpu")
) -> dict[str, Any]:
    # Dict to store verification result
    verification_result = {}

    # V(0) = 0
    zero_input = torch.zeros(problem.state_dim, device=device)
    zero_output = problem.nn_lyapunov(zero_input)
    if isclose(zero_output.item(), 0.0):
        verification_result["origin"] = (True, f"Zero input ==> {zero_output}.")
    else:
        verification_result["origin"] = (False, f"Zero input ==> {zero_output}.")

    # Build config for abcrown
    config = (
        ConfigBuilder.from_defaults()
        .set(general__device=device)
        .set(general__show_adv_example=True)
    )

    # V(x) > 0 for x != 0, for all x in region
    result_gt_zero = check_greater_zero(problem, config)
    verification_result["positive"] = result_gt_zero

    # dot{V(x)} = DV(x)F(X) < 0, for all x in region

    return verification_result


if __name__ == "__main__":
    # Example usage
    from lyapunov import LyapunovProblem
    from torch import nn

    class TrivialNN(nn.Module):
        def forward(self, x):
            return (x**2).sum(dim=-1, keepdim=True)

    class TrvialDyanamics(nn.Module):
        def forward(self, x):
            return x

    # Define a simple Lyapunov problem
    nn_lyapunov = TrivialNN()
    dynamics = TrvialDyanamics()
    region = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])  # 2D region
    problem = LyapunovProblem(nn_lyapunov=nn_lyapunov, dynamics=dynamics, region=region)

    # Verify the Lyapunov function
    result = verify_lyapunov_nn(problem)
    print("==============================================")
    print(f"Origin result: {result['origin']}")
    print("==============================================")
    print(f"Positive result: {result['positive']}\n\n")
    print(f"Dim 0: {result['positive'][0]}\n\n")
    print(f"Dim 1: {result['positive'][1]}\n\n")
