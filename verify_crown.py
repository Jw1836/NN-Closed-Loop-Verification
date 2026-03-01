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

def check_greater_zero(problem: LyapunovProblem, config):
    """V(x) > 0 for all x in region, x != 0.
    
    The region must contain the origin,
    so to not check zero does an independent half-plane check along each state dimension."""
    # Symbolic variables
    x = input_vars(problem.state_dim)
    y = output_vars(1) # Lyapunov outputs scalar
    for d in range(problem.state_dim):
        # For each dimension, check both half planes
        d_min = problem.region[d,0]
        d_max = problem.region[d,1]
        negative_halfplane = (x[d] > d_min) & (x[d] < 0)
        positive_halfplane = (x[d] > 0) & (x[d] < d_max)
        # Output is simple scalar test
        output_greater_zero = y[0] > 0.0
        negative_spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=negative_halfplane,
            output_constraint=output_greater_zero
        )
        positive_spec = VerificationSpec.build_spec(
            input_vars=x,
            output_vars=y,
            input_constraint=positive_halfplane,
            output_constraint=output_greater_zero
        )
        # Solve the spec

        



def verify_lyapunov_nn(problem: LyapunovProblem, device:str='cpu')->dict[str, tuple[bool,str]]:
    # Dict to store verification result
    verification_result = {}
    
    # V(0) = 0
    zero_input = torch.zeros(problem.state_dim)
    zero_output = LyapunovProblem.nn_lyapunov(zero_input)
    if isclose(zero_input, 0.0):
        verification_result["zero"] = (True, f"Zero input ==> {zero_output}.")
    else:        
        verification_result["zero"] = (False, f"Zero input ==> {zero_output}.")


    # V(x) > 0 for x != 0, for all x in region

        



    # dot{V(x)} = DV(x)F(X) < 0, for all x in region

    return verification_result