import numpy as np
import sympy as sp
import torch.nn as nn

from relu_vnn.hyperplane import extract_weights


def build_hidden_hyperplanes(
    W_matrix: np.ndarray, B_vector: np.ndarray, variables: list[sp.Symbol]
) -> list[sp.Expr]:
    """Build hyperplane equations w_i · x + b_i = 0 for hidden neurons."""
    if W_matrix.shape[0] != len(variables):
        raise ValueError("Number of variables must match input dimension")
    if W_matrix.shape[1] != B_vector.shape[0]:
        raise ValueError("Hidden dimension mismatch between W_matrix and B_vector")

    hyperplanes: list[sp.Expr] = []
    for i in range(W_matrix.shape[1]):
        expr = sum(
            float(W_matrix[j, i]) * variables[j] for j in range(W_matrix.shape[0])
        )
        expr += float(B_vector[i])
        hyperplanes.append(sp.expand(expr))
    return hyperplanes


def get_characteristic_poly(hyperplanes, variables):
    t = sp.symbols("t")
    n = len(variables)

    # Base Case: Empty arrangement has poly t^n
    if not hyperplanes:
        return t**n

    # Pick a hyperplane H for the recurrence
    H = hyperplanes[0]
    A_minus_H = hyperplanes[1:]

    # Solve H=0 for one variable to perform restriction
    coeffs = [H.coeff(v) for v in variables]
    pivot_var = next((v for v, c in zip(variables, coeffs) if c != 0), None)

    if pivot_var is None:
        return get_characteristic_poly(A_minus_H, variables)

    # Restriction: substitute pivot_var in all other equations
    sub_expr = sp.solve(H, pivot_var)[0]
    remaining_vars = [v for v in variables if v != pivot_var]
    A_restricted = []

    for h in A_minus_H:
        h_res = h.subs(pivot_var, sub_expr).expand()
        # Keep only unique, non-zero hyperplanes
        if h_res != 0 and not h_res.is_number:
            if not any(
                sp.simplify(h_res / existing).is_number for existing in A_restricted
            ):
                A_restricted.append(h_res)

    return get_characteristic_poly(A_minus_H, variables) - get_characteristic_poly(
        A_restricted, remaining_vars
    )


def num_regions_via_char_poly(model: nn.Module) -> int:
    """Return the number of regions from hidden-neuron hyperplanes via chi(-1)."""
    W_matrix, B_vector, _ = extract_weights(model)
    input_dim = int(W_matrix.shape[0])

    variables = [sp.Symbol(f"x{i + 1}") for i in range(input_dim)]
    hyperplanes = build_hidden_hyperplanes(W_matrix, B_vector, variables)

    poly = get_characteristic_poly(hyperplanes, variables)
    t = sp.symbols("t")
    chi_at_neg1 = sp.simplify(poly.subs(t, -1))
    regions = sp.simplify(((-1) ** input_dim) * chi_at_neg1)
    return int(regions)


if __name__ == "__main__":
    import argparse
    import torch

    from relu_vnn.__main__ import load_problem_module
    from relu_vnn.checkpoint import _load_model_state

    parser = argparse.ArgumentParser(
        description="Compute the Möbius number of regions for a trained ReLU Lyapunov network.",
        usage="python -m relu_vnn.mobius [-h] [--hidden-size HIDDEN_SIZE] problem_file [checkpoint]",
    )
    parser.add_argument(
        "problem_file", help="Path to a Python file defining make_problem()"
    )
    parser.add_argument("checkpoint", nargs="?", help="Path to a .pt checkpoint file")
    parser.add_argument("--hidden-size", type=int, default=None)
    args = parser.parse_args()

    kwargs = {}
    if args.hidden_size is not None:
        kwargs["hidden_size"] = args.hidden_size

    mod = load_problem_module(args.problem_file)
    problem = mod.make_problem(**kwargs)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=False)
        _load_model_state(problem.nn_lyapunov, ckpt["model_state"])

    n = num_regions_via_char_poly(problem.nn_lyapunov)
    print(n)
