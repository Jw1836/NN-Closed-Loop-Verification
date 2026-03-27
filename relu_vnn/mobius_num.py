import numpy as np
import sympy as sp
import torch
import torch.nn as nn


class SingleHiddenReLUNet(nn.Module):
    """Single hidden-layer ReLU network with one output neuron."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def extract_weights(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract hidden-layer and output-layer weights from [Linear, ReLU, Linear].

    Returns
    -------
    W_matrix : ndarray, shape (input_dim, hidden_dim)
    B_vector : ndarray, shape (hidden_dim,)
    W_out_vec : ndarray, shape (hidden_dim,)
    """
    network = model.network
    layer0 = network[0]
    layer2 = network[2]
    if not isinstance(layer0, nn.Linear) or not isinstance(layer2, nn.Linear):
        raise TypeError("model.network must be [Linear, ReLU, Linear]")

    W_matrix = layer0.weight.detach().cpu().numpy().T
    B_vector = layer0.bias.detach().cpu().numpy()
    W_out_vec = layer2.weight.detach().cpu().numpy().flatten()
    return W_matrix, B_vector, W_out_vec


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
        expr = sum(float(W_matrix[j, i]) * variables[j] for j in range(W_matrix.shape[0]))
        expr += float(B_vector[i])
        hyperplanes.append(sp.expand(expr))
    return hyperplanes


def get_characteristic_poly(hyperplanes, variables):
    t = sp.symbols('t')
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
            if not any(sp.simplify(h_res / existing).is_number for existing in A_restricted):
                A_restricted.append(h_res)
                
    return get_characteristic_poly(A_minus_H, variables) - \
           get_characteristic_poly(A_restricted, remaining_vars)

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
    model = SingleHiddenReLUNet(input_dim=2, hidden_dim=4)
    print(num_regions_via_char_poly(model))
