import numpy as np
import torch
import plotly.graph_objects as go

from hyperplane import extract_weights, analytic_gradient


def _verify_point_with_dynamics(
    W_matrix, B_vector, W_out_vec, x1_val, x2_val, dynamics_list
):
    x_state = np.array([[x1_val], [x2_val]])
    V_grad = analytic_gradient(W_matrix, B_vector, W_out_vec, x_state)

    f1 = dynamics_list[0](x_state)
    f2 = dynamics_list[1](x_state)

    dot_product = V_grad[0, 0] * f1 + V_grad[1, 0] * f2
    if dot_product >= 0:
        return False
    else:
        return True


def make_CEX_plot_comp(
    x1_min,
    x1_max,
    x2_min,
    x2_max,
    my_nn,
    input_dim,
    hidden_layer_size,
    dynamics_list,
    counterexamples,
    polygons,
    vertex_dict,
):
    fig = go.Figure()

    W_matrix, B_vector, W_out_vec = extract_weights(my_nn)

    x1_grid = np.linspace(x1_min, x1_max, 200)
    x2_grid = np.linspace(x2_min, x2_max, 200)
    X_grid, Y_grid = np.meshgrid(x1_grid, x2_grid)
    analytic_search_counterexamples = []

    for j in range(X_grid.shape[0]):
        for k in range(X_grid.shape[1]):
            x1_pt = X_grid[j, k]
            x2_pt = Y_grid[j, k]
            check = _verify_point_with_dynamics(
                W_matrix,
                B_vector,
                W_out_vec,
                x1_pt,
                x2_pt,
                dynamics_list,
            )
            if not check:
                analytic_search_counterexamples.append((x1_pt, x2_pt))

    x1_vals = np.linspace(x1_min - 1, x1_max + 1, 150)
    x2_vals = np.linspace(x2_min - 1, x2_max + 1, 150)
    X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)
    grid_points = np.stack([X1_grid.ravel(), X2_grid.ravel()], axis=1)
    with torch.no_grad():
        V_values = (
            my_nn(torch.tensor(grid_points, dtype=torch.float32))
            .numpy()
            .reshape(X1_grid.shape)
        )

    fig.add_trace(
        go.Contour(
            x=x1_vals,
            y=x2_vals,
            z=V_values,
            contours_coloring="lines",
            line_width=1,
            ncontours=20,
            colorscale="Viridis",
            opacity=0.4,
            showscale=False,
            name="NN Level Sets",
        )
    )

    if len(analytic_search_counterexamples) > 0:
        ce_x_analytic = [pt[0] for pt in analytic_search_counterexamples]
        ce_y_analytic = [pt[1] for pt in analytic_search_counterexamples]
        fig.add_trace(
            go.Scatter(
                x=ce_x_analytic,
                y=ce_y_analytic,
                mode="markers",
                marker=dict(color="blue", size=4),
                name="Grid Search Method",
            )
        )

    if len(counterexamples) > 0:
        ce_x = [pt[0] for pt in counterexamples]
        ce_y = [pt[1] for pt in counterexamples]
        fig.add_trace(
            go.Scatter(
                x=ce_x,
                y=ce_y,
                mode="markers",
                marker=dict(color="red", size=10),
                name="Hyperplane Method",
            )
        )

    for polygon_nodes in polygons:
        coords = [vertex_dict[v] for v in polygon_nodes]
        if len(coords) < 3:
            continue

        x_vals = [pt[0] for pt in coords] + [coords[0][0]]
        y_vals = [pt[1] for pt in coords] + [coords[0][1]]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(width=1.2, color="black"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Counterexample Comparison: Hyperplane Method vs Refined Grid Search",
        xaxis_title="x1",
        yaxis_title="x2",
        xaxis=dict(range=[x1_min - 1, x1_max + 1], constrain="domain"),
        yaxis=dict(range=[x2_min - 1, x2_max + 1], scaleanchor="x", scaleratio=1),
        width=800,
        height=800,
        showlegend=True,
        template="plotly_white",
    )

    fig.show()
