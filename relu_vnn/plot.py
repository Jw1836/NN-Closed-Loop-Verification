"""Plotting utilities for Lyapunov verification results."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .lyapunov import LyapunovProblem


def plot_verification(
    checkpoint_dir: str,
    filename: str,
    problem: LyapunovProblem,
    counterexamples,
    polygons,
    grid_pts: int,
    title: str,
):
    if problem.state_dim != 2:
        print(
            f"Plotting skipped (only supported for 2D problems, got {problem.state_dim}D)"
        )
        return
    x1_min, x1_max = problem.region[0, 0].item(), problem.region[0, 1].item()
    x2_min, x2_max = problem.region[1, 0].item(), problem.region[1, 1].item()

    fig, ax = plt.subplots(figsize=(7, 6))
    x1_t = np.linspace(x1_min, x1_max, grid_pts)
    x2_t = np.linspace(x2_min, x2_max, grid_pts)
    x1g, x2g = np.meshgrid(x1_t, x2_t)
    plot_pts = torch.tensor(
        np.stack([x1g.ravel(), x2g.ravel()], axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        V = problem.nn_lyapunov(plot_pts).numpy().reshape(x1g.shape)
    ax.contourf(x1g, x2g, V, levels=20, cmap="viridis", alpha=0.5)
    ax.contour(x1g, x2g, V, levels=20, colors="white", linewidths=0.4, alpha=0.3)
    for coords in polygons:
        xs = [c[0] for c in coords] + [coords[0][0]]
        ys = [c[1] for c in coords] + [coords[0][1]]
        ax.plot(xs, ys, "k-", linewidth=0.6, alpha=0.5)
    if counterexamples:
        cx = [p[0] for p in counterexamples]
        cy = [p[1] for p in counterexamples]
        ax.scatter(
            cx,
            cy,
            c="red",
            s=50,
            zorder=5,
            label=f"Counterexamples ({len(counterexamples)})",
        )
        ax.legend()
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.tight_layout()
    save_path = os.path.join(checkpoint_dir, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {save_path}")


def plot_cex_history(checkpoint_dir: str, cex_history, problem: LyapunovProblem):
    if problem.state_dim != 2:
        return
    x1_min, x1_max = problem.region[0, 0].item(), problem.region[0, 1].item()
    x2_min, x2_max = problem.region[1, 0].item(), problem.region[1, 1].item()

    n_iters = len(cex_history)
    if n_iters == 0:
        return
    counts = [len(h) for h in cex_history]
    cmap = plt.colormaps["plasma"]
    colors = [cmap(i / max(n_iters - 1, 1)) for i in range(n_iters)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(range(n_iters), counts, color=colors)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Counterexamples found")
    ax.set_title("Counterexample count per iteration")
    if n_iters > 10:
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    else:
        ax.set_xticks(range(n_iters))

    ax = axes[1]
    for i, cexs in enumerate(cex_history):
        if cexs:
            ax.scatter(
                [p[0] for p in cexs],
                [p[1] for p in cexs],
                s=15,
                alpha=0.6,
                color=colors[i],
                label=f"iter {i} ({len(cexs)})",
            )
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Counterexample locations by iteration")
    if any(cex_history):
        ncol = max(1, n_iters // 10)
        ax.legend(fontsize=6, markerscale=1.5, loc="lower left", ncol=ncol)

    ax = axes[2]
    all_dists = []
    for cexs in cex_history:
        for p in cexs:
            all_dists.append(np.sqrt(p[0] ** 2 + p[1] ** 2))
    if all_dists:
        # Compute shared bin edges so overlapping histograms are comparable
        n_bins = min(20, max(1, len(set(np.round(all_dists, 6)))))
        bin_edges = np.linspace(min(all_dists), max(all_dists), n_bins + 1)
        for i, cexs in enumerate(cex_history):
            if cexs:
                dists = [np.sqrt(p[0] ** 2 + p[1] ** 2) for p in cexs]
                ax.hist(
                    dists, bins=bin_edges, alpha=0.4, color=colors[i], label=f"iter {i}"
                )
    ax.set_xlabel("Distance from origin")
    ax.set_ylabel("Count")
    ax.set_title("Distance distribution of counterexamples")
    if any(cex_history):
        ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, "cex_history.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {checkpoint_dir}/cex_history.png")
