#!/usr/bin/env python
# coding: utf-8

print("Starting")

import csv
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time

from pendulum import PendulumProblem
from lyapunov import train_lyapunov_2d, lyapunov_loss_function
from hyperplane import full_method

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_SIZE = 30
MAX_ITERATIONS = 5
NUM_EPOCHS = 800
LEARNING_RATE = 1e-3
GRID_PTS = 300
RETRAIN_LR = 4e-4
CEX_WEIGHT = 10.0
EPSILON = 1e-4  # radius: skip counterexamples within this distance of origin
CEX_WINDOW = 3  # retrain on current + this many prior iterations
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints_pendulum")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CEX_LOG = os.path.join(CHECKPOINT_DIR, "counterexample_log.csv")


def append_cex_log(iteration: int, cexs: list[tuple]):
    """Append counterexamples (x1, x2, V_dot) for this iteration to the CSV log."""
    write_header = not os.path.exists(CEX_LOG)
    with open(CEX_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["iteration", "x1", "x2", "dVx"])
        for p in cexs:
            writer.writerow([iteration, p[0], p[1], p[2]])


def save_checkpoint(tag, problem, **extra):
    path = os.path.join(CHECKPOINT_DIR, f"{tag}.pt")
    torch.save(
        {
            "model_state": problem.nn_lyapunov.state_dict(),
            "region": problem.region,
            **extra,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_checkpoint(tag):
    path = os.path.join(CHECKPOINT_DIR, f"{tag}.pt")
    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        print(f"Checkpoint loaded: {path}")
        return data
    return None


if __name__ == "__main__":
    device = torch.device("cpu")

    # ──────────────────────────────────────────────────────────────────────────────

    pend = PendulumProblem(
        hidden_size=HIDDEN_SIZE
    )  # region built from physics defaults
    print(pend)

    x1_min, x1_max = pend.region[0, 0].item(), pend.region[0, 1].item()
    x2_min, x2_max = pend.region[1, 0].item(), pend.region[1, 1].item()

    # ## Initial Training (or resume from checkpoint)

    ckpt = load_checkpoint("initial_train")
    if ckpt is not None:
        pend.nn_lyapunov.load_state_dict(ckpt["model_state"])
        print("Skipping initial training — loaded from checkpoint.")
    else:
        train_lyapunov_2d(
            pend, grid_pts=GRID_PTS, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
        )
        save_checkpoint("initial_train", pend)

    pend.to("cpu")

    counterexamples, polygons, vertex_dict = full_method(pend)
    print(f"\nResult: {len(counterexamples)} counterexample(s) found.")

    # Plot initial verification
    fig, ax = plt.subplots(figsize=(7, 6))
    x1_t = np.linspace(x1_min, x1_max, GRID_PTS)
    x2_t = np.linspace(x2_min, x2_max, GRID_PTS)
    x1g, x2g = np.meshgrid(x1_t, x2_t)
    plot_pts = torch.tensor(
        np.stack([x1g.ravel(), x2g.ravel()], axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        V = pend.nn_lyapunov(plot_pts).numpy().reshape(x1g.shape)
    ax.contourf(x1g, x2g, V, levels=20, cmap="viridis", alpha=0.5)
    ax.contour(x1g, x2g, V, levels=20, colors="white", linewidths=0.4, alpha=0.3)
    for poly_nodes in polygons:
        coords = [vertex_dict[v] for v in poly_nodes]
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
    ax.set_title("Hyperplane verification: polygon tessellation + counterexamples")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("theta_dot (rad/s)")
    plt.tight_layout()
    fig.savefig(os.path.join(CHECKPOINT_DIR, "initial_verification.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {CHECKPOINT_DIR}/initial_verification.png")

    # ## CEGIS Loop

    base_grid = torch.tensor(
        np.stack([x1g.ravel(), x2g.ravel()], axis=1), dtype=torch.float32
    )
    cex_history: list[list[tuple]] = []
    start_iteration = 0

    for resume_i in range(MAX_ITERATIONS - 1, -1, -1):
        ckpt = load_checkpoint(f"iter_{resume_i}")
        if ckpt is not None:
            pend.nn_lyapunov.load_state_dict(ckpt["model_state"])
            cex_history = ckpt.get("cex_history", [])
            start_iteration = resume_i + 1
            print(f"Resuming from iteration {start_iteration}")
            break

    for i in range(start_iteration, MAX_ITERATIONS):
        start = time.time()
        pend.to("cpu")
        counterexamples2, _, _ = full_method(pend)
        print(f"Iteration {i}: {len(counterexamples2)} counterexample(s)")

        counterexamples2 = [
            p for p in counterexamples2 if (p[0] ** 2 + p[1] ** 2) >= EPSILON**2
        ]
        print(f"  {len(counterexamples2)} remain after filtering near-origin points.")
        cex_history.append(counterexamples2)

        if len(counterexamples2) == 0:
            print("No counterexamples — done.")
            break

        append_cex_log(i, counterexamples2)

        # Sliding window: train on the last CEX_WINDOW iterations only.
        # Avoids gradient conflicts from stale counterexamples that the network
        # has already corrected.
        window: list[tuple] = []
        for h in cex_history[-CEX_WINDOW:]:
            window.extend(h)
        print(
            f"  {len(window)} counterexamples in training window (last {CEX_WINDOW} iters)."
        )

        # Strip dVx column — training expects 2D state inputs
        cex_xy = [(p[0], p[1]) for p in window]
        cex_tensor = torch.tensor(cex_xy, dtype=torch.float32)
        pend.to(device)
        grid_dev = base_grid.to(device)
        cex_dev = cex_tensor.to(device)

        optimizer = torch.optim.Adam(pend.nn_lyapunov.parameters(), lr=RETRAIN_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
        )
        for epoch in range(NUM_EPOCHS):
            pend.nn_lyapunov.train()
            optimizer.zero_grad()
            loss_grid = lyapunov_loss_function(
                grid_dev.clone(), pend.nn_lyapunov, pend.dynamics
            )
            loss_cex = lyapunov_loss_function(
                cex_dev.clone(), pend.nn_lyapunov, pend.dynamics
            )
            loss = loss_grid + CEX_WEIGHT * loss_cex
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (epoch + 1) % 100 == 0:
                print(
                    f"  retrain epoch [{epoch + 1}/{NUM_EPOCHS}]  "
                    f"grid={loss_grid.item():.4f}  cex={loss_cex.item():.4f}"
                )

        print(f"Iteration {i} completed in {time.time() - start:.2f}s\n")
        save_checkpoint(
            f"iter_{i}",
            pend,
            iteration=i,
            cex_history=cex_history,
        )

    # ## Final Verification Plot

    pend.to("cpu")
    final_cexs, final_polygons, final_vertex_dict = full_method(pend)
    print(f"\nFinal verification: {len(final_cexs)} counterexample(s) found.")

    fig, ax = plt.subplots(figsize=(7, 6))
    with torch.no_grad():
        V = pend.nn_lyapunov(plot_pts).numpy().reshape(x1g.shape)
    ax.contourf(x1g, x2g, V, levels=20, cmap="viridis", alpha=0.5)
    ax.contour(x1g, x2g, V, levels=20, colors="white", linewidths=0.4, alpha=0.3)
    for poly_nodes in final_polygons:
        coords = [final_vertex_dict[v] for v in poly_nodes]
        xs = [c[0] for c in coords] + [coords[0][0]]
        ys = [c[1] for c in coords] + [coords[0][1]]
        ax.plot(xs, ys, "k-", linewidth=0.6, alpha=0.5)
    if final_cexs:
        ax.scatter(
            [p[0] for p in final_cexs],
            [p[1] for p in final_cexs],
            c="red",
            s=50,
            zorder=5,
            label=f"Counterexamples ({len(final_cexs)})",
        )
        ax.legend()
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_title("Final verification: polygon tessellation + counterexamples")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("theta_dot (rad/s)")
    plt.tight_layout()
    fig.savefig(os.path.join(CHECKPOINT_DIR, "final_verification.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {CHECKPOINT_DIR}/final_verification.png")

    # ## Counterexample History Plot

    n_iters = len(cex_history)
    counts = [len(h) for h in cex_history]
    cmap = plt.colormaps["plasma"]
    colors = [cmap(i / max(n_iters - 1, 1)) for i in range(n_iters)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(range(n_iters), counts, color=colors)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Counterexamples found")
    ax.set_title("Counterexample count per iteration")
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
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("theta_dot (rad/s)")
    ax.set_title("Counterexample locations by iteration")
    ax.legend(fontsize=7, markerscale=1.5, loc="lower left")

    ax = axes[2]
    for i, cexs in enumerate(cex_history):
        if cexs:
            dists = [np.sqrt(p[0] ** 2 + p[1] ** 2) for p in cexs]
            ax.hist(dists, bins=20, alpha=0.4, color=colors[i], label=f"iter {i}")
    ax.set_xlabel("Distance from origin")
    ax.set_ylabel("Count")
    ax.set_title("Distance distribution of counterexamples")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(CHECKPOINT_DIR, "cex_history.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {CHECKPOINT_DIR}/cex_history.png")
