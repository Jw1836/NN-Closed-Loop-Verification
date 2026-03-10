"""CEGIS loop for ReLU Lyapunov verification.

Usage:
    python -m verify_relu_lyapunov <problem_file.py> [options]

The problem file must define a `make_problem()` function that returns a
LyapunovProblem instance.

Example:
    python -m verify_relu_lyapunov problems/pendulum.py --hidden-size 30
    python -m verify_relu_lyapunov problems/duffing_oscillator.py --epochs 800
"""

import argparse
import csv
import importlib.util
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .lyapunov import LyapunovProblem, lyapunov_loss_function, train_lyapunov_2d
from .hyperplane import full_method


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(checkpoint_dir: str, tag: str, problem: LyapunovProblem, **extra):
    path = os.path.join(checkpoint_dir, f"{tag}.pt")
    torch.save(
        {
            "model_state": {
                k: v.cpu() for k, v in problem.nn_lyapunov.state_dict().items()
            },
            "region": problem.region,
            **extra,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_checkpoint(checkpoint_dir: str, tag: str):
    path = os.path.join(checkpoint_dir, f"{tag}.pt")
    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        print(f"Checkpoint loaded: {path}")
        return data
    return None


def append_cex_log(cex_log_path: str, iteration: int, cexs: list[tuple]):
    write_header = not os.path.exists(cex_log_path)
    with open(cex_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["iteration", "x1", "x2", "dVx"])
        for p in cexs:
            writer.writerow([iteration, p[0], p[1], p[2]])


# ── Problem loader ────────────────────────────────────────────────────────────


def load_problem_module(problem_path: str):
    """Dynamically import a problem file and call its make_problem()."""
    path = os.path.abspath(problem_path)
    if not os.path.isfile(path):
        print(f"Error: problem file not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Derive a stable module name so multiprocessing can pickle classes from it.
    # Also add the file's directory to sys.path so spawned workers can import it.
    mod_dir = os.path.dirname(path)
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    mod_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "make_problem"):
        print(
            f"Error: {problem_path} must define a make_problem() function "
            f"that returns a LyapunovProblem.",
            file=sys.stderr,
        )
        sys.exit(1)

    return mod


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_verification(
    checkpoint_dir: str,
    filename: str,
    problem: LyapunovProblem,
    counterexamples,
    polygons,
    vertex_dict,
    grid_pts: int,
    title: str,
):
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
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, filename), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {checkpoint_dir}/{filename}")


def plot_cex_history(checkpoint_dir: str, cex_history, problem: LyapunovProblem):
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
        ax.legend(fontsize=7, markerscale=1.5, loc="lower left")

    ax = axes[2]
    for i, cexs in enumerate(cex_history):
        if cexs:
            dists = [np.sqrt(p[0] ** 2 + p[1] ** 2) for p in cexs]
            ax.hist(dists, bins=20, alpha=0.4, color=colors[i], label=f"iter {i}")
    ax.set_xlabel("Distance from origin")
    ax.set_ylabel("Count")
    ax.set_title("Distance distribution of counterexamples")
    if any(cex_history):
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(checkpoint_dir, "cex_history.png"), dpi=150)
    plt.close(fig)
    print(f"Plot saved: {checkpoint_dir}/cex_history.png")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="CEGIS loop for ReLU Lyapunov verification"
    )
    parser.add_argument(
        "problem_file",
        help="Path to a Python file defining make_problem() -> LyapunovProblem",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoints (default: checkpoints_<problem_name>)",
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial training LR")
    parser.add_argument("--grid-pts", type=int, default=300)
    parser.add_argument("--retrain-lr", type=float, default=4e-4)
    parser.add_argument("--cex-weight", type=float, default=10.0)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
        help="Skip counterexamples within this distance of origin",
    )
    parser.add_argument(
        "--cex-window",
        type=int,
        default=3,
        help="Retrain on current + this many prior iterations",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Override hidden layer size in the Lyapunov net",
    )
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    args = parser.parse_args()

    # ── Load problem ──────────────────────────────────────────────────────────
    mod = load_problem_module(args.problem_file)
    kwargs = {}
    if args.hidden_size is not None:
        kwargs["hidden_size"] = args.hidden_size
    problem = mod.make_problem(**kwargs)

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    if args.checkpoint_dir is None:
        name = os.path.splitext(os.path.basename(args.problem_file))[0]
        checkpoint_dir = f"checkpoints_{name}"
    else:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    cex_log = os.path.join(checkpoint_dir, "counterexample_log.csv")

    device = torch.device(args.device)

    # ── Print config ──────────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in problem.nn_lyapunov.parameters())
    hidden_size = getattr(problem.nn_lyapunov, "hidden_size", "?")
    print(f"Problem:        {problem}")
    print(f"Lyapunov net:   hidden_size={hidden_size}  params={n_params:,}")
    print(f"Dynamics:       {problem.dynamics}")
    print(f"Device:         {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(
        f"Config:         epochs={args.epochs}  lr={args.lr}  grid={args.grid_pts}"
        f"  retrain_lr={args.retrain_lr}  cex_weight={args.cex_weight}"
        f"  epsilon={args.epsilon}  cex_window={args.cex_window}"
        f"  max_iter={args.max_iterations}"
    )
    print()

    # ── Initial training (or resume) ──────────────────────────────────────────
    ckpt = load_checkpoint(checkpoint_dir, "initial_train")
    if ckpt is not None:
        problem.nn_lyapunov.load_state_dict(ckpt["model_state"])
        print("Skipping initial training — loaded from checkpoint.")
    else:
        problem.to(device)
        train_lyapunov_2d(
            problem,
            grid_pts=args.grid_pts,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )
        save_checkpoint(checkpoint_dir, "initial_train", problem)

    # ── Initial verification ──────────────────────────────────────────────────
    problem.to("cpu")
    counterexamples_raw, polygons, vertex_dict = full_method(problem)
    counterexamples = [
        p for p in counterexamples_raw if (p[0] ** 2 + p[1] ** 2) >= args.epsilon**2
    ]
    n_origin = len(counterexamples_raw) - len(counterexamples)
    origin_note = f" ({n_origin} near-origin filtered)" if n_origin else ""
    print(f"\nResult: {len(counterexamples)} counterexample(s){origin_note}.")

    plot_verification(
        checkpoint_dir,
        "initial_verification.png",
        problem,
        counterexamples,
        polygons,
        vertex_dict,
        args.grid_pts,
        "Hyperplane verification: polygon tessellation + counterexamples",
    )

    # ── Build base grid for retraining ────────────────────────────────────────
    x1_min, x1_max = problem.region[0, 0].item(), problem.region[0, 1].item()
    x2_min, x2_max = problem.region[1, 0].item(), problem.region[1, 1].item()
    x1_t = np.linspace(x1_min, x1_max, args.grid_pts)
    x2_t = np.linspace(x2_min, x2_max, args.grid_pts)
    x1g, x2g = np.meshgrid(x1_t, x2_t)
    base_grid = torch.tensor(
        np.stack([x1g.ravel(), x2g.ravel()], axis=1), dtype=torch.float32
    )

    # ── CEGIS loop ────────────────────────────────────────────────────────────
    cex_history: list[list[tuple]] = []
    start_iteration = 0

    for resume_i in range(args.max_iterations - 1, -1, -1):
        ckpt = load_checkpoint(checkpoint_dir, f"iter_{resume_i}")
        if ckpt is not None:
            problem.nn_lyapunov.load_state_dict(ckpt["model_state"])
            cex_history = ckpt.get("cex_history", [])
            start_iteration = resume_i + 1
            print(f"Resuming from iteration {start_iteration}")
            break

    for i in range(start_iteration, args.max_iterations):
        start = time.time()
        problem.to("cpu")
        counterexamples2, _, _ = full_method(problem)
        print(f"Iteration {i}: {len(counterexamples2)} counterexample(s)")

        counterexamples2 = [
            p for p in counterexamples2 if (p[0] ** 2 + p[1] ** 2) >= args.epsilon**2
        ]
        print(f"  {len(counterexamples2)} remain after filtering near-origin points.")
        cex_history.append(counterexamples2)

        if len(counterexamples2) == 0:
            print("No counterexamples — done.")
            break

        append_cex_log(cex_log, i, counterexamples2)

        # Sliding window
        window: list[tuple] = []
        for h in cex_history[-args.cex_window :]:
            window.extend(h)
        print(
            f"  {len(window)} counterexamples in training window "
            f"(last {args.cex_window} iters)."
        )

        cex_xy = [(p[0], p[1]) for p in window]
        cex_tensor = torch.tensor(cex_xy, dtype=torch.float32)
        problem.to(device)
        grid_dev = base_grid.to(device)
        cex_dev = cex_tensor.to(device)

        optimizer = torch.optim.Adam(
            problem.nn_lyapunov.parameters(), lr=args.retrain_lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )
        for epoch in range(args.epochs):
            problem.nn_lyapunov.train()
            optimizer.zero_grad()
            loss_grid = lyapunov_loss_function(
                grid_dev.clone(), problem.nn_lyapunov, problem.dynamics
            )
            loss_cex = lyapunov_loss_function(
                cex_dev.clone(), problem.nn_lyapunov, problem.dynamics
            )
            loss = loss_grid + args.cex_weight * loss_cex
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (epoch + 1) % 100 == 0:
                print(
                    f"  retrain epoch [{epoch + 1}/{args.epochs}]  "
                    f"grid={loss_grid.item():.4f}  cex={loss_cex.item():.4f}"
                )

        print(f"Iteration {i} completed in {time.time() - start:.2f}s\n")
        save_checkpoint(
            checkpoint_dir,
            f"iter_{i}",
            problem,
            iteration=i,
            cex_history=cex_history,
        )

    # ── Final verification ────────────────────────────────────────────────────
    problem.to("cpu")
    final_cexs_raw, final_polygons, final_vertex_dict = full_method(problem)
    final_cexs = [
        p for p in final_cexs_raw if (p[0] ** 2 + p[1] ** 2) >= args.epsilon**2
    ]
    n_origin = len(final_cexs_raw) - len(final_cexs)
    origin_note = f" ({n_origin} near-origin filtered)" if n_origin else ""
    print(f"\nFinal verification: {len(final_cexs)} counterexample(s){origin_note}.")

    plot_verification(
        checkpoint_dir,
        "final_verification.png",
        problem,
        final_cexs,
        final_polygons,
        final_vertex_dict,
        args.grid_pts,
        "Final verification: polygon tessellation + counterexamples",
    )

    plot_cex_history(checkpoint_dir, cex_history, problem)


if __name__ == "__main__":
    main()
