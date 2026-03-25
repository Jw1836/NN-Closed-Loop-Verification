"""Verification loop for ReLU Lyapunov functions.

Usage:
    python -m relu_vnn train <problem_file.py> [options]
    python -m relu_vnn verify <problem_file.py> [options]

The problem file must define a `make_problem()` function that returns a
LyapunovProblem instance.

Example:
    python -m relu_vnn train problems/pendulum.py --hidden-size 30
    python -m relu_vnn train problems/duffing_oscillator.py --epochs 800
    python -m relu_vnn verify problems/bilinear_oscillator.py --checkpoint checkpoints_bilinear_oscillator/iter_3.pt
"""

import argparse
import csv
import importlib.util
import json
import logging
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import cast

from .lyapunov import LyapunovProblem
from .train import (
    train_initial,
    finetune,
    default_initial_grid_pts,
    default_finetune_grid_pts,
)


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


def _load_model_state(model: torch.nn.Module, state_dict: dict):
    """Load state_dict with strict=True, tolerating only a missing 'shift' key."""
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if "shift" in str(e):
            print("Model does not have shift attribute. Loading strict=False")
            model.load_state_dict(state_dict, strict=False)
        else:
            raise


def append_verify_log(log_path: str, iteration: int, verify_results: dict):
    """Append one JSONL entry per iteration to the verification log."""
    entry: dict = {"iteration": iteration, "timestamp": time.time()}
    for key in ("n_cells", "origin", "positive", "decrease"):
        val = verify_results.get(key)
        if isinstance(val, dict):
            # Exclude large counterexample lists from the summary log
            entry[key] = {k: v for k, v in val.items() if k != "counterexamples"}
            entry[key]["n_counterexamples"] = (
                len(val["counterexamples"]) if val["counterexamples"] is not None else 0
            )
        else:
            entry[key] = val
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


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
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    if not hasattr(mod, "make_problem"):
        print(
            f"Error: {problem_path} must define a make_problem() function "
            f"that returns a LyapunovProblem.",
            file=sys.stderr,
        )
        sys.exit(1)

    return mod


# ── Verify result helpers ────────────────────────────────────────────────────


def _unpack_verify(
    results: dict,
) -> tuple[list[tuple], list[tuple], list[list]]:
    """Extract counterexamples from a verify() result dict.

    Returns (origin_cexs, spatial_cexs, cell_coords).
    The check_* methods in LyapunovProblem are responsible for any hole filtering.
    """
    from scipy.spatial import HalfspaceIntersection

    origin_cexs: list[tuple] = []
    spatial_cexs: list[tuple] = []

    origin = results["origin"]
    if origin is not None and not origin["passed"]:
        for row in origin["counterexamples"] or []:
            origin_cexs.append(tuple(float(v) for v in row))

    for key in ("positive", "decrease"):
        check = results[key]
        if check is not None and not check["passed"]:
            for row in check["counterexamples"] or []:
                spatial_cexs.append(tuple(float(v) for v in row))

    cells = cast(list[HalfspaceIntersection], results.get("cells", []))
    cell_coords = [cell.intersections.tolist() for cell in cells]
    return origin_cexs, spatial_cexs, cell_coords


def _print_verify_summary(
    origin_cexs: list[tuple], spatial_cexs: list[tuple], label: str = "Result"
):
    parts = []
    if origin_cexs:
        parts.append(f"origin: FAILED (V(0)={origin_cexs[0][-1]:.6f})")
    if spatial_cexs:
        parts.append(f"{len(spatial_cexs)} spatial counterexample(s)")
    if not parts:
        print(f"\n{label}: PASSED — no counterexamples.")
    else:
        print(f"\n{label}: {', '.join(parts)}")


def _run_verify(
    problem: LyapunovProblem, label: str
) -> tuple[list[tuple], list[tuple], list[list], dict]:
    """Verify, print summary, and return (origin_cexs, spatial_cexs, cell_coords, raw_results)."""
    raw = problem.verify()
    origin_cexs, spatial_cexs, cell_coords = _unpack_verify(raw)
    _print_verify_summary(origin_cexs, spatial_cexs, label=label)
    return origin_cexs, spatial_cexs, cell_coords, raw


def _write_cex_csv(output_path: str, raw_results: dict, state_dim: int):
    """Write counterexamples from all checks to a CSV file."""
    rows = []
    for check_type in ("origin", "positive", "decrease"):
        check = raw_results.get(check_type)
        if check is not None and not check["passed"]:
            for row in check["counterexamples"] or []:
                rows.append((check_type, *[float(v) for v in row]))

    if not rows:
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check"] + [f"x{i + 1}" for i in range(state_dim)] + ["value"])
        writer.writerows(rows)
    print(f"Counterexamples saved: {output_path}")


def _problem_slug(problem: LyapunovProblem) -> str:
    """Convert the problem class name to snake_case for use in filenames."""
    name = type(problem).__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _find_resume_point(
    checkpoint_dir: str, max_iterations: int, problem: LyapunovProblem
) -> tuple[int, list[list[tuple]]]:
    """Scan checkpoints newest-first; return (start_iteration, cex_history)."""
    for i in range(max_iterations - 1, -1, -1):
        ckpt = load_checkpoint(checkpoint_dir, f"iter_{i}")
        if ckpt is not None:
            _load_model_state(problem.nn_lyapunov, ckpt["model_state"])
            problem.update_shift()
            cex_history = ckpt.get("cex_history", [])
            print(f"Resuming from iteration {i + 1}")
            return i + 1, cex_history
    return 0, []


# ── Plotting ──────────────────────────────────────────────────────────────────


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


# ── Subcommands ───────────────────────────────────────────────────────────────


def cmd_verify(args):
    mod = load_problem_module(args.problem_file)
    kwargs = {}
    if args.hidden_size is not None:
        kwargs["hidden_size"] = args.hidden_size
    problem = mod.make_problem(**kwargs)

    problem.early_exit = args.early_exit
    problem.max_workers = args.max_workers
    if args.hole is not None:
        problem.hole = args.hole
    else:
        spans = problem.region[:, 1] - problem.region[:, 0]
        problem.hole = float(spans.min().item()) * 0.001

    if args.checkpoint is not None:
        path = args.checkpoint
        if not os.path.isfile(path):
            print(f"Error: checkpoint not found: {path}", file=sys.stderr)
            sys.exit(1)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        _load_model_state(problem.nn_lyapunov, ckpt["model_state"])
        print(f"Checkpoint loaded: {path}")

    problem.to("cpu")  # normalise device; verify always runs on CPU
    problem.update_shift()

    n_params = sum(p.numel() for p in problem.nn_lyapunov.parameters())
    hidden_size = getattr(problem.nn_lyapunov, "hidden_size", "?")
    print(f"Problem:      {problem}")
    print(f"Lyapunov net: hidden_size={hidden_size}  params={n_params:,}")
    print(f"Device:       cpu")
    print(
        f"Config:       early_exit={args.early_exit}  max_workers={args.max_workers}  hole={problem.hole}"
    )
    print()

    origin_cexs, spatial_cexs, polygons, raw = _run_verify(problem, "Verification")

    slug = _problem_slug(problem)
    output_dir = args.output_dir

    _write_cex_csv(
        os.path.join(output_dir, f"{slug}_counterexamples.csv"),
        raw,
        problem.state_dim,
    )
    plot_grid_pts = args.grid_pts if args.grid_pts is not None else 200
    plot_verification(
        output_dir,
        f"{slug}_verification.png",
        problem,
        origin_cexs + spatial_cexs,
        polygons,
        plot_grid_pts,
        f"{type(problem).__name__}: polygon tessellation + counterexamples",
    )


def cmd_train(args):
    # ── Load problem ──────────────────────────────────────────────────────────
    mod = load_problem_module(args.problem_file)
    kwargs = {}
    if args.hidden_size is not None:
        kwargs["hidden_size"] = args.hidden_size
    problem = mod.make_problem(**kwargs)

    problem.early_exit = args.early_exit
    problem.max_workers = args.max_workers
    if args.hole is not None:
        problem.hole = args.hole
    else:
        spans = problem.region[:, 1] - problem.region[:, 0]
        problem.hole = float(spans.min().item()) * 0.001

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    if args.checkpoint_dir is None:
        name = os.path.splitext(os.path.basename(args.problem_file))[0]
        checkpoint_dir = f"checkpoints_{name}"
    else:
        checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    verify_log = os.path.join(checkpoint_dir, "verification_log.jsonl")

    device = torch.device(args.device)

    # ── Print config ──────────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in problem.nn_lyapunov.parameters())
    hidden_size = getattr(problem.nn_lyapunov, "hidden_size", "?")
    _initial_grid = (
        args.grid_pts
        if args.grid_pts is not None
        else default_initial_grid_pts(problem)
    )
    _finetune_grid = (
        args.finetune_grid_pts
        if args.finetune_grid_pts is not None
        else default_finetune_grid_pts(problem)
    )
    plot_grid_pts = args.grid_pts if args.grid_pts is not None else 200
    print(f"Problem:        {problem}")
    print(f"Lyapunov net:   hidden_size={hidden_size}  params={n_params:,}")
    print(f"Dynamics:       {problem.dynamics}")
    print(f"Device:         {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(
        f"Config:         epochs={args.epochs}  finetune_epochs={args.finetune_epochs}  lbfgs_max_iter={args.lbfgs_max_iter}  lr={args.lr}  alpha={args.alpha}  grid={_initial_grid}  finetune_grid={_finetune_grid}"
        f"  cex_oversample={args.cex_oversample}  max_iter={args.max_iterations}"
        f"  early_exit={args.early_exit}  max_workers={args.max_workers}  hole={problem.hole}"
    )
    print()

    # ── Check for mid-loop resume point first ────────────────────────────────
    # Peek at iter checkpoints before deciding whether to train, so we don't
    # re-train from scratch when only initial_train.pt is missing.
    start_iteration, cex_history = _find_resume_point(
        checkpoint_dir, args.max_iterations, problem
    )

    if start_iteration > 0:
        # Model already loaded from iter checkpoint — skip training and initial verify
        problem.to(device)
        problem.update_shift()
    else:
        # ── Initial training (or resume from initial_train.pt) ────────────────
        ckpt = load_checkpoint(checkpoint_dir, "initial_train")
        if ckpt is not None:
            _load_model_state(problem.nn_lyapunov, ckpt["model_state"])
            problem.to(device)
            problem.update_shift()
            print("Skipping initial training — loaded from checkpoint.")
        else:
            problem.to(device)
            train_initial(
                problem,
                grid_pts=args.grid_pts,  # None → heuristic inside train_initial
                epochs=args.epochs,
                lr=args.lr,
                alpha=args.alpha,
            )
            problem.update_shift()
            save_checkpoint(checkpoint_dir, "initial_train", problem)

        # ── Initial verification ──────────────────────────────────────────────
        problem.to("cpu")
        origin_cexs, spatial_cexs, polygons, raw_initial = _run_verify(
            problem, "Initial verification"
        )
        _write_cex_csv(
            os.path.join(checkpoint_dir, "initial_counterexamples.csv"),
            raw_initial,
            problem.state_dim,
        )
        plot_verification(
            checkpoint_dir,
            "initial_verification.png",
            problem,
            origin_cexs + spatial_cexs,
            polygons,
            plot_grid_pts,
            "Hyperplane verification: polygon tessellation + counterexamples",
        )

        if not origin_cexs and not spatial_cexs:
            return

    # ── Verification / retrain loop ───────────────────────────────────────────
    for i in range(start_iteration, args.max_iterations):
        start = time.time()
        problem.to("cpu")
        origin_cexs, spatial_cexs, _, raw = _run_verify(problem, f"Iteration {i}")
        _write_cex_csv(
            os.path.join(checkpoint_dir, f"iter_{i}_counterexamples.csv"),
            raw,
            problem.state_dim,
        )
        cexs = origin_cexs + spatial_cexs
        cex_history.append(cexs)

        if not cexs:
            print("No counterexamples — done.")
            break

        append_verify_log(verify_log, i, raw)

        finetune(
            problem,
            cexs,
            args.finetune_epochs,
            device,
            alpha=args.alpha,
            cex_oversample=args.cex_oversample,
            grid_pts=args.finetune_grid_pts,
            lbfgs_max_iter=args.lbfgs_max_iter,
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
    origin_cexs, spatial_cexs, final_polygons, raw_final = _run_verify(
        problem, "Final verification"
    )
    _write_cex_csv(
        os.path.join(checkpoint_dir, "final_counterexamples.csv"),
        raw_final,
        problem.state_dim,
    )
    plot_verification(
        checkpoint_dir,
        "final_verification.png",
        problem,
        origin_cexs + spatial_cexs,
        final_polygons,
        plot_grid_pts,
        "Final verification: polygon tessellation + counterexamples",
    )
    if not origin_cexs and not spatial_cexs:
        save_checkpoint(checkpoint_dir, "passed", problem)
    plot_cex_history(checkpoint_dir, cex_history, problem)


# ── Argument parsing ──────────────────────────────────────────────────────────


def _add_common_args(parser: argparse.ArgumentParser):
    """Args shared between verify and train."""
    parser.add_argument(
        "problem_file",
        help="Path to a Python file defining make_problem() -> LyapunovProblem",
    )
    parser.add_argument(
        "--early-exit",
        action="store_true",
        default=False,
        help="Stop verification after the first failing condition",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Max worker processes for per-cell Lie derivative checks",
    )
    parser.add_argument(
        "--hole",
        type=float,
        default=None,
        help="Exclusion radius around origin: positivity and decrease counterexamples within this ball are not required (default: 0.1%% of smallest region span)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Override hidden layer size in the Lyapunov net",
    )
    parser.add_argument(
        "--grid-pts",
        type=int,
        default=None,
        help="Grid points per dimension for plotting and (train) initial AdamW training (default: D * max_span * 40)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-cell certified lines during verification",
    )


def main():
    parser = argparse.ArgumentParser(
        description="ReLU Lyapunov function training and verification"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ── verify ────────────────────────────────────────────────────────────────
    verify_parser = subparsers.add_parser(
        "verify",
        help="Run only the verification pipeline on an existing trained model",
    )
    _add_common_args(verify_parser)
    verify_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .pt checkpoint file to load before verifying",
    )
    verify_parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files (default: current working directory)",
    )

    # ── train ─────────────────────────────────────────────────────────────────
    train_parser = subparsers.add_parser(
        "train",
        help="Train a Lyapunov network and run the verify/retrain loop",
    )
    _add_common_args(train_parser)
    train_parser.add_argument(
        "--device", default="cpu", help="torch device (cpu, cuda, mps)"
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoints (default: checkpoints_<problem_name>)",
    )
    train_parser.add_argument("--max-iterations", type=int, default=5)
    train_parser.add_argument(
        "--epochs", type=int, default=400, help="Epochs for initial AdamW training"
    )
    train_parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=5,
        help="L-BFGS restart count after each verification (default: 5; L-BFGS converges in 1, rest are safety)",
    )
    train_parser.add_argument(
        "--finetune-grid-pts",
        type=int,
        default=None,
        help="Grid points per dimension for the fresh grid in each finetune step (default: D * max_span * 4)",
    )
    train_parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=20,
        help="Max closure evaluations per L-BFGS step (default: 20; increase for harder problems)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate for initial training",
    )
    train_parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Flatness penalty scale: penalises V(x) < alpha*||x|| (default: 1.0)",
    )
    train_parser.add_argument(
        "--cex-oversample",
        type=int,
        default=3,
        help="How many copies of each counterexample point to include in the training grid",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("relu_vnn").setLevel(
        logging.DEBUG if args.verbose else logging.INFO
    )
    if args.subcommand == "verify":
        cmd_verify(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
