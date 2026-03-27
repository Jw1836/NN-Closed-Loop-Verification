"""Checkpoint save/load helpers for the Lyapunov training loop."""

import os

import torch

from .lyapunov import LyapunovProblem


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


def find_resume_point(
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
