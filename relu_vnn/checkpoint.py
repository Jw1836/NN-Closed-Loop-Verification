"""Checkpoint save/load helpers for the Lyapunov training loop."""

import os
import re

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
    checkpoint_dir: str, problem: LyapunovProblem
) -> tuple[int, list[list[tuple]]]:
    """Scan dir for highest iter_N.pt; load it and return (N, cex_history), or (-1, [])."""
    highest = -1
    if os.path.isdir(checkpoint_dir):
        for fname in os.listdir(checkpoint_dir):
            m = re.match(r"^iter_(\d+)\.pt$", fname)
            if m:
                highest = max(highest, int(m.group(1)))

    if highest == -1:
        return -1, []

    ckpt = load_checkpoint(checkpoint_dir, f"iter_{highest}")
    if ckpt is None:
        return -1, []
    _load_model_state(problem.nn_lyapunov, ckpt["model_state"])
    problem.update_shift()
    cex_history = ckpt.get("cex_history", [])
    print(f"Resuming from iter_{highest}.pt")
    return highest, cex_history
