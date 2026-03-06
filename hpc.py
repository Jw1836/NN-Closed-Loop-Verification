#!/usr/bin/env python
# coding: utf-8

# # HPC Experiments

# ## Setup

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from duffing_oscillator import DuffingProblem
from lyapunov import train_lyapunov_2d, lyapunov_loss_function
from hyperplane import full_method
import time

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def save_checkpoint(tag, problem, **extra):
    path = os.path.join(CHECKPOINT_DIR, f"{tag}.pt")
    data = {
        "model_state": problem.nn_lyapunov.state_dict(),
        "region": problem.region,
        **extra,
    }
    torch.save(data, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(tag):
    path = os.path.join(CHECKPOINT_DIR, f"{tag}.pt")
    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        print(f"Checkpoint loaded: {path}")
        return data
    return None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    raise RuntimeError("CUDA not available.")


# In[ ]:


# Problem
HIDDEN_SIZE = 150
REGION = torch.tensor([[-3.0, 3.0], [-3.0, 3.0]])

# Training
MAX_ITERATIONS = 10
NUM_EPOCHS = 400
LEARNING_RATE = 1e-3
GRID_PTS = 300
RETRAIN_LR = 3e-4
CEX_WEIGHT = 10.0
EPSILON = 1e-5


# In[ ]:


duff = DuffingProblem(region=REGION, hidden_size=HIDDEN_SIZE)
print(duff)


# ## Initial Training (or resume from checkpoint)

# In[ ]:


ckpt = load_checkpoint("initial_train")
if ckpt is not None:
    duff.nn_lyapunov.load_state_dict(ckpt["model_state"])
    print("Skipping initial training — loaded from checkpoint.")
else:
    train_lyapunov_2d(
        duff, grid_pts=GRID_PTS, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE
    )
    save_checkpoint("initial_train", duff)


# In[ ]:


# Plotting functions expect model on CPU
duff.to("cpu")


# In[ ]:


# Names easier to work with
x1_min, x1_max = duff.region[0, 0].item(), duff.region[0, 1].item()
x2_min, x2_max = duff.region[1, 0].item(), duff.region[1, 1].item()


# In[ ]:


counterexamples, polygons, vertex_dict = full_method(duff)
print(f"\nResult: {len(counterexamples)} counterexample(s) found.")


# Plot the counterexamples on polygons overlaid over the loss function.

# In[ ]:


fig, ax = plt.subplots(figsize=(7, 6))

x1_t = np.linspace(x1_min, x1_max, GRID_PTS)
x2_t = np.linspace(x2_min, x2_max, GRID_PTS)
x1g, x2g = np.meshgrid(x1_t, x2_t)
plot_pts = torch.tensor(
    np.stack([x1g.ravel(), x2g.ravel()], axis=1), dtype=torch.float32
)
with torch.no_grad():
    V = duff.nn_lyapunov(plot_pts).numpy().reshape(x1g.shape)
ax.contourf(x1g, x2g, V, levels=20, cmap="viridis", alpha=0.5)
ax.contour(x1g, x2g, V, levels=20, colors="white", linewidths=0.4, alpha=0.3)

# Polygon tessellation
for poly_nodes in polygons:
    coords = [vertex_dict[v] for v in poly_nodes]
    xs = [c[0] for c in coords] + [coords[0][0]]
    ys = [c[1] for c in coords] + [coords[0][1]]
    ax.plot(xs, ys, "k-", linewidth=0.6, alpha=0.5)

# Counterexamples
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
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.tight_layout()
plt.show()


# ## Iterate Training

# In[ ]:


base_grid = torch.tensor(np.stack([x1g.ravel(), x2g.ravel()], axis=1), dtype=torch.float32)
all_counterexamples: list[tuple] = []
cex_history: list[list[tuple]] = []  # per-iteration counterexamples for comparison
start_iteration = 0

# Resume from latest iteration checkpoint if available
for resume_i in range(MAX_ITERATIONS - 1, -1, -1):
    ckpt = load_checkpoint(f"iter_{resume_i}")
    if ckpt is not None:
        duff.nn_lyapunov.load_state_dict(ckpt["model_state"])
        all_counterexamples = ckpt.get("all_counterexamples", [])
        cex_history = ckpt.get("cex_history", [])
        start_iteration = resume_i + 1
        print(f"Resuming from iteration {start_iteration}")
        break

for i in range(start_iteration, MAX_ITERATIONS):
    start = time.time()
    duff.to("cpu")
    counterexamples2, _, _ = full_method(duff)
    print(f"Iteration {i}: {len(counterexamples2)} counterexample(s)")

    counterexamples2 = [
        p for p in counterexamples2 if not (abs(p[0]) < EPSILON and abs(p[1]) < EPSILON)
    ]
    print(f"  {len(counterexamples2)} remain after filtering near-origin points.")
    cex_history.append(counterexamples2)

    if len(counterexamples2) == 0:
        print("No counterexamples — done.")
        break

    seen = {(round(p[0], 6), round(p[1], 6)) for p in all_counterexamples}
    for p in counterexamples2:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            all_counterexamples.append(p)
            seen.add(key)
    print(f"  {len(all_counterexamples)} total accumulated counterexamples.")

    cex_tensor = torch.tensor(all_counterexamples, dtype=torch.float32)
    duff.to(device)
    grid_dev = base_grid.to(device)
    cex_dev = cex_tensor.to(device)

    optimizer = torch.optim.Adam(duff.nn_lyapunov.parameters(), lr=RETRAIN_LR)
    for epoch in range(NUM_EPOCHS):
        duff.nn_lyapunov.train()
        optimizer.zero_grad()
        loss_grid = lyapunov_loss_function(
            grid_dev.clone(), duff.nn_lyapunov, duff.dynamics
        )
        loss_cex = lyapunov_loss_function(
            cex_dev.clone(), duff.nn_lyapunov, duff.dynamics
        )
        loss = loss_grid + CEX_WEIGHT * loss_cex
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(
                f"  retrain epoch [{epoch + 1}/{NUM_EPOCHS}]  "
                f"grid={loss_grid.item():.4f}  cex={loss_cex.item():.4f}"
            )

    end = time.time()
    print(f"Iteration {i} completed in {end - start:.2f}s\n")

    save_checkpoint(
        f"iter_{i}",
        duff,
        iteration=i,
        all_counterexamples=all_counterexamples,
        cex_history=cex_history,
    )


# ## Counterexample History

# In[ ]:


n_iters = len(cex_history)
counts = [len(h) for h in cex_history]
cmap = plt.colormaps["plasma"]
colors = [cmap(i / max(n_iters - 1, 1)) for i in range(n_iters)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Left: counterexample count per iteration ---
ax = axes[0]
ax.bar(range(n_iters), counts, color=colors)
ax.set_xlabel("Iteration")
ax.set_ylabel("Counterexamples found")
ax.set_title("Counterexample count per iteration")
ax.set_xticks(range(n_iters))

# --- Middle: spatial scatter, coloured by iteration ---
ax = axes[1]
for i, cexs in enumerate(cex_history):
    if cexs:
        xs = [p[0] for p in cexs]
        ys = [p[1] for p in cexs]
        ax.scatter(
            xs, ys, s=15, alpha=0.6, color=colors[i], label=f"iter {i} ({len(cexs)})"
        )
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Counterexample locations by iteration")
ax.legend(fontsize=7, markerscale=1.5, loc="lower left")

# --- Right: histogram of distances from origin per iteration ---
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
plt.show()

