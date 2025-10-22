import torch
import numpy as np
import math

# === 1) Define your neural-network controller model ===
# Replace this with your actual trained PyTorch model
# For example (dummy model):
class Controller(torch.nn.Module):
    def forward(self, x):
        # Suppose your network approximates u ≈ -20*(x)
        return -20 * x

model = Controller()
model.eval()


# === 2) Closed-loop dynamics ===
def f_closed_loop_scalar(x, r):
    """
    Closed-loop system: dot{x} = 1 - x^2 + u
    where u = NN(x - r)
    """
    x_t = torch.tensor([[x - r]], dtype=torch.float32)
    u = float(model(x_t)[0, 0])
    return 1 - x**2 + u


# === 3) Numerical equilibrium finder ===
def find_equilibrium_numeric(r, x0=None, maxit=50, tol=1e-10):
    """
    Find equilibrium x_star satisfying f(x_star) = 0
    """
    if x0 is None:
        x0 = r  # start near reference
    x = float(x0)
    for _ in range(maxit):
        fx = f_closed_loop_scalar(x, r)
        h = 1e-6
        fpx = (f_closed_loop_scalar(x + h, r) - f_closed_loop_scalar(x - h, r)) / (2 * h)
        if abs(fpx) < 1e-8:
            break
        dx = -fx / fpx
        x += dx
        if abs(dx) < tol:
            return x
    return x


# === 4) Lyapunov verification ===
def verify_local_lyapunov(r, neigh=1.0, domain_points=201, verbose=True):
    """
    Compute equilibrium x_star(r), test V(x)=(x-x_star)^2 as Lyapunov
    in a local neighborhood of size `neigh`.
    """
    # find equilibrium
    x_star = find_equilibrium_numeric(r)
    xs = np.linspace(x_star - neigh, x_star + neigh, domain_points)
    
    # compute V and dV/dt
    V = lambda x: (x - x_star)**2
    dV = lambda x: 2*(x - x_star) * f_closed_loop_scalar(x, r)
    Vs = np.array([V(x) for x in xs])
    dVs = np.array([dV(x) for x in xs])
    
    # check conditions (excluding equilibrium point itself)
    mask = np.abs(xs - x_star) > 1e-8
    is_pos_def = np.all(Vs[mask] > 0)
    is_neg_def = np.all(dVs[mask] < 0)
    
    if verbose:
        print(f"\n=== Reference r = {r:.4f} ===")
        print(f"Equilibrium x* ≈ {x_star:.6f}")
        print(f"Min dV: {dVs[mask].min():.6f}, Max dV: {dVs[mask].max():.6f}")
        if is_pos_def and is_neg_def:
            print("✅ V(x)=(x-x*)² is a valid local Lyapunov function!")
        else:
            print("⚠️ Lyapunov conditions not fully met.")
            if not is_pos_def: print("   - V not positive definite (check domain).")
            if not is_neg_def: print("   - dV not negative definite (try smaller neighborhood).")
    
    return x_star, xs, Vs, dVs


# === 5) Test it for a few reference values ===
for r in [0.0, 1.0, 2.0, 3.0]:
    verify_local_lyapunov(r)
