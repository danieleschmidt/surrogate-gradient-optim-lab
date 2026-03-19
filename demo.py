#!/usr/bin/env python3
"""
Surrogate Gradient Optimization — Demo
=======================================
Optimises a black-box Rosenbrock function (2-D) with:
  1. Neural-network surrogate + autograd gradient descent
  2. Random-forest surrogate + finite-difference gradient descent
  3. Pure random search (baseline)

No gradients are passed to the optimisers — only f(x) values.

Run:
  python demo.py
"""
import sys
import numpy as np

from surrogate_gradient import (
    NeuralSurrogateOptimizer,
    RFSurrogateOptimizer,
    RandomSearchOptimizer,
    rosenbrock,
    sphere,
    ackley,
)


# ── Configuration ──────────────────────────────────────────────────────────────
DIM = 4
BOUNDS = [(-3.0, 3.0)] * DIM
N_INIT = 15        # initial LHS evaluations
N_ITER = 35        # surrogate-guided iterations
N_RS = N_INIT + N_ITER   # same budget for random search
SEED = 42
FUNCTION_NAME = "Sphere (4-D)"
FUNCTION = sphere


# ── Run optimisers ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Surrogate Gradient Optimization Demo")
print(f"  Function : {FUNCTION_NAME}  (dim={DIM})")
print(f"  Bounds   : {BOUNDS[0]}  (each axis)")
print(f"  Budget   : {N_INIT} init + {N_ITER} guided = {N_INIT+N_ITER} evals")
print(f"{'='*60}\n")

print("[1/3] Running NeuralSurrogate ...")
neural_opt = NeuralSurrogateOptimizer(
    hidden=(64, 32),
    n_init=N_INIT,
    n_epochs=200,
    lr_surrogate=1e-3,
    lr_optim=0.2,
    grad_steps=10,
    seed=SEED,
    verbose=False,
)
r_neural = neural_opt.run(FUNCTION, BOUNDS, n_iter=N_ITER)
print(f"   ✓ f_best = {r_neural.f_best:.6f}  (x = {np.round(r_neural.x_best, 4)})")

print("\n[2/3] Running RFSurrogate ...")
rf_opt = RFSurrogateOptimizer(
    n_estimators=40,
    n_init=N_INIT,
    lr_optim=0.2,
    grad_steps=5,
    fd_eps=1e-2,
    seed=SEED,
    verbose=False,
)
r_rf = rf_opt.run(FUNCTION, BOUNDS, n_iter=N_ITER)
print(f"   ✓ f_best = {r_rf.f_best:.6f}  (x = {np.round(r_rf.x_best, 4)})")

print("\n[3/3] Running RandomSearch (baseline) ...")
rs_opt = RandomSearchOptimizer(seed=SEED)
r_rs = rs_opt.run(FUNCTION, BOUNDS, n_evals=N_RS)
print(f"   ✓ f_best = {r_rs.f_best:.6f}  (x = {np.round(r_rs.x_best, 4)})")


# ── ASCII Convergence Plot ──────────────────────────────────────────────────────
def ascii_convergence(histories, labels, width=60, height=18):
    """Print a simple ASCII convergence curve."""
    # Find y range (log scale for Rosenbrock)
    all_vals = [v for h in histories for v in h]
    y_min = max(min(all_vals), 1e-8)
    y_max = max(all_vals)
    import math
    log_min = math.log10(y_min + 1e-10)
    log_max = math.log10(y_max + 1)

    n_pts = max(len(h) for h in histories)
    chars = "o#@"

    print("\n" + "─" * (width + 14))
    print(f"  Convergence  (log₁₀ scale, {n_pts} evaluations)")
    print("─" * (width + 14))

    # Build grid
    grid = [[" "] * width for _ in range(height)]
    for i, (hist, label, ch) in enumerate(zip(histories, labels, chars)):
        for t, val in enumerate(hist):
            col = int((t / (n_pts - 1)) * (width - 1))
            col = min(col, width - 1)
            log_val = math.log10(val + 1e-10)
            row = int((log_max - log_val) / (log_max - log_min + 1e-10) * (height - 1))
            row = min(max(row, 0), height - 1)
            if grid[row][col] == " ":
                grid[row][col] = ch

    for row_idx, row in enumerate(grid):
        log_y = log_max - (row_idx / (height - 1)) * (log_max - log_min)
        label = f"  1e{log_y:+.1f} |"
        print(label + "".join(row))

    print("         +" + "-" * width)
    print("          0" + " " * (width // 2 - 2) + f"{n_pts//2}" + " " * (width // 2 - 4) + f"{n_pts}")
    print("          evaluations →")
    print()
    for ch, label in zip(chars, labels):
        print(f"   {ch}  {label}")
    print()


ascii_convergence(
    [r_neural.history, r_rf.history, r_rs.history],
    [r_neural.method, r_rf.method, r_rs.method],
)


# ── Summary Table ───────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print(f"  {'Method':<22} {'f_best':>12}  {'evals':>6}")
print("─" * 50)
for r in [r_neural, r_rf, r_rs]:
    print(f"  {r.method:<22} {r.f_best:>12.6f}  {r.n_evals:>6}")
print("─" * 50)
global_min_loc = "(1,...,1)" if "Rosenbrock" in FUNCTION_NAME else "(0,...,0)"
print(f"\n  Global min of {FUNCTION_NAME} = 0.0 at x = {global_min_loc}")

# Improvement vs random search
for r in [r_neural, r_rf]:
    if r_rs.f_best > 0:
        improvement = (r_rs.f_best - r.f_best) / r_rs.f_best * 100
        print(f"  {r.method} beats random search by {improvement:.1f}%")
print()
