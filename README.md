# surrogate-gradient-optim-lab

**Learning gradient approximations for black-box functions.**

Surrogate gradient optimization is useful when:
- The function has no gradient API (simulators, hardware, human feedback)
- Each evaluation is expensive (must budget evaluations carefully)
- The function is noisy or non-differentiable

This repo implements two surrogate methods and a random-search baseline,
all operating on the same black-box interface: only `f(x) → float` is visible.

---

## Methods

### 1. Neural Surrogate (`NeuralSurrogateOptimizer`)
Trains a small MLP (PyTorch) to approximate `f(x)`.  
Gradients are computed via **autograd** (`torch.autograd`) through the network.

```
Algorithm:
  1. Sample N_init points via Latin Hypercube Sampling
  2. Train MLP on (X, f(X))
  3. Gradient-descend x through the MLP: x ← x − η · ∇f̂(x)
  4. Evaluate real black-box at new x, add to training set
  5. Refit surrogate (warm-start), repeat
```

### 2. Random-Forest Surrogate (`RFSurrogateOptimizer`)
Trains a Random Forest regressor, then approximates gradients via **finite differences**:

```
∂f̂/∂xᵢ ≈ (f̂(x + ε·eᵢ) − f̂(x − ε·eᵢ)) / (2ε)
```

Multiple candidate starts (exploitation + random restart) reduce the risk of getting stuck.

### 3. Random Search (baseline)
Uniform random sampling in bounds.  Used to measure how much the surrogates help.

---

## Quick Start

```bash
pip install numpy scipy scikit-learn torch
python demo.py
```

### Demo output (Sphere, 4-D, 50 evaluations total)

```
============================================================
  Surrogate Gradient Optimization Demo
  Function : Sphere (4-D)  (dim=4)
  Bounds   : (-3.0, 3.0)  (each axis)
  Budget   : 15 init + 35 guided = 50 evals
============================================================

[1/3] Running NeuralSurrogate ...
   ✓ f_best = 1.081710  (x = [ 0.3365  0.5817 -0.2742 -0.745 ])

[2/3] Running RFSurrogate ...
   ✓ f_best = 2.316201  (x = [-0.1645 -0.5092 -0.4116 -1.364 ])

[3/3] Running RandomSearch (baseline) ...
   ✓ f_best = 2.266590  (x = [ 0.8083  0.3215  0.3552 -1.1763])
```

### Convergence plot (log₁₀ scale)

```
  1e+1.0 |@@@
  1e+0.9 |   @@ @@@
  1e+0.8 |         @@ @@@@@
  1e+0.7 |ooooo ooooo ooooo ooo
  1e+0.6 |                  @  oo
  1e+0.5 |                        o
  1e+0.4 |                  ##### ###oo ooooo ooo## ##### ##### #### #
  1e+0.3 |                                       oo
  1e+0.2 |
  1e+0.1 |                                           o
  1e+0.1 |                                            ooo ooooo oooo o
         +------------------------------------------------------------
          0                            25                          50
          evaluations →

   o  NeuralSurrogate    f_best = 1.08   (52% better than random)
   #  RFSurrogate        f_best = 2.32
   @  RandomSearch       f_best = 2.27   (baseline)
```

The neural surrogate uses gradients through the learned model to consistently descend
toward the optimum, while random search plateaus once it gets lucky.

---

## Usage

```python
from surrogate_gradient import NeuralSurrogateOptimizer, RFSurrogateOptimizer

# Define your black-box (no gradient API needed)
def my_simulator(x):
    # ... call hardware / simulator / expensive model ...
    return float(cost)

bounds = [(-5.0, 5.0)] * 3   # 3-D search space

# Neural surrogate
opt = NeuralSurrogateOptimizer(
    hidden=(64, 64),    # MLP architecture
    n_init=20,          # LHS samples before guided search
    n_epochs=300,       # surrogate training epochs per iter
    lr_optim=0.1,       # gradient step size on x
    grad_steps=10,      # gradient steps per eval
)
result = opt.run(my_simulator, bounds, n_iter=50)
print(result.x_best, result.f_best)

# RF surrogate
from surrogate_gradient import RFSurrogateOptimizer
opt = RFSurrogateOptimizer(n_estimators=100, n_init=20, lr_optim=0.1)
result = opt.run(my_simulator, bounds, n_iter=50)
```

---

## Benchmark Functions

```python
from surrogate_gradient import rosenbrock, sphere, ackley, rastrigin

# All accept 1-D numpy arrays, return float
rosenbrock(np.ones(2))   # 0.0  (global min at x=1)
sphere(np.zeros(4))      # 0.0
ackley(np.zeros(2))      # ~0.0
rastrigin(np.zeros(3))   # 0.0
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

28 tests cover:
- Benchmark function correctness
- Latin Hypercube Sampling stratification
- History monotonicity (best-so-far is always non-increasing)
- Bound enforcement
- Convergence on sphere
- Neural vs RF vs random comparison

---

## Structure

```
surrogate_gradient/
  __init__.py       — public API
  benchmarks.py     — rosenbrock, sphere, ackley, rastrigin
  neural.py         — NeuralSurrogateOptimizer (PyTorch autograd)
  rf.py             — RFSurrogateOptimizer (sklearn RF + finite diff)
  random_search.py  — RandomSearchOptimizer (baseline)
  sampling.py       — Latin Hypercube Sampling
  result.py         — OptimizationResult dataclass

tests/
  test_surrogates.py   — 28 unit tests

demo.py               — comparison demo with ASCII convergence plot
```

---

## When to Use Each Method

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| Neural surrogate | Smooth gradient estimates, works well on continuous functions | Needs enough data to train; slower per iter |
| RF surrogate | Non-parametric, robust to outliers | FD gradient is noisy; scales poorly to high-dim |
| Random search | Zero assumptions, parallelizable | Wasteful — doesn't learn anything |

Both surrogate methods shine when evaluations are expensive (< 200 budget).
For cheap functions, just use scipy.optimize.

---

## License

MIT
