# scm-equilibrium

Compute SM equilibria for the **Simple Closed Model** (SCM) from [Deshpande & Sohoni (2021)](https://arxiv.org/abs/2109.09248). Supports linear, piecewise-linear concave (PLC), and general S-segment SPLC utilities. Includes the **Consumer Choice Game** (CCG) framework for analysing strategic preference expression, zone decomposition, Nash equilibrium search, and visualization.

## Quick start

```bash
pip install -r requirements.txt
python main.py                                        # edit economy in code
python cli.py examples/economy_2x2_linear.json        # or load from JSON
python examples/paper_2x2_reproduction.py             # reproduce paper's 2×2 analysis with verification
```

## The model

The Simple Closed Model is a general-equilibrium economic model with **m labour classes** and **n goods**. Each class supplies labour, earns wages, and spends those wages on goods. The model has three primitives:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| **T** | (m, n) | Technology matrix: `T[i,j]` = units of labour class *i* needed to produce one unit of good *j* |
| **U** | (m, n) | Utility matrix: `U[i,j]` = utility per unit of good *j* for class *i* |
| **Y** | (m,) | Labour endowments: `Y[i]` = total labour supply of class *i* |

For PLC utilities, **U** is replaced by **U1**, **U2** (segment utilities with U2 ≤ U1) and **L1** (segment-1 capacity limits). For general SPLC, **U** has shape (m, n, S) with S segments of decreasing marginal utility.

### SM equilibrium

A price vector **p** is an **SM equilibrium** if applying one full SCM round returns the same **p**. At equilibrium, 10 conditions hold simultaneously: money conservation, price non-negativity, labour feasibility, production non-negativity, market clearing, budget exhaustion, wage consistency, bang-per-buck optimality, fixed-point stability, and production optimality.

### Tatonnement

**Tatonnement** iterates the SCM map `p_{t+1} = SCM_round(p_t)` to converge to an SM equilibrium. **Damped tatonnement** uses `p_{t+1} = (1-α)p_t + α·G(p_t)` with optional price normalisation, which stabilises cycling and diverging economies.

### Consumer Choice Game (CCG)

The CCG models **strategic preference expression**. Each labour class has true utilities `U_true` but can express different preferences `U_expressed`. The economy runs at equilibrium under `U_expressed`, and each player's payoff is evaluated using `U_true` on the resulting allocations.

Key concepts from the paper:
- **Friction** = consumers playing `U_expressed ≠ U_true` (brand loyalty, habit, ignorance)
- **AI agents** = forcing `U_expressed → U_true` (optimal preference expression)
- **Zone decomposition (I, J, F)**: the strategy space decomposes into combinatorial zones indexed by active labour I, active goods J, and Fisher forest F. Within each zone, payoffs are smooth algebraic functions. Zone boundaries are regime shifts.
- **Fisher forest (F)**: the spending pattern — which goods each class buys, ordered by bang-per-buck.

## Usage

### Python API

```python
import numpy as np
from scm import solve_robust, check_scm_equilibrium

T = np.array([[1.0, 0.0], [1.0, 1.0]])
U = np.array([[1.0, 0.8], [0.8, 1.0]])
Y = np.array([2.0, 4.0])

result = solve_robust(T, U, Y, p_init=np.array([1.0, 1.0]))

print(f"Prices: {result['p']}")
print(f"Production: {result['q']}")
print(f"Allocations:\n{result['X']}")

checks, all_pass = check_scm_equilibrium(result, T, U, Y)
```

### CCG analysis

```python
from scm import ccg_payoff_detailed, ccg_zone_map, extract_forest

# Single payoff evaluation
payoffs, payoff_mat, wages, prices, quantities, X, zone = \
    ccg_payoff_detailed(T, U_true, U_expressed, Y, p_init)

# 2D zone map over strategy space
def U_func(params):
    return np.array([[params['alpha'], 1], [params['beta'], 1]])

zone_grid, payoff_grid, wage_grid, forest_grid = ccg_zone_map(
    T, U_true, Y, p_init, U_func, alpha_range, beta_range)
```

### Nash equilibrium search

```python
from scm import find_nash_candidates

candidates = find_nash_candidates(T, U_true, Y, p_init, n_restarts=5)
best = candidates[0]
print(best['payoffs'], best['convergence_gap'])
```

### CLI

```bash
python cli.py examples/economy_2x2_linear.json
python cli.py examples/economy_2x2_linear.json --max-iter 300 --tol 1e-8
```

## API reference

### Core functions

| Function | Description |
|----------|-------------|
| `solve_production(T, Y, p)` | Production LP: quantities, wages, active sets I, J |
| `solve_fisher(U, q, budgets)` | Linear Fisher market (Eisenberg-Gale) |
| `solve_fisher_plc(U1, U2, L1, q, budgets)` | 2-segment PLC Fisher market |
| `solve_fisher_splc(U, L, q, budgets)` | General S-segment PLC Fisher market |
| `scm_round(T, U, Y, p)` | One full SCM round (linear) |

### Equilibrium solvers

| Function | Description |
|----------|-------------|
| `compute_equilibrium(T, U, Y, p_init)` | Tatonnement loop (linear) |
| `compute_equilibrium_plc(...)` | Tatonnement loop (2-segment PLC) |
| `compute_equilibrium_splc(...)` | Tatonnement loop (SPLC) with damping |
| `solve_robust(T, U, Y, p_init)` | Cascading solver: standard → Broyden → damped |
| `solve_damped(T, U, Y, p_init)` | Damped tatonnement with alpha sweep |
| `solve_broyden(T, U, Y, p_init)` | Broyden's quasi-Newton method |

### Consumer Choice Game

| Function | Description |
|----------|-------------|
| `ccg_payoff(T, U_true, U_expressed, Y, p_init)` | CCG payoff evaluation |
| `ccg_payoff_detailed(...)` | Payoff with per-good breakdown, zone data, Fisher forest |
| `ccg_sweep(T, U_true, Y, p_init, U_func, grid)` | Sweep payoffs over parameter grid |
| `ccg_gradient(T, U_true, U_expressed, Y, p_init)` | Numerical Jacobian |
| `ccg_zone_map(...)` | 2D zone structure map with forest tracking |
| `extract_forest(U, p, X, I, J)` | Extract Fisher forest from equilibrium |
| `describe_forest(I, J, forest, m)` | Structural classification of forest pattern |

### Nash equilibrium

| Function | Description |
|----------|-------------|
| `best_response_direction(...)` | Gradient direction for one player |
| `nash_iteration(T, U_true, U_init, Y, p_init)` | Simultaneous gradient ascent |
| `find_nash_candidates(T, U_true, Y, p_init, n_restarts)` | Multi-start Nash search |

### Visualization (requires matplotlib)

| Function | Description |
|----------|-------------|
| `plot_zone_map(zone_grid, p1, p2)` | 2D zone heatmap with boundaries |
| `plot_zone_map_with_payoff(...)` | Side-by-side zone + payoff heatmap |
| `plot_payoff_trajectory(params, payoffs)` | 1D payoff curves with zone transitions |
| `plot_forest_diagram(X, I, J)` | Bipartite spending-flow graph |
| `plot_gradient_field(grad, p1, p2)` | Quiver plot on zone background |

### Verification

| Function | Description |
|----------|-------------|
| `check_scm_equilibrium(result, T, U, Y)` | Check all 10 SM equilibrium conditions |
| `check_plc_equilibrium(...)` | Check all 11 PLC equilibrium conditions |

## Project structure

```
scm-equilibrium/
  scm/                           Core library (general m×n)
  examples/                      Runnable scripts and JSON economy configs
  tests/                         Test suite (155 tests)
  docs/                          Generated documentation and figures (not tracked)
  matlab_reference/              Original MATLAB codes for provenance
  main.py                        Quick-start script
  cli.py                         CLI interface
```

## Paper verification

The solver has been verified against the paper's 2×2 soap market economy (T=[[0.2501,0],[0.25,1]], U=[[1,1],[1,1]], Y=[2,4]).

**Appendix A tables:** Table 1 achieves a perfect 32/32 match. Table 2 achieves 12/32; mismatches occur in the multi-equilibria region (αβ > 1) where the Eisenberg-Gale solver selects a different equilibrium branch than the MATLAB Lemke pivot.

**Section 6 zone decomposition (U = [[α, 1], [β, 1]]):**

| Zone | Match | Description |
|------|-------|-------------|
| Forest-1 | 94.7% (666/703) | C0→G0, C1→both. Mismatches on α=β diagonal only. |
| Forest-2 | 100% (561/561) | C0→both, C1→G0 |
| Forest-3 | 100% (12/12) | C0→G1, C1→both |
| Forest-4 | 100% (204/204) | C0→G1, C1→G0 (complete specialisation) |
| Zone-5 | 100% (120/120) | Only class 1 active (β ≤ 1/4) |

Overall: **97.7%** (1563/1600). Excluding the α=β non-generic boundary: **100%** (1482/1482). All five zones predicted by the paper are found by the solver.

## Testing

```bash
pip install pytest matplotlib
pytest tests/ -v
```

## Dependencies

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- CVXPY >= 1.3 (with CLARABEL solver, included by default)
- Matplotlib >= 3.5 (optional, for visualization)

## References

Deshpande, A. & Sohoni, M. (2021). *A Simple Closed Economy Model.* arXiv:2109.09248.
