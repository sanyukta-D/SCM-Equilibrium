# scm-equilibrium

Compute SM equilibria for the **Simple Closed Model** (SCM) from [Deshpande & Sohoni (2021)](https://arxiv.org/abs/2109.09248). Supports linear, 2-segment PLC, and general S-segment SPLC utilities. Includes the **Consumer Choice Game** (CCG) framework for analysing strategic preference expression, zone decomposition, Nash equilibrium search, and visualization.

## Quick start

```bash
pip install -r requirements.txt
python main.py                                        # edit economy in code
python cli.py examples/economy_2x2_linear.json        # or load from JSON
python examples/ccg_soap_market.py                    # CCG analysis (soap market from paper)
python examples/ccg_analysis_template.py              # CCG template (modify for your economy)
python examples/full_demo.py                          # full walkthrough with all visualizations
```

For a guided visual tour, open `docs/walkthrough/CCG_Walkthrough.html` in your browser.

## What is the SCM?

The Simple Closed Model is a general-equilibrium economic model with **m labour classes** and **n goods**. Each class supplies labour, earns wages, and spends those wages on goods. The model has three primitives:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| **T** | (m, n) | Technology matrix: `T[i,j]` = units of labour class *i* needed to produce one unit of good *j* |
| **U** | (m, n) | Utility matrix: `U[i,j]` = utility per unit of good *j* for class *i* |
| **Y** | (m,) | Labour endowments: `Y[i]` = total labour supply of class *i* |

For PLC utilities, **U** is replaced by **U1**, **U2** (segment utilities with U2 ≤ U1) and **L1** (segment-1 capacity limits). For general SPLC, **U** has shape (m, n, S) with S segments of decreasing marginal utility.

## Two key concepts

### SM Equilibrium (the static object)

A price vector **p** is an **SM equilibrium** if applying one full SCM round returns the same **p**. At equilibrium, 10 conditions hold simultaneously:

1. **Money conservation:** total revenue equals total wages (p · q = ΣW)
2. **Price non-negativity:** p ≥ 0
3. **Labour feasibility:** production doesn't exceed labour supply (T q ≤ Y)
4. **Production non-negativity:** q ≥ 0
5. **Market clearing:** all produced goods are consumed
6. **Budget exhaustion:** every worker spends all income
7. **Wage consistency:** wages match the matrix-inverse formula
8. **Bang-per-buck optimality:** consumers spend only on goods with maximum utility-per-dollar
9. **Fixed point:** one more SCM round leaves prices unchanged
10. **Production optimality:** firms maximize profit at these prices

### Tatonnement (the dynamic process)

**Tatonnement** iterates the SCM map `p_{t+1} = SCM_round(p_t)` hoping to converge to an SM equilibrium. It may converge (finding an exact equilibrium), cycle (finding an approximate one), or diverge.

**Damped tatonnement** uses `p_{t+1} = (1-α)p_t + α·G(p_t)` with optional price normalisation, which stabilises cycling and diverging economies.

## One SCM round

Each iteration of tatonnement applies these steps:

```
prices p  →  Production LP: max p·q s.t. Tq ≤ Y  →  quantities q, active sets I, J
          →  Wages: w = p[J] @ inv(T[I,J]),  W[i] = w[i] · Y[i]
          →  Fisher Market: allocate goods to classes using EG convex program
          →  New prices p'  (from market-clearing dual variables)
```

## Consumer Choice Game (CCG)

The CCG models **strategic preference expression** in the SCM. Each labour class (player) has true utilities `U_true` but can express different preferences `U_expressed`. The economy runs at equilibrium under `U_expressed`, and each player's payoff is evaluated using `U_true` on the resulting allocations.

Key concepts:

- **Friction** = consumers playing `U_expressed ≠ U_true` (brand loyalty, habit, ignorance)
- **AI agents** = forcing `U_expressed → U_true` (optimal preference expression)
- **Zone decomposition (I, J, F)**: the strategy space decomposes into combinatorial zones indexed by active labour I, active goods J, and Fisher forest F. Within each zone, payoffs are smooth algebraic functions. Zone boundaries are regime shifts.
- **Fisher forest (F)**: the spending pattern — which goods each class buys, ordered by bang-per-buck (BPB = U[i,j]/p[j]).

### CCG Python API

```python
from scm.ccg import ccg_payoff, ccg_payoff_detailed, ccg_sweep, ccg_gradient
from scm.ccg import ccg_zone_map, extract_forest, zone_label

# Single payoff evaluation
payoffs, result = ccg_payoff(T, U_true, U_expressed, Y, p_init)

# Detailed payoff with decomposition and forest
payoffs, payoff_mat, wages, prices, quantities, X, zone = \
    ccg_payoff_detailed(T, U_true, U_expressed, Y, p_init)

# Sweep over parameter grid (replicates MATLAB FeigningU.m)
# Section 6 convention: alpha, beta = pref for good 0 relative to good 1
def U_func(params):
    return np.array([[params['alpha'], 1], [params['beta'], 1]])

results = ccg_sweep(T, U_true, Y, p_init, U_func, param_grid)

# 2D zone map
zone_grid, payoff_grid, wage_grid, forest_grid = ccg_zone_map(
    T, U_true, Y, p_init, U_func,
    alpha_range, beta_range,
    param1_name='alpha', param2_name='beta')

# Numerical gradient (Jacobian)
J = ccg_gradient(T, U_true, U_expressed, Y, p_init)
```

### Nash equilibrium search

```python
from scm.nash import nash_iteration, find_nash_candidates

# Single run from a starting point
result = nash_iteration(T, U_true, U_init, Y, p_init,
                        max_iter=50, lr=0.1, tol=1e-4)

# Multi-start search (ranked by convergence quality)
candidates = find_nash_candidates(T, U_true, Y, p_init, n_restarts=5)
best = candidates[0]
print(best['payoffs'], best['convergence_gap'])
```

### Visualization (requires matplotlib)

```python
from scm.visualize import (
    plot_zone_map, plot_zone_map_with_payoff,
    plot_payoff_trajectory, plot_wage_trajectory,
    plot_allocation_pattern, plot_forest_diagram,
    plot_gradient_field,
)

# 2D zone structure heatmap
plot_zone_map(zone_grid, alpha_range, beta_range,
              output_file='zone_map.png')

# Side-by-side zone + payoff heatmap
plot_zone_map_with_payoff(zone_grid, payoff_grid, alpha_range, beta_range,
                           player=0, output_file='zone_payoff.png')

# 1D payoff/wage/price trajectories with zone transition markers
plot_payoff_trajectory(param_vals, payoff_arr, zone_labels=zone_labels)

# Bipartite graph showing spending flows (Fisher forest)
plot_forest_diagram(X, I, J, output_file='forest.png')

# Gradient direction quiver plot overlaid on zone map
plot_gradient_field(grad_grid, param1_grid, param2_grid, zone_grid=zone_grid)
```

## Usage

### Option 1: Edit `main.py`

Open `main.py`, set your economy parameters (T, U, Y, p_init), and run:

```bash
python main.py
```

### Option 2: JSON config via `cli.py`

Create a JSON file describing your economy:

```json
{
    "type": "linear",
    "T": [[1, 0], [1, 1]],
    "U": [[1, 0.8], [0.8, 1]],
    "Y": [2, 4],
    "p_init": [1, 1]
}
```

Then run:

```bash
python cli.py your_economy.json
python cli.py your_economy.json --max-iter 300 --tol 1e-8
```

### Option 3: Python API

```python
import numpy as np
from scm import compute_equilibrium, check_scm_equilibrium

T = np.array([[1.0, 0.0], [1.0, 1.0]])
U = np.array([[1.0, 0.8], [0.8, 1.0]])
Y = np.array([2.0, 4.0])

result = compute_equilibrium(T, U, Y, p_init=np.array([1.0, 1.0]))

print(f"Status: {result['status']}")
print(f"Prices: {result['p']}")
print(f"Production: {result['q']}")
print(f"Wages: {result['W']}")
print(f"Allocations:\n{result['X']}")

checks, all_pass = check_scm_equilibrium(result, T, U, Y)
```

For PLC:

```python
from scm import compute_equilibrium_plc, check_plc_equilibrium
result = compute_equilibrium_plc(T, U1, U2, L1, Y, p_init)
checks, ok = check_plc_equilibrium(result, T, U1, U2, L1, Y)
```

For general SPLC (S segments):

```python
from scm import compute_equilibrium_splc
result = compute_equilibrium_splc(T, U, L, Y, p_init, damped=True, alpha=0.3)
```

## API reference

### Core functions

| Function | Description |
|----------|-------------|
| `solve_production(T, Y, p)` | Production LP: returns quantities q, wage rates w, wages W, active sets I, J |
| `solve_fisher(U, q, budgets)` | Linear Fisher market: returns prices, allocations, bang-per-buck |
| `solve_fisher_plc(U1, U2, L1, q, budgets)` | 2-segment PLC Fisher market |
| `solve_fisher_splc(U, L, q, budgets)` | General S-segment PLC Fisher market |
| `scm_round(T, U, Y, p)` | One full SCM round (linear): prices in, prices out |
| `scm_round_plc(T, U1, U2, L1, Y, p)` | One PLC SCM round (2-segment) |
| `scm_round_splc(T, U, L, Y, p)` | One SPLC SCM round (S-segment) |

### Equilibrium solvers

| Function | Description |
|----------|-------------|
| `compute_equilibrium(T, U, Y, p_init, ...)` | Tatonnement loop (linear) |
| `compute_equilibrium_plc(T, U1, U2, L1, Y, p_init, ...)` | Tatonnement loop (2-segment PLC) |
| `compute_equilibrium_splc(T, U, L, Y, p_init, ...)` | Tatonnement loop (SPLC) with damping + normalisation |

### Robust solvers (v0.2.0)

| Function | Description |
|----------|-------------|
| `solve_robust(T, U, Y, p_init)` | Cascading solver: standard → Broyden → damped. Solves 33/35 benchmark economies. |
| `solve_damped(T, U, Y, p_init, ...)` | Damped tatonnement with alpha sweep and normalisation. |
| `solve_broyden(T, U, Y, p_init, ...)` | Broyden's quasi-Newton on F(p) = G(p) - p = 0. |

### Consumer Choice Game (v0.4.0)

| Function | Description |
|----------|-------------|
| `ccg_payoff(T, U_true, U_expressed, Y, p_init)` | CCG payoff: equilibrium under U_expressed, evaluate at U_true |
| `ccg_payoff_detailed(...)` | Payoff with per-good breakdown, zone data, Fisher forest |
| `ccg_sweep(T, U_true, Y, p_init, U_func, grid)` | Sweep payoffs over parameter grid (≈ FeigningU.m) |
| `ccg_gradient(T, U_true, U_expressed, Y, p_init)` | Numerical Jacobian via finite differences |
| `ccg_zone_map(T, U_true, Y, p_init, U_func, p1, p2)` | 2D zone structure map with forest tracking |
| `extract_forest(U, p, X, I, J)` | Extract Fisher forest (BPB spending pattern) from equilibrium |
| `zone_label(I, J, forest)` | Readable zone label string |

### Nash equilibrium (v0.4.0)

| Function | Description |
|----------|-------------|
| `best_response_direction(T, U_true, U_expressed, Y, p_init, player)` | Gradient direction for one player |
| `nash_iteration(T, U_true, U_init, Y, p_init)` | Simultaneous gradient ascent for all players |
| `find_nash_candidates(T, U_true, Y, p_init, n_restarts)` | Multi-start Nash search, ranked by convergence |

### Visualization (v0.4.0, requires matplotlib)

| Function | Description |
|----------|-------------|
| `plot_zone_map(zone_grid, p1, p2)` | 2D zone heatmap with boundaries |
| `plot_zone_map_with_payoff(zone_grid, payoff_grid, p1, p2)` | Side-by-side zone + payoff heatmap |
| `plot_payoff_trajectory(params, payoffs)` | 1D payoff curves with zone transitions |
| `plot_wage_trajectory(params, wages)` | 1D wage curves |
| `plot_price_trajectory(params, prices)` | 1D price curves |
| `plot_allocation_pattern(X)` | Stacked bar chart of goods per class |
| `plot_forest_diagram(X, I, J)` | Bipartite spending-flow graph |
| `plot_gradient_field(grad, p1, p2)` | Quiver plot on zone background |

### Verification

| Function | Description |
|----------|-------------|
| `check_scm_equilibrium(result, T, U, Y, tol)` | Check all 10 SM equilibrium conditions |
| `check_plc_equilibrium(result, T, U1, U2, L1, Y, tol)` | Check all 11 PLC equilibrium conditions |

## Project structure

```
scm-equilibrium/
  scm/                           Core Python library
    production.py                  Production LP and wage computation
    fisher_market.py               Linear Fisher market (Eisenberg-Gale)
    fisher_market_plc.py           2-segment PLC Fisher market
    fisher_market_splc.py          General S-segment SPLC Fisher market
    scm_round.py                   One SCM round (linear)
    scm_round_plc.py               One SCM round (PLC)
    scm_round_splc.py              One SCM round (SPLC)
    equilibrium.py                 Tatonnement iterator (linear)
    equilibrium_plc.py             Tatonnement iterator (PLC)
    equilibrium_splc.py            Tatonnement iterator (SPLC)
    solvers.py                     Robust solvers: damped, Broyden, cascading
    verify.py                      Equilibrium condition checker (10/11 conditions)
    ccg.py                         Consumer Choice Game: payoffs, sweeps, zone map, forest
    nash.py                        Nash equilibrium finder: gradient ascent, multi-start
    visualize.py                   Matplotlib plots: zone maps, trajectories, forests

  examples/                      Runnable scripts and economy configs
    ccg_soap_market.py             CCG analysis of paper's 2×2 soap market
    ccg_analysis_template.py       Generic CCG template (copy and modify for your economy)
    full_demo.py                   Complete walkthrough: 2×2 + 3×3 with all visualizations
    three_piece_plc.py             3-piece PLC utility example
    economy_*.json                 Economy JSON configs for cli.py

  tests/                         Test suite (40 tests)
    test_ccg.py                    CCG: payoffs, sweeps, zones, forest extraction
    test_nash.py                   Nash: gradient, iteration, multi-start
    test_visualize.py              Visualization: all plot functions
    test_equilibrium.py            Core equilibrium convergence
    test_fisher_market.py          Fisher market allocation
    test_production.py             Production LP
    test_solvers.py                Robust solver cascade
    test_many_economies.py         35-economy benchmark suite

  docs/
    walkthrough/                   Interactive CCG walkthrough
      CCG_Walkthrough.html           Open in browser — full guided tour with figures
      figures/                       All generated plots (2×2 and 3×3)
    paper_verification.md          Detailed verification against paper results
    CCG_CAPABILITIES.md            Analysis of CCG capabilities and gaps
    DAMPED_TATONNEMENT_MATH.md     Mathematical derivation of damped tatonnement
    PROJECT_SUMMARY.md             Development progress log

  matlab_reference/              Original MATLAB codes for provenance
  main.py                       Quick-start: define economy in code
  cli.py                        CLI: load economy from JSON
```

## Paper verification

The solver has been verified against the paper's 2×2 soap market economy (T=[[0.2501,0],[0.25,1]], U_true=[[1,1],[1,1]], Y=[2,4]).

**Appendix A tables:** Table 1 achieves a perfect 32/32 match against values extracted directly from the paper's PDF. Table 2 achieves 12/32; mismatches occur in the multi-equilibria region (αβ > 1) where the Eisenberg-Gale solver selects a different equilibrium branch than the MATLAB Lemke pivot.

**Section 6 zone decomposition (U = [[α, 1], [β, 1]]):**

| Zone | Match | Description |
|------|-------|-------------|
| Forest-1 | 96.4% | C0→G0, C1→both (class 1 indifferent). 55 mismatches on α=β diagonal only. |
| Forest-2 | 100% | C0→both, C1→G0 (class 0 indifferent) |
| Forest-3 | 100% | C0→G1, C1→both |
| Zone-5 | 100% | Only class 1 active (β ≤ 1/4) |

Excluding the α=β non-generic boundary, match rate is **100%**. Forest-4 (C0→G1, C1→G0) does not exist for β < T[1,1]/T[1,0] = 4 in this economy due to production constraints; the paper's generic zone condition "α ≤ 1/2, β ≥ 1/2" has economy-specific boundaries.

## Testing

```bash
pip install pytest matplotlib
pytest tests/ -v                              # full test suite
pytest tests/test_ccg.py -v                   # CCG + forest tests only
pytest tests/test_nash.py -v                  # Nash equilibrium tests
pytest tests/test_visualize.py -v             # Visualization tests
python tests/test_many_economies.py           # 35-economy benchmark table
```

## Dependencies

- Python >= 3.9
- NumPy >= 1.21
- SciPy >= 1.7
- CVXPY >= 1.3 (with CLARABEL solver, included by default)
- Matplotlib >= 3.5 (optional, for visualization)

## References

Deshpande, A. & Sohoni, M. (2021). *A Simple Closed Economy Model.* arXiv:2109.09248.
