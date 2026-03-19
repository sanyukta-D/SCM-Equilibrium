# Simple Closed Model — Equilibrium Computation and Consumer Choice Game

This repository implements the economic model and computational framework from:

> **Deshpande, A. & Sohoni, M. (2021).** *A Simple Closed Economy Model.* [arXiv:2109.09248](https://arxiv.org/abs/2109.09248)

The Simple Closed Model (SCM) is a general-equilibrium model of an economy with multiple labour classes and goods. This codebase provides solvers for computing SM equilibria, a full implementation of the Consumer Choice Game (CCG) for analysing strategic preference expression, and tools for zone decomposition, Nash equilibrium search, and visualisation.

---

## The Model

The SCM describes an economy with **m labour classes** and **n goods**. Each class supplies labour, earns wages determined by production, and spends those wages on goods according to its preferences. The economy is characterised by three primitives:

- **T** (m × n) — the technology matrix, where T[i,j] is the labour of class i required to produce one unit of good j
- **U** (m × n) — the utility matrix, where U[i,j] is the marginal utility of good j for class i
- **Y** (m,) — the labour endowment vector, where Y[i] is the total labour supply of class i

A price vector **p** is an **SM equilibrium** when applying one full SCM round — production optimisation, wage computation, and consumer spending via a Fisher market — returns the same price vector. At equilibrium, ten conditions hold simultaneously: money conservation, price non-negativity, labour feasibility, production non-negativity, market clearing, budget exhaustion, wage consistency, bang-per-buck optimality, fixed-point stability, and production optimality.

The codebase supports three utility variants: linear utilities, two-segment piecewise-linear-concave (PLC) utilities (with diminishing returns after a capacity threshold), and general S-segment SPLC utilities.

## The Consumer Choice Game

The CCG, introduced in Section 6 of the paper, models **strategic preference expression**. Each labour class has true utilities U_true but may express different preferences U_expressed to the market. The economy reaches equilibrium under expressed utilities, and each class's payoff is evaluated using its true utilities on the resulting allocations.

This framework captures several economically meaningful phenomena:

- **Friction** arises when consumers play U_expressed ≠ U_true, whether due to brand loyalty, habit, or incomplete information.
- **AI intermediation** corresponds to forcing U_expressed → U_true, eliminating friction and recovering optimal preference expression.
- **Zone decomposition** — the strategy space decomposes into combinatorial zones indexed by the active labour set I, active goods set J, and Fisher forest F. Within each zone, payoffs are smooth algebraic functions; zone boundaries represent regime shifts in the equilibrium structure.
- **Fisher forest** — the bipartite spending pattern describing which goods each class purchases, ordered by bang-per-buck.

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

The core dependencies are NumPy, SciPy, and CVXPY (with the CLARABEL solver, included by default). Matplotlib is needed only for visualisation.

### Running Examples

```bash
python main.py                                        # minimal example with inline economy
python cli.py examples/economy_2x2_linear.json        # load economy from JSON
python examples/paper_2x2_reproduction.py             # full reproduction of the paper's 2×2 analysis
python examples/full_demo.py                          # end-to-end walkthrough: equilibrium → CCG → zones → Nash
```

### Python API

Computing an equilibrium:

```python
import numpy as np
from scm import solve_robust, check_scm_equilibrium

T = np.array([[1.0, 0.0], [1.0, 1.0]])
U = np.array([[1.0, 0.8], [0.8, 1.0]])
Y = np.array([2.0, 4.0])

result = solve_robust(T, U, Y, p_init=np.array([1.0, 1.0]))
checks, all_pass = check_scm_equilibrium(result, T, U, Y)
```

Running a CCG analysis:

```python
from scm import ccg_payoff_detailed, ccg_zone_map

# Evaluate payoffs under strategic play
payoffs, payoff_mat, wages, prices, quantities, X, zone = \
    ccg_payoff_detailed(T, U_true, U_expressed, Y, p_init)

# Map the zone structure across a 2D strategy space
zone_grid, payoff_grid, wage_grid, forest_grid = ccg_zone_map(
    T, U_true, Y, p_init, U_func, alpha_range, beta_range)
```

Searching for Nash equilibria:

```python
from scm import find_nash_candidates

candidates = find_nash_candidates(T, U_true, Y, p_init, n_restarts=5)
```

---

## Repository Structure

```
scm/                    Core library — fully general for any m×n economy
  production.py           Production LP and wage computation
  fisher_market.py        Linear-utility Fisher market (Eisenberg-Gale)
  fisher_market_plc.py    Two-segment PLC Fisher market via CVXPY
  fisher_market_splc.py   General S-segment PLC Fisher market
  scm_round.py            One SCM round: production → wages → spending → prices
  equilibrium.py          Tâtonnement iteration loop
  solvers.py              Robust cascading solver, damped tâtonnement, Broyden
  ccg.py                  Consumer Choice Game: payoffs, sweeps, zone maps
  nash.py                 Nash equilibrium search via gradient ascent
  verify.py               Equilibrium verification (10 linear + 11 PLC conditions)
  visualize.py            Zone maps, payoff surfaces, forest diagrams

examples/               Runnable demonstrations and economy configurations
  paper_2x2_reproduction.py   Full verification against the paper's 2×2 economy
  full_demo.py                End-to-end walkthrough of all capabilities
  ccg_soap_market.py          Paper's soap market example with CCG analysis
  ccg_analysis_template.py    Reusable template for CCG analysis of any economy
  economy_*.json              Example economy definitions (2×2, 3×3, linear, PLC)

tests/                  Test suite (155+ tests across all modules)
docs/                   Technical documentation, theory audit, and generated figures
```

The library code in `scm/` is written to be fully general for any m×n economy. Paper-specific names, mappings, and display logic appear only in the example scripts, never in the core library.

## Equilibrium Solvers

The codebase provides several equilibrium computation methods, benchmarked across 35 economies ranging from 2×2 to 6×6:

- **Standard tâtonnement** — iterates the SCM map p_{t+1} = SCM_round(p_t) until convergence. Effective for well-behaved economies but can cycle or diverge.
- **Damped tâtonnement** — uses convex combination p_{t+1} = (1−α)p_t + α·G(p_t) with an automatic step-size sweep and price normalisation. The most reliable single method, solving 32 of 35 benchmark economies.
- **Broyden quasi-Newton** — treats equilibrium-finding as root-finding. Fast convergence when it works, and complements damped tâtonnement on different economy types.
- **Robust cascading solver** (`solve_robust`) — the recommended entry point. Tries standard tâtonnement first, falls back to Broyden, then to damped tâtonnement, returning the best result. Solves 33 of 35 benchmark economies to fixed-point error below 1e-4.

---

## Verification Against the Paper

The solver has been systematically verified against the paper's 2×2 soap market economy (T = [[0.2501, 0], [0.25, 1]], U = [[1, 1], [1, 1]], Y = [2, 4]).

**Appendix A numerical tables:** Table 1 achieves a 32/32 match against the paper's documented equilibria. Table 2 achieves 12/32; mismatches occur in the multi-equilibria region (αβ > 1) where the Eisenberg-Gale convex program selects a different equilibrium branch than the paper's MATLAB Lemke pivot — both are valid equilibria.

**Section 6 zone decomposition** (parameterised as U = [[α, 1], [β, 1]]):

- All five zones predicted by the paper are identified by the solver.
- Overall match: 97.7% (1563 of 1600 grid points).
- Excluding the non-generic α = β boundary: 100% (1482 of 1482).

The full reproduction script (`examples/paper_2x2_reproduction.py`) regenerates all verification tables and zone maps.

## Testing

```bash
pytest tests/ -v
```

The test suite covers production LP solving, Fisher market computation (linear and PLC), single-round SCM dynamics, tâtonnement convergence, all equilibrium solvers, CCG payoff evaluation, zone decomposition, Nash equilibrium search, and visualisation.

---

## Citation

If you use this code in your research, please cite the underlying paper:

```bibtex
@article{deshpande2021simple,
  title={A Simple Closed Economy Model},
  author={Deshpande, Anirudha and Sohoni, Milind},
  journal={arXiv preprint arXiv:2109.09248},
  year={2021}
}
```
