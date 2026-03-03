# scm-equilibrium

Compute SM equilibria for the **Simple Closed Model** (SCM) from [Deshpande & Sohoni (2021)](https://arxiv.org/abs/2109.09248). Supports both linear and piecewise-linear-concave (PLC) utilities.

## Quick start

```bash
pip install -r requirements.txt
python main.py                                        # edit economy in code
python cli.py examples/economy_2x2_linear.json        # or load from JSON
```

## What is the SCM?

The Simple Closed Model is a general-equilibrium economic model with **m labour classes** and **n goods**. Each class supplies labour, earns wages, and spends those wages on goods. The model has three primitives:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| **T** | (m, n) | Technology matrix: `T[i,j]` = units of labour class *i* needed to produce one unit of good *j* |
| **U** | (m, n) | Utility matrix: `U[i,j]` = utility per unit of good *j* for class *i* |
| **Y** | (m,) | Labour endowments: `Y[i]` = total labour supply of class *i* |

For PLC utilities, **U** is replaced by **U1** (segment-1 utilities), **U2** (segment-2 utilities, with U2 <= U1), and **L1** (segment-1 capacity limits).

## Two key concepts

### SM Equilibrium (the static object)

A price vector **p** is an **SM equilibrium** if applying one full SCM round returns the same **p**. At equilibrium, 10 conditions hold simultaneously:

1. **Money conservation:** total revenue equals total wages (p . q = sum of W)
2. **Price non-negativity:** p >= 0
3. **Labour feasibility:** production doesn't exceed labour supply (T q <= Y)
4. **Production non-negativity:** q >= 0
5. **Market clearing:** all produced goods are consumed
6. **Budget exhaustion:** every worker spends all income
7. **Wage consistency:** wages match the matrix-inverse formula
8. **Bang-per-buck optimality:** consumers spend only on goods with maximum utility-per-dollar
9. **Fixed point:** one more SCM round leaves prices unchanged
10. **Production optimality:** firms maximize profit at these prices

For PLC utilities, an additional segment-1 capacity constraint is checked (11 conditions total).

### Tatonnement (the dynamic process)

**Tatonnement** iterates the SCM map `p_{t+1} = SCM_round(p_t)` hoping to converge to an SM equilibrium. It may converge (finding an exact equilibrium), cycle (finding an approximate one), or diverge.

## One SCM round

Each iteration of tatonnement applies these steps:

```
prices p  -->  Production LP: max p.q s.t. Tq <= Y  -->  quantities q, active sets I, J
          -->  Wages: w = p[J] @ inv(T[I,J]),  W[i] = w[i] * Y[i]
          -->  Fisher Market: allocate goods to classes using EG convex program
          -->  New prices p'  (from market-clearing dual variables)
```

## Usage

### Option 1: Edit `main.py`

Open `main.py`, set your economy parameters (T, U, Y, p_init), and run:

```bash
python main.py
```

The script solves the equilibrium, prints results, and verifies all conditions.

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

For PLC economies, use `"type": "plc"` with `"U1"`, `"U2"`, `"L1"` instead of `"U"`. See `examples/` for templates.

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

# Verify all equilibrium conditions
checks, all_pass = check_scm_equilibrium(result, T, U, Y)
```

For PLC:

```python
from scm import compute_equilibrium_plc, check_plc_equilibrium

result = compute_equilibrium_plc(T, U1, U2, L1, Y, p_init)
checks, ok = check_plc_equilibrium(result, T, U1, U2, L1, Y)
```

## API reference

### Core functions

| Function | Description |
|----------|-------------|
| `solve_production(T, Y, p)` | Production LP: returns quantities q, wage rates w, wages W, active sets I, J |
| `solve_fisher(U, q, budgets)` | Linear Fisher market: returns prices, allocations, bang-per-buck |
| `solve_fisher_plc(U1, U2, L1, q, budgets)` | PLC Fisher market: returns prices, segment allocations |
| `scm_round(T, U, Y, p)` | One full SCM round: prices in, prices out |
| `scm_round_plc(T, U1, U2, L1, Y, p)` | One PLC SCM round |

### Equilibrium solvers

| Function | Description |
|----------|-------------|
| `compute_equilibrium(T, U, Y, p_init, ...)` | Tatonnement loop (linear). Returns dict with p, q, W, X, I, J, status |
| `compute_equilibrium_plc(T, U1, U2, L1, Y, p_init, ...)` | Tatonnement loop (PLC). Returns dict with p, q, W, X1, X2, X, I, J, status |

### Verification

| Function | Description |
|----------|-------------|
| `check_scm_equilibrium(result, T, U, Y, tol)` | Check all 10 SM equilibrium conditions. Returns (checks_dict, all_pass) |
| `check_plc_equilibrium(result, T, U1, U2, L1, Y, tol)` | Check all 11 PLC equilibrium conditions |

## Project structure

```
scm-equilibrium/
  scm/                     Python package (core solvers + verification)
  main.py                  Quick-start: define economy in code
  cli.py                   CLI: load economy from JSON
  examples/                Example economy JSON files
  tests/                   pytest test suite (17 tests)
  matlab_reference/        Original MATLAB codes for provenance
  docs/                    Theory audit + project history
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

All 17 tests verify against the paper's documented numerical examples, analytical solutions with known closed-form answers, and synthetic economies checked exhaustively against all equilibrium conditions.

## Dependencies

- Python >= 3.9
- NumPy >= 1.21
- CVXPY >= 1.3 (with CLARABEL solver, included by default)

## References

Deshpande, A. & Sohoni, M. (2021). *A Simple Closed Economy Model.* arXiv:2109.09248.
