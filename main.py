#!/usr/bin/env python3
"""
main.py  –  Quick-start entry point for the SCM equilibrium solver

Edit the economy parameters below, then run:
    python main.py

The script will:
  1. Run tatonnement to find an SM equilibrium
  2. Print the equilibrium (prices, production, wages, allocations)
  3. Run all equilibrium verification conditions
"""

import numpy as np
from scm import (
    compute_equilibrium, print_equilibrium, check_scm_equilibrium,
    compute_equilibrium_plc, print_equilibrium_plc, check_plc_equilibrium,
)


# ═══════════════════════════════════════════════════════════════════════
#  DEFINE YOUR ECONOMY HERE
# ═══════════════════════════════════════════════════════════════════════

# Technology matrix T  (m x n)
#   T[i,j] = units of labour class i needed to produce one unit of good j
#   m = number of labour classes, n = number of goods
T = np.array([
    [1.0, 0.0],
    [1.0, 1.0],
])

# Utility matrix U  (m x n)
#   U[i,j] = utility per unit of good j for class i
#   (For PLC economies, use U1, U2, L1 instead — see below)
U = np.array([
    [1.0, 0.8],
    [0.8, 1.0],
])

# Labour endowments Y  (m,)
#   Y[i] = total labour supply of class i
Y = np.array([2.0, 4.0])

# Starting price vector p_init  (n,)
#   Any positive vector — tatonnement will iterate from here
p_init = np.array([1.0, 1.0])


# ═══════════════════════════════════════════════════════════════════════
#  SOLVER SETTINGS
# ═══════════════════════════════════════════════════════════════════════

MAX_ITER = 200      # maximum tatonnement iterations
TOL      = 1e-7     # convergence tolerance on max |p_new - p|


# ═══════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  SCM Equilibrium Solver")
    print("=" * 60)
    print(f"\n  Economy: {T.shape[0]} labour classes, {T.shape[1]} goods")
    print(f"  Utility type: linear")
    print(f"  Max iterations: {MAX_ITER},  tolerance: {TOL}")

    # --- Solve ---
    result = compute_equilibrium(T, U, Y, p_init,
                                  max_iter=MAX_ITER, tol=TOL)

    # --- Print results ---
    print()
    print_equilibrium(result, label="Equilibrium result")

    # --- Verify all conditions ---
    print("\n  Equilibrium condition checks:")
    checks, all_pass = check_scm_equilibrium(
        result, T, U, Y, tol=1e-3, verbose=True
    )
    print(f"\n  {'ALL CONDITIONS PASS' if all_pass else 'SOME CONDITIONS FAILED'}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
#  PLC EXAMPLE  (uncomment to use piecewise-linear-concave utilities)
# ═══════════════════════════════════════════════════════════════════════
#
#  T = np.array([
#      [1.0, 0.0],
#      [1.0, 1.0],
#  ])
#
#  # Segment-1 utilities (higher marginal utility)
#  U1 = np.array([
#      [1.0, 0.8],
#      [0.8, 1.0],
#  ])
#
#  # Segment-2 utilities (lower marginal utility, U2 <= U1)
#  U2 = np.array([
#      [0.5, 0.4],
#      [0.4, 0.5],
#  ])
#
#  # Segment-1 capacity limits (units)
#  L1 = np.array([
#      [1.0, 1.0],
#      [1.0, 1.0],
#  ])
#
#  Y      = np.array([2.0, 4.0])
#  p_init = np.array([1.0, 1.0])
#
#  result = compute_equilibrium_plc(T, U1, U2, L1, Y, p_init,
#                                    max_iter=300, tol=1e-7)
#  print_equilibrium_plc(result, label="PLC Equilibrium")
#  checks, ok = check_plc_equilibrium(result, T, U1, U2, L1, Y, tol=1e-4)
#  print(f"  {'ALL PASS' if ok else 'FAILURES DETECTED'}")
