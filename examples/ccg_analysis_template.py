"""
ccg_analysis_template.py — Generic CCG analysis template

Given ANY economy (T, U_true, Y), this script runs a complete CCG analysis:
  1. Baseline equilibrium (honest play)
  2. Zone map over a 2D parameter grid
  3. Payoff trajectories (1D sweeps)
  4. Fisher forest visualization at selected points
  5. Gradient field analysis
  6. Nash equilibrium search

Modify the ECONOMY DEFINITION section below for your economy.

Usage:
    python examples/ccg_analysis_template.py
"""

import numpy as np
import os
import sys

# Add parent dir to path if running from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scm import solve_robust
from scm.ccg import (ccg_payoff, ccg_payoff_detailed, ccg_sweep,
                     ccg_gradient, ccg_zone_map, extract_forest, zone_label)
from scm.nash import nash_iteration, find_nash_candidates

# Optional: visualization (requires matplotlib)
try:
    from scm.visualize import (
        plot_zone_map, plot_zone_map_with_payoff,
        plot_payoff_trajectory, plot_wage_trajectory,
        plot_allocation_pattern, plot_forest_diagram,
        plot_gradient_field,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("matplotlib not found — skipping plots")


# ======================================================================
# ECONOMY DEFINITION  (modify this section for your economy)
# ======================================================================

# Economy name (used in plot titles and output folder)
ECONOMY_NAME = "example_2x2"

# Technology matrix: T[i,j] = units of labour class i per unit of good j
T = np.array([
    [1.0, 0.0],
    [1.0, 1.0],
])

# True utility matrix: U_true[i,j] = true utility per unit of good j for class i
U_true = np.array([
    [1.0, 0.8],
    [0.8, 1.0],
])

# Labour endowments
Y = np.array([2.0, 4.0])

# Initial price guess
p_init = np.array([1.0, 1.0])

# Labels for display
class_labels = ['Skilled', 'Unskilled']
good_labels = ['Good 0', 'Good 1']

# Strategy parameterization:
# Define how U_expressed depends on parameters.
# Here: each class can scale their utility for each good.
# U_expressed[0,:] = [U_true[0,0], alpha * U_true[0,1]]
# U_expressed[1,:] = [beta * U_true[1,0], U_true[1,1]]
# alpha = class 0's expressed preference for good 1 (relative to true)
# beta  = class 1's expressed preference for good 0 (relative to true)

def make_U_expressed(params):
    """Construct U_expressed from parameter dict."""
    alpha = params.get('alpha', 1.0)
    beta = params.get('beta', 1.0)
    return np.array([
        [U_true[0, 0], alpha * U_true[0, 1]],
        [beta * U_true[1, 0], U_true[1, 1]],
    ])

# Parameter ranges for sweeps
ALPHA_RANGE = np.linspace(0.2, 2.5, 25)
BETA_RANGE = np.linspace(0.2, 2.5, 25)

# ======================================================================
# OUTPUT SETUP
# ======================================================================

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'docs', 'figures', ECONOMY_NAME)
os.makedirs(OUTDIR, exist_ok=True)


# ======================================================================
# 1. BASELINE EQUILIBRIUM
# ======================================================================

def run_baseline():
    print("=" * 60)
    print("1. BASELINE EQUILIBRIUM (honest play)")
    print("=" * 60)

    payoffs, payoff_mat, wages, prices, quantities, X, zone = \
        ccg_payoff_detailed(T, U_true, U_true, Y, p_init)

    print(f"  Prices:       {prices}")
    print(f"  Production:   {quantities}")
    print(f"  Wages:        {wages}")
    print(f"  Payoffs:      {payoffs}")
    print(f"  Zone:         I={zone['I']}, J={zone['J']}")
    print(f"  Forest:       {zone['F']}")
    print(f"  Status:       {zone['status']}")
    print(f"  Allocations:\n{X}")

    if HAS_VIZ:
        I = zone['I']
        J = zone['J']
        plot_allocation_pattern(X, class_labels, good_labels,
                                title=f'{ECONOMY_NAME}: Honest Allocation',
                                output_file=os.path.join(OUTDIR, 'baseline_allocation.png'))
        if len(I) > 0 and len(J) > 0:
            plot_forest_diagram(X, I, J, class_labels, good_labels,
                                title=f'{ECONOMY_NAME}: Honest Forest',
                                output_file=os.path.join(OUTDIR, 'baseline_forest.png'))

    return payoffs, X, zone


# ======================================================================
# 2. ZONE MAP
# ======================================================================

def run_zone_map():
    print("\n" + "=" * 60)
    print("2. ZONE MAP (alpha × beta)")
    print("=" * 60)

    zone_grid, payoff_grid, wage_grid, forest_grid = ccg_zone_map(
        T, U_true, Y, p_init, make_U_expressed,
        ALPHA_RANGE, BETA_RANGE,
        param1_name='alpha', param2_name='beta',
        verbose=True
    )

    # Count unique zones
    unique = sorted(set(zone_grid.ravel()))
    print(f"\n  Found {len(unique)} distinct zones:")
    for z in unique:
        count = np.sum(zone_grid == z)
        print(f"    {z}  ({count} grid points)")

    if HAS_VIZ:
        plot_zone_map(zone_grid, ALPHA_RANGE, BETA_RANGE,
                      param1_name='alpha (class 0 expression)',
                      param2_name='beta (class 1 expression)',
                      title=f'{ECONOMY_NAME}: Zone Structure',
                      output_file=os.path.join(OUTDIR, 'zone_map.png'))

        for player in range(T.shape[0]):
            plot_zone_map_with_payoff(
                zone_grid, payoff_grid, ALPHA_RANGE, BETA_RANGE,
                player=player,
                param1_name='alpha', param2_name='beta',
                title=f'{ECONOMY_NAME}: Zones + {class_labels[player]} Payoff',
                output_file=os.path.join(OUTDIR, f'zone_payoff_class{player}.png'))

    return zone_grid, payoff_grid, wage_grid, forest_grid


# ======================================================================
# 3. PAYOFF TRAJECTORIES (1D sweeps)
# ======================================================================

def run_trajectories():
    print("\n" + "=" * 60)
    print("3. PAYOFF TRAJECTORIES")
    print("=" * 60)

    # Sweep beta at fixed alpha=1
    grid_beta = [{'alpha': 1.0, 'beta': b} for b in BETA_RANGE]
    results_beta = ccg_sweep(T, U_true, Y, p_init,
                              make_U_expressed, grid_beta, verbose=True)

    payoff_arr = np.array([r['payoffs'] for r in results_beta])
    wage_arr = np.array([r['wages'] for r in results_beta])
    zone_labels = [r['zone_label'] for r in results_beta]

    if HAS_VIZ:
        plot_payoff_trajectory(BETA_RANGE, payoff_arr, class_labels,
                                param_name='beta (class 1 expression)',
                                title=f'{ECONOMY_NAME}: Payoffs vs beta (alpha=1)',
                                zone_labels=zone_labels,
                                output_file=os.path.join(OUTDIR, 'payoff_vs_beta.png'))

        plot_wage_trajectory(BETA_RANGE, wage_arr, class_labels,
                              param_name='beta (class 1 expression)',
                              title=f'{ECONOMY_NAME}: Wages vs beta (alpha=1)',
                              zone_labels=zone_labels,
                              output_file=os.path.join(OUTDIR, 'wage_vs_beta.png'))

    # Sweep alpha at fixed beta=1
    grid_alpha = [{'alpha': a, 'beta': 1.0} for a in ALPHA_RANGE]
    results_alpha = ccg_sweep(T, U_true, Y, p_init,
                               make_U_expressed, grid_alpha)

    payoff_arr2 = np.array([r['payoffs'] for r in results_alpha])
    zone_labels2 = [r['zone_label'] for r in results_alpha]

    if HAS_VIZ:
        plot_payoff_trajectory(ALPHA_RANGE, payoff_arr2, class_labels,
                                param_name='alpha (class 0 expression)',
                                title=f'{ECONOMY_NAME}: Payoffs vs alpha (beta=1)',
                                zone_labels=zone_labels2,
                                output_file=os.path.join(OUTDIR, 'payoff_vs_alpha.png'))

    return results_beta, results_alpha


# ======================================================================
# 4. GRADIENT ANALYSIS
# ======================================================================

def run_gradient():
    print("\n" + "=" * 60)
    print("4. GRADIENT ANALYSIS")
    print("=" * 60)

    # Gradient at honest play
    J = ccg_gradient(T, U_true, U_true, Y, p_init)
    m, n = T.shape
    print(f"  Jacobian shape: {J.shape}")

    for player in range(m):
        print(f"\n  {class_labels[player]}'s payoff gradient:")
        grad = J[player]
        for k in range(m):
            for l in range(n):
                print(f"    ∂payoff/∂U_expr[{k},{l}] = {grad[k,l]:+.6f}")

        # Own-row gradient
        own_grad = grad[player]
        mag = np.linalg.norm(own_grad)
        print(f"    Own-row gradient magnitude: {mag:.6f}")
        if mag > 1e-6:
            print(f"    Direction: {own_grad / mag}")

    return J


# ======================================================================
# 5. NASH EQUILIBRIUM SEARCH
# ======================================================================

def run_nash():
    print("\n" + "=" * 60)
    print("5. NASH EQUILIBRIUM SEARCH")
    print("=" * 60)

    candidates = find_nash_candidates(
        T, U_true, Y, p_init,
        n_restarts=5, max_iter=40, lr=0.1, tol=1e-4,
        verbose=True
    )

    print(f"\n  Found {len(candidates)} candidates:")
    for i, c in enumerate(candidates):
        print(f"\n  Candidate {i+1}:")
        print(f"    Converged: {c['converged']}")
        print(f"    Gap:       {c['convergence_gap']:.6f}")
        print(f"    Payoffs:   {c['payoffs']}")
        print(f"    U_expressed:\n{c['U_expressed']}")

    return candidates


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    print(f"\nCCG Analysis: {ECONOMY_NAME}")
    print(f"Economy: {T.shape[0]} classes × {T.shape[1]} goods")
    print(f"Output: {OUTDIR}\n")

    baseline_payoffs, baseline_X, baseline_zone = run_baseline()
    zone_grid, payoff_grid, wage_grid, forest_grid = run_zone_map()
    results_beta, results_alpha = run_trajectories()
    J = run_gradient()
    candidates = run_nash()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    if HAS_VIZ:
        print(f"Plots saved to: {OUTDIR}")
    else:
        print("Install matplotlib for visualization: pip install matplotlib")
