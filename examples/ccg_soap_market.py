#!/usr/bin/env python3
"""
Soap Market CCG Analysis
========================

Full worked example from the paper's 2×2 soap market (Section 4.3/6.1).
Replicates MATLAB FeigningU.m and extends with Fisher forest tracking,
zone mapping, gradient analysis, and Nash equilibrium search.

Economy:
  T = [[0.2501, 0], [0.25, 1]]   — Class 0 specialises in good 0
  U_true = [[1, 1], [1, 1]]       — Both classes value both goods equally
  Y = [2, 4]                      — Class 1 has more labour

CCG parameterisation (Section 6 / zone analysis):
  U_expressed = [[alpha, 1], [beta, 1]]
  alpha = class 0's expressed preference for good 0 (relative to good 1)
  beta  = class 1's expressed preference for good 0 (relative to good 1)

  Note: The paper's Appendix A tables use a transposed convention
  U = [[1, beta], [alpha, 1]], but the zone analysis in Section 6
  explicitly uses U = [[alpha, 1], [beta, 1]] (see paper page 15).

Outputs saved to docs/figures/soap_*.png
"""

import sys
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scm.ccg import (
    ccg_payoff_detailed, ccg_sweep, ccg_gradient,
    ccg_zone_map, zone_label, extract_forest,
)
from scm.nash import best_response_direction, nash_iteration
from scm.visualize import (
    plot_zone_map, plot_zone_map_with_payoff,
    plot_payoff_trajectory, plot_wage_trajectory, plot_price_trajectory,
    plot_allocation_pattern, plot_forest_diagram,
)

# ──────────────────────────────────────────────────────────────────────
# Economy
# ──────────────────────────────────────────────────────────────────────
T = np.array([[0.2501, 0.0],
              [0.25,   1.0]])
U_TRUE = np.array([[1.0, 1.0],
                    [1.0, 1.0]])
Y = np.array([2.0, 4.0])
P_INIT = np.array([2.0, 3.0])

CLASS_LABELS = ['Class 0 (specialist)', 'Class 1 (generalist)']
GOOD_LABELS = ['Good 0', 'Good 1']

def U_func(params):
    """Expressed utility: [[alpha, 1], [beta, 1]] (Section 6 convention)."""
    return np.array([[params.get('alpha', 1.0), 1.0],
                     [params.get('beta', 1.0), 1.0]])

# Output directory
FIGDIR = Path(__file__).parent.parent / 'docs' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

def savepath(name):
    return str(FIGDIR / name)

# ──────────────────────────────────────────────────────────────────────
# 1. Zone Map over (alpha, beta)
# ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("1. ZONE MAP: (alpha, beta) in [0.3, 2.0]^2")
print("=" * 70)

alpha_grid = np.linspace(0.3, 2.0, 25)
beta_grid = np.linspace(0.3, 2.0, 25)

zone_grid, payoff_grid, wage_grid, forest_grid = ccg_zone_map(
    T, U_TRUE, Y, P_INIT, U_func, alpha_grid, beta_grid,
    param1_name='alpha', param2_name='beta', verbose=True)

unique_zones = sorted(set(zone_grid.ravel()))
print(f"\nFound {len(unique_zones)} distinct zones:")
for z in unique_zones:
    count = np.sum(zone_grid == z)
    print(f"  {z}  ({count} grid points)")

plot_zone_map(zone_grid, alpha_grid, beta_grid,
              param1_name=r'$\alpha$ (class 0 pref for good 0)',
              param2_name=r'$\beta$ (class 1 pref for good 0)',
              title='Soap Market: Zone Structure (I, J, F)',
              output_file=savepath('soap_zone_map.png'))
print(f"Saved: {savepath('soap_zone_map.png')}")

# Side-by-side with payoff
for player in range(2):
    plot_zone_map_with_payoff(
        zone_grid, payoff_grid, alpha_grid, beta_grid, player=player,
        param1_name=r'$\alpha$', param2_name=r'$\beta$',
        title=f'Zone Map + {CLASS_LABELS[player]} Payoff',
        output_file=savepath(f'soap_zone_payoff_class{player}.png'))
    print(f"Saved: {savepath(f'soap_zone_payoff_class{player}.png')}")


# ──────────────────────────────────────────────────────────────────────
# 2. Payoff / Wage / Price Trajectories (FeigningU.m replication)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. PAYOFF SWEEPS (replicating FeigningU.m)")
print("=" * 70)

# Sweep beta with fixed alpha values (matches MATLAB R=[0.5,0.75,1.001,1.5,1.7])
for alpha_val in [0.5, 0.75, 1.0, 1.5]:
    beta_sweep = np.linspace(0.3, 1.9, 20)
    param_grid = [{'alpha': alpha_val, 'beta': b} for b in beta_sweep]

    results = ccg_sweep(T, U_TRUE, Y, P_INIT, U_func, param_grid, verbose=False)

    payoffs = np.array([r['payoffs'] for r in results])
    wages = np.array([r['wages'] for r in results])
    prices = np.array([r['prices'] for r in results])
    zlabels = [r['zone_label'] for r in results]

    suffix = f'a{alpha_val:.1f}'.replace('.', 'p')

    plot_payoff_trajectory(
        beta_sweep, payoffs, class_labels=CLASS_LABELS,
        param_name=r'$\beta$', zone_labels=zlabels,
        title=f'Payoff (alpha={alpha_val:.2f})',
        output_file=savepath(f'soap_payoff_{suffix}.png'))

    plot_wage_trajectory(
        beta_sweep, wages, class_labels=CLASS_LABELS,
        param_name=r'$\beta$', zone_labels=zlabels,
        title=f'Wages (alpha={alpha_val:.2f})',
        output_file=savepath(f'soap_wages_{suffix}.png'))

    plot_price_trajectory(
        beta_sweep, prices, good_labels=GOOD_LABELS,
        param_name=r'$\beta$', zone_labels=zlabels,
        title=f'Prices (alpha={alpha_val:.2f})',
        output_file=savepath(f'soap_prices_{suffix}.png'))

    print(f"  alpha={alpha_val:.2f}: saved payoff/wage/price plots")


# ──────────────────────────────────────────────────────────────────────
# 3. Forest Diagrams at Key Points
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. FISHER FOREST DIAGRAMS")
print("=" * 70)

key_points = [
    {'alpha': 0.5, 'beta': 0.5},
    {'alpha': 1.0, 'beta': 1.0},
    {'alpha': 1.5, 'beta': 0.7},
]

for params in key_points:
    U_expr = U_func(params)
    payoffs, _, wages, prices, q, X, zone = \
        ccg_payoff_detailed(T, U_TRUE, U_expr, Y, P_INIT)

    zlabel = zone_label(zone['I'], zone['J'], zone['F'])
    a, b = params['alpha'], params['beta']
    print(f"\n  (alpha={a}, beta={b}):")
    print(f"    Zone: {zlabel}")
    print(f"    Payoffs: {payoffs}")
    print(f"    Wages: {wages}")
    print(f"    Prices: {prices}")
    print(f"    Allocations:\n{X}")

    suffix = f'a{a:.1f}_b{b:.1f}'.replace('.', 'p')

    plot_allocation_pattern(
        X, class_labels=CLASS_LABELS, good_labels=GOOD_LABELS,
        title=f'Allocation (alpha={a}, beta={b})',
        output_file=savepath(f'soap_alloc_{suffix}.png'))

    plot_forest_diagram(
        X, zone['I'], zone['J'],
        class_labels=CLASS_LABELS, good_labels=GOOD_LABELS,
        title=f'Forest (alpha={a}, beta={b}) — {zlabel}',
        output_file=savepath(f'soap_forest_{suffix}.png'))


# ──────────────────────────────────────────────────────────────────────
# 4. Gradient Analysis
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. GRADIENT ANALYSIS")
print("=" * 70)

test_points = [
    {'alpha': 0.7, 'beta': 0.7},
    {'alpha': 1.0, 'beta': 1.0},
    {'alpha': 1.3, 'beta': 0.8},
]

for params in test_points:
    U_expr = U_func(params)
    a, b = params['alpha'], params['beta']
    print(f"\n  (alpha={a}, beta={b}):")

    for player in range(2):
        direction, magnitude = best_response_direction(
            T, U_TRUE, U_expr, Y, P_INIT, player)
        print(f"    Player {player}: direction={direction}, magnitude={magnitude:.6f}")
        if magnitude < 1e-4:
            print(f"      → At local optimum (within zone)")
        else:
            print(f"      → Can improve by shifting U[{player},:] in direction {direction}")


# ──────────────────────────────────────────────────────────────────────
# 5. Nash Equilibrium Search
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. NASH EQUILIBRIUM SEARCH")
print("=" * 70)

# Start from U_true (honest play)
result = nash_iteration(T, U_TRUE, U_TRUE.copy(), Y, P_INIT,
                         max_iter=30, lr=0.1, tol=1e-4, verbose=True)

print(f"\nConverged: {result['converged']}")
print(f"Iterations: {result['n_iter']}")
print(f"Final payoffs: {result['payoffs'][-1]}")
print(f"Final U_expressed:\n{result['profiles'][-1]}")

# Multi-start search
print("\nMulti-start Nash search (3 restarts)...")
candidates = []
from scm.nash import find_nash_candidates
candidates = find_nash_candidates(T, U_TRUE, Y, P_INIT, n_restarts=3,
                                   max_iter=30, lr=0.1, tol=1e-4)

print(f"\nFound {len(candidates)} candidates:")
for i, c in enumerate(candidates):
    print(f"\n  Candidate {i + 1}:")
    print(f"    Convergence gap: {c['convergence_gap']:.2e}")
    print(f"    Payoffs: {c['payoffs']}")
    print(f"    U_expressed:\n{c['U_expressed']}")
    print(f"    Converged: {c['converged']}")


# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"ALL FIGURES SAVED TO: {FIGDIR}")
print("=" * 70)
