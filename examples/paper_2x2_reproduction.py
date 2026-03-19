#!/usr/bin/env python3
"""
paper_2x2_reproduction.py — Full reproduction of the paper's 2×2 economy analysis
================================================================================

Reproduces the complete analysis from Deshpande & Sohoni (arXiv:2109.09248)
for the 2×2 soap market economy, using ONLY the scm library's general solvers
and CCG functions.

Paper's 2×2 Economy (Section 6):
  T      = [[0.2501, 0], [0.25, 1]]    (technology)
  U_true = [[1, 1], [1, 1]]            (true utilities — both classes equal)
  Y      = [2, 4]                      (labour endowments)

  Class 0 ("Skilled"): can only produce good 0 (specialist)
  Class 1 ("Unskilled"): can produce both goods (generalist)

CCG Parameterization (Section 6 convention):
  U_expressed = [[alpha, 1], [beta, 1]]
  alpha = Skilled's expressed preference for good 0 (vs good 1)
  beta  = Unskilled's expressed preference for good 0 (vs good 1)
  alpha = beta = 1 → honest play (U_expressed = U_true)

What this script reproduces:
  1. Baseline equilibrium (honest play)
  2. Zone decomposition map (I, J, F) across (alpha, beta) space
  3. Fisher forest extraction at key points
  4. Payoff surfaces for each class overlaid on zones
  5. 1D payoff/wage/price sweeps (beta sweep at alpha=1, alpha sweep at beta=1)
  6. Gradient fields showing each player's incentive direction
  7. Nash equilibrium search
  8. Verification against paper's analytical zone boundaries

All figures saved to output directory. Console prints verification results.

Usage:
  python examples/paper_2x2_reproduction.py
  python examples/paper_2x2_reproduction.py --outdir my_figures
"""

import sys
import os
import argparse
import numpy as np

# Ensure the repo root is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scm import (
    solve_robust, check_scm_equilibrium,
    ccg_payoff, ccg_payoff_detailed, ccg_sweep, ccg_gradient,
    ccg_zone_map, extract_forest, zone_label,
    describe_forest, classify_zone,
    best_response_direction, nash_iteration, find_nash_candidates,
)
from scm.visualize import (
    plot_zone_map, plot_zone_map_with_payoff,
    plot_payoff_trajectory, plot_wage_trajectory, plot_price_trajectory,
    plot_allocation_pattern, plot_forest_diagram, plot_gradient_field,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ======================================================================
# Economy definition (from the paper)
# ======================================================================

T = np.array([[0.2501, 0.0],
              [0.25,   1.0]])

U_TRUE = np.array([[1.0, 1.0],
                    [1.0, 1.0]])

Y = np.array([2.0, 4.0])

P_INIT = np.array([1.0, 1.0])

CLASS_LABELS = ['Skilled (L0)', 'Unskilled (L1)']
GOOD_LABELS  = ['Soap (g0)', 'Haircuts (g1)']

# Paper-specific zone names (Section 6, arXiv:2109.09248).
# Maps structural labels from describe_forest() → paper's zone names.
# This is ONLY for display/verification in this 2×2 reproduction script.
PAPER_ZONE_MAP = {
    'C0:{g0}|C1:{g0,g1}':     'F1',   # C0 specialist(g0), C1 generalist
    'C0:{g0,g1}|C1:{g0}':     'F2',   # C0 generalist, C1 specialist(g0)
    'C0:{g1}|C1:{g0,g1}':     'F3',   # C0 specialist(g1), C1 generalist
    'C0:{g1}|C1:{g0}':        'F4',   # Both specialists, opposite goods
    'C0:{g0,g1}|C1:{g0,g1}':  'CY',   # Both generalists (α≈β diagonal)
}


def structural_zone_label(I, J, forest):
    """Get structural zone label (general, non-paper-specific).

    Returns the structural label from describe_forest() without any
    paper-specific mapping. For general m×n economies.
    """
    desc = describe_forest(I, J, forest, m=2)
    return desc['label']


def U_func(params):
    """Parameterization: U = [[alpha, 1], [beta, 1]].

    alpha: class 0's expressed preference for good 0 (vs good 1)
    beta: class 1's expressed preference for good 0 (vs good 1)
    """
    return np.array([[params.get('alpha', 1.0), 1.0],
                     [params.get('beta',  1.0), 1.0]])


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def savepath(outdir, name):
    return os.path.join(outdir, name)


# ======================================================================
# Analysis functions
# ======================================================================

def step1_baseline(outdir):
    """Step 1: Honest equilibrium (alpha=beta=1)."""
    separator("STEP 1: BASELINE EQUILIBRIUM (honest play)")

    result = solve_robust(T, U_TRUE, Y, P_INIT)

    print(f"  Status:      {result['status']}")
    print(f"  Prices:      {result['p']}")
    print(f"  Production:  {result['q']}")
    print(f"  Wages:       W = {result['W']}")
    print(f"  Allocations:")
    X = result['X']
    for i in range(2):
        print(f"    {CLASS_LABELS[i]}: {X[i]}")

    # Verify all 10 equilibrium conditions
    checks, all_pass = check_scm_equilibrium(result, T, U_TRUE, Y)
    print(f"\n  Equilibrium check: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    for name, passed in checks.items():
        print(f"    {'✓' if passed else '✗'} {name}")

    # Fisher forest at honest play
    I, J = result['I'], result['J']
    forest, bpb = extract_forest(U_TRUE, result['p'], X, I, J)
    print(f"\n  Fisher forest (honest): {forest}")
    print(f"  Active classes I={I}, Active goods J={J}")

    # Allocation bar chart
    fig1, _ = plot_allocation_pattern(
        X, CLASS_LABELS, GOOD_LABELS,
        title='Equilibrium Allocation (Honest Play)',
        output_file=savepath(outdir, '01_baseline_allocation.png'))
    plt.close(fig1)

    # Forest diagram
    fig2, _ = plot_forest_diagram(
        X, I, J, CLASS_LABELS, GOOD_LABELS,
        title='Fisher Forest at Honest Play',
        output_file=savepath(outdir, '01_baseline_forest.png'))
    plt.close(fig2)

    return result


def step2_zone_map(outdir):
    """Step 2: Zone decomposition across (alpha, beta) space."""
    separator("STEP 2: ZONE DECOMPOSITION MAP")

    alpha_grid = np.linspace(0.05, 3.0, 40)
    beta_grid  = np.linspace(0.05, 3.0, 40)

    print(f"  Grid: {len(alpha_grid)} × {len(beta_grid)} = {len(alpha_grid)*len(beta_grid)} points")
    print(f"  Computing zone map...")

    # NOTE: param1=beta (y-axis), param2=alpha (x-axis)
    # This arrangement makes beta the vertical axis and alpha the horizontal axis.
    zone_grid, payoff_grid, wage_grid, forest_grid = ccg_zone_map(
        T, U_TRUE, Y, P_INIT, U_func,
        beta_grid, alpha_grid,
        param1_name='beta', param2_name='alpha',
        verbose=True)

    # Count zones
    unique, counts = np.unique(zone_grid, return_counts=True)
    print(f"\n  Zones found ({len(unique)}):")
    for z, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        pct = 100 * c / zone_grid.size
        print(f"    {z:40s}  {c:5d} pts  ({pct:.1f}%)")

    # Classify into structural zone types using describe_forest
    # (without paper-specific naming)
    major_grid = np.empty_like(zone_grid, dtype=object)
    for i in range(zone_grid.shape[0]):
        for j in range(zone_grid.shape[1]):
            f = forest_grid[i, j]
            if f is None or zone_grid[i, j] == 'ERROR':
                major_grid[i, j] = 'ERR'
            else:
                n_classes = len(f)
                if n_classes == 1:
                    I_local = np.array([1])
                    J_local = np.array([0, 1])
                else:
                    I_local = np.array([0, 1])
                    all_goods = set()
                    for goods in f:
                        all_goods.update(goods)
                    J_local = np.array(sorted(all_goods))
                major_grid[i, j] = structural_zone_label(I_local, J_local, f)

    # Build mapping from structural label to short display name
    unique_structural = sorted(set(major_grid.flatten()))
    structural_to_short = {}
    zone_idx = 1
    for label in unique_structural:
        if label == 'ERR':
            structural_to_short[label] = 'ERR'
        else:
            structural_to_short[label] = f'Z{zone_idx}'
            zone_idx += 1

    # Print major zone counts using structural labels
    unique_major, counts_major = np.unique(major_grid, return_counts=True)
    print(f"\n  Major structural zones ({len(unique_major)}):")
    for z, c in sorted(zip(unique_major, counts_major), key=lambda x: -x[1]):
        pct = 100 * c / major_grid.size
        short = structural_to_short.get(z, z)
        print(f"    {short:6s} ({z})")
        print(f"      {c:5d} pts  ({pct:.1f}%)")

    # --- Detailed zone map (all fine-grained zones) ---
    fig1, _ = plot_zone_map(
        zone_grid, beta_grid, alpha_grid,
        param1_name=r'$\beta$ (Unskilled pref for soap)',
        param2_name=r'$\alpha$ (Skilled pref for soap)',
        title='Detailed Zone Decomposition (I, J, F) — 2×2 Soap Market',
        output_file=savepath(outdir, '02_zone_map_detailed.png'))
    plt.close(fig1)

    # --- Simplified zone map (computed structural zones) ---
    # Auto-assign colors to structural zones
    color_palette = [
        '#ff6666', '#6699ff', '#66ff66', '#ffcc66', '#cc66ff',
        '#ff99ff', '#99ffff', '#ffff99', '#ff99cc', '#99ccff'
    ]
    zone_colors = {'ERR': '#333333', 'OTH': '#cccccc'}
    for i, label in enumerate(unique_structural):
        if label != 'ERR':
            zone_colors[label] = color_palette[i % len(color_palette)]

    fig2, ax = plt.subplots(figsize=(10, 8))
    n1, n2 = major_grid.shape
    rgb = np.zeros((n1, n2, 3))
    for code, color in zone_colors.items():
        mask = (major_grid == code)
        if mask.any():
            r = int(color[1:3], 16) / 255
            g = int(color[3:5], 16) / 255
            b = int(color[5:7], 16) / 255
            rgb[mask] = [r, g, b]

    # NOTE: major_grid is indexed [beta_idx, alpha_idx]
    # imshow with origin='lower' and extent matching axes
    ax.imshow(rgb, origin='lower', aspect='auto',
              extent=[alpha_grid[0], alpha_grid[-1], beta_grid[0], beta_grid[-1]],
              interpolation='nearest')

    # Compute zone boundary contours from the data (NOT hardcoded from paper)
    zone_codes_sorted = sorted(set(major_grid.flatten()))
    zone_to_int = {z: i for i, z in enumerate(zone_codes_sorted)}
    int_grid = np.array([[zone_to_int[major_grid[i, j]]
                          for j in range(n2)] for i in range(n1)], dtype=float)
    ax.contour(alpha_grid, beta_grid, int_grid, levels=len(zone_codes_sorted)-1,
               colors='black', linewidths=1.5, alpha=0.7)

    # Place zone labels at computed centroids (data-driven, not hardcoded positions)
    for code in zone_codes_sorted:
        mask = (major_grid == code)
        if not mask.any() or code in ('ERR', 'OTH'):
            continue
        indices = np.argwhere(mask)
        bi_mean = indices[:, 0].mean()
        ai_mean = indices[:, 1].mean()
        a_centroid = np.interp(ai_mean, np.arange(n2), alpha_grid)
        b_centroid = np.interp(bi_mean, np.arange(n1), beta_grid)
        fontsize = 16 if mask.sum() > 50 else 12
        short_name = structural_to_short.get(code, code)
        ax.text(a_centroid, b_centroid, short_name, fontsize=fontsize, ha='center',
                fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlabel(r'$\alpha$ (Skilled expressed pref for soap)', fontsize=13)
    ax.set_ylabel(r'$\beta$ (Unskilled expressed pref for soap)', fontsize=13)
    ax.set_title('Computed Zone Decomposition — 2×2 Soap Market', fontsize=14)

    # Build legend dynamically from zones found
    from matplotlib.patches import Patch
    legend_elements = []
    for code in zone_codes_sorted:
        if code in ('ERR', 'OTH'):
            continue
        mask = (major_grid == code)
        if not mask.any():
            continue
        color = zone_colors.get(code, '#cccccc')
        short_name = structural_to_short.get(code, code)
        count = mask.sum()
        # Use structural label for description (truncate if verbose)
        desc = code if len(code) < 30 else code[:27] + '...'
        legend_elements.append(Patch(facecolor=color, label=f'{short_name}: {desc} ({count} pts)'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    fig2.tight_layout()
    fig2.savefig(savepath(outdir, '02_zone_map.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # Representative forest diagrams for each structural zone
    print("\n  Generating representative forest diagrams...")
    for zone_label in unique_structural:
        if zone_label == 'ERR':
            continue
        # Find a representative point
        mask = (major_grid == zone_label)
        if not mask.any():
            continue
        # Pick center-ish point
        indices = np.argwhere(mask)
        mid = indices[len(indices)//2]
        bi, ai = mid
        a_val = alpha_grid[ai]
        b_val = beta_grid[bi]

        U_expr = U_func({'alpha': a_val, 'beta': b_val})
        try:
            payoffs, payoff_mat, wages, prices, quantities, X, zone = \
                ccg_payoff_detailed(T, U_TRUE, U_expr, Y, P_INIT)

            desc = describe_forest(zone['I'], zone['J'], zone['F'], m=2)
            short_name = structural_to_short[zone_label]
            print(f"    {short_name} ({zone_label}) at (α={a_val:.2f}, β={b_val:.2f})")

            fig, _ = plot_forest_diagram(
                X, zone['I'], zone['J'], CLASS_LABELS, GOOD_LABELS,
                title=f'Forest in Zone {short_name} (α={a_val:.2f}, β={b_val:.2f})',
                output_file=savepath(outdir, f'02_forest_{short_name.replace(" ", "_").lower()}.png'))
            plt.close(fig)
        except Exception as e:
            short_name = structural_to_short[zone_label]
            print(f"    {short_name}: error — {e}")

    # Zone + payoff for each class
    for player in range(2):
        fig, _ = plot_zone_map_with_payoff(
            zone_grid, payoff_grid, beta_grid, alpha_grid,
            player=player,
            param1_name=r'$\beta$', param2_name=r'$\alpha$',
            title=f'{CLASS_LABELS[player]}: Zone Structure + Payoff',
            output_file=savepath(outdir, f'02_zone_payoff_class{player}.png'))
        plt.close(fig)

    return zone_grid, payoff_grid, wage_grid, forest_grid, major_grid, alpha_grid, beta_grid


def step3_forests(outdir):
    """Step 3: Fisher forests at key strategic points."""
    separator("STEP 3: FISHER FORESTS AT KEY POINTS")

    key_points = [
        {'alpha': 1.0,  'beta': 1.0,  'label': 'honest',              'desc': 'Truth-telling (α=β=1)'},
        {'alpha': 0.3,  'beta': 1.0,  'label': 'L0_understates_soap', 'desc': 'Skilled understates soap pref (α=0.3)'},
        {'alpha': 2.0,  'beta': 1.0,  'label': 'L0_overstates_soap',  'desc': 'Skilled overstates soap pref (α=2.0)'},
        {'alpha': 1.0,  'beta': 0.3,  'label': 'L1_understates_soap', 'desc': 'Unskilled understates soap pref (β=0.3)'},
        {'alpha': 1.0,  'beta': 2.0,  'label': 'L1_overstates_soap',  'desc': 'Unskilled overstates soap pref (β=2.0)'},
    ]

    for pt in key_points:
        U_expr = U_func(pt)
        payoffs, payoff_mat, wages, prices, quantities, X, zone = \
            ccg_payoff_detailed(T, U_TRUE, U_expr, Y, P_INIT)

        forest_str = zone_label(zone['I'], zone['J'], zone['F'])
        print(f"\n  {pt['desc']}")
        print(f"    U_expressed = {U_expr.tolist()}")
        print(f"    Zone: {forest_str}")
        print(f"    Payoffs:  Skilled={payoffs[0]:.4f}, Unskilled={payoffs[1]:.4f}")
        print(f"    Wages:    Skilled={wages[0]:.4f}, Unskilled={wages[1]:.4f}")
        print(f"    Prices:   soap={prices[0]:.4f}, haircuts={prices[1]:.4f}")
        print(f"    Allocations:")
        for i in range(2):
            print(f"      {CLASS_LABELS[i]}: soap={X[i,0]:.4f}, haircuts={X[i,1]:.4f}")

        # Forest diagram with structural label
        desc = describe_forest(zone['I'], zone['J'], zone['F'], m=2)
        forest_label = desc['label']

        fig, _ = plot_forest_diagram(
            X, zone['I'], zone['J'], CLASS_LABELS, GOOD_LABELS,
            title=f"Forest: {pt['desc']}\n{forest_label}",
            output_file=savepath(outdir, f"03_forest_{pt['label']}.png"))
        plt.close(fig)

        # Allocation pattern
        fig, _ = plot_allocation_pattern(
            X, CLASS_LABELS, GOOD_LABELS,
            title=f"Allocation: {pt['desc']}",
            output_file=savepath(outdir, f"03_alloc_{pt['label']}.png"))
        plt.close(fig)


def step4_sweeps(outdir):
    """Step 4: 1D payoff/wage/price sweeps."""
    separator("STEP 4: PAYOFF, WAGE, AND PRICE SWEEPS")

    sweep_vals = np.linspace(0.05, 3.0, 80)

    # --- Beta sweep (alpha fixed at 1) ---
    print("  Beta sweep (alpha=1, beta varies)...")
    beta_params = [{'alpha': 1.0, 'beta': b} for b in sweep_vals]
    beta_results = ccg_sweep(T, U_TRUE, Y, P_INIT, U_func, beta_params)

    payoff_beta = np.array([r['payoffs'] for r in beta_results])
    wage_beta   = np.array([r['wages']   for r in beta_results])
    price_beta  = np.array([r['prices']  for r in beta_results])
    zone_beta   = [r['zone_label'] for r in beta_results]

    fig1, _ = plot_payoff_trajectory(
        sweep_vals, payoff_beta, CLASS_LABELS,
        param_name=r'$\beta$ (Unskilled pref for soap)',
        title='Payoffs vs β (α = 1)',
        zone_labels=zone_beta,
        output_file=savepath(outdir, '04_payoff_vs_beta.png'))
    plt.close(fig1)

    fig2, _ = plot_wage_trajectory(
        sweep_vals, wage_beta, CLASS_LABELS,
        param_name=r'$\beta$ (Unskilled pref for soap)',
        title='Wages vs β (α = 1)',
        zone_labels=zone_beta,
        output_file=savepath(outdir, '04_wage_vs_beta.png'))
    plt.close(fig2)

    fig3, _ = plot_price_trajectory(
        sweep_vals, price_beta, GOOD_LABELS,
        param_name=r'$\beta$ (Unskilled pref for soap)',
        title='Prices vs β (α = 1)',
        zone_labels=zone_beta,
        output_file=savepath(outdir, '04_price_vs_beta.png'))
    plt.close(fig3)

    # --- Alpha sweep (beta fixed at 1) ---
    print("  Alpha sweep (beta=1, alpha varies)...")
    alpha_params = [{'alpha': a, 'beta': 1.0} for a in sweep_vals]
    alpha_results = ccg_sweep(T, U_TRUE, Y, P_INIT, U_func, alpha_params)

    payoff_alpha = np.array([r['payoffs'] for r in alpha_results])
    zone_alpha   = [r['zone_label'] for r in alpha_results]

    fig4, _ = plot_payoff_trajectory(
        sweep_vals, payoff_alpha, CLASS_LABELS,
        param_name=r'$\alpha$ (Skilled pref for soap)',
        title='Payoffs vs α (β = 1)',
        zone_labels=zone_alpha,
        output_file=savepath(outdir, '04_payoff_vs_alpha.png'))
    plt.close(fig4)

    # Print key values
    honest_idx = np.argmin(np.abs(sweep_vals - 1.0))
    print(f"\n  At honest play (α=1, β=1):")
    print(f"    Payoffs: {payoff_beta[honest_idx]}")
    print(f"    Wages:   {wage_beta[honest_idx]}")
    print(f"    Prices:  {price_beta[honest_idx]}")

    return beta_results, alpha_results


def step5_gradients(outdir, zone_grid, beta_grid, alpha_grid):
    """Step 5: Gradient fields — which direction should each player deviate?"""
    separator("STEP 5: GRADIENT FIELDS (player incentives)")

    # Use a coarser grid for gradient computation (expensive)
    g_beta  = np.linspace(0.2, 2.8, 15)
    g_alpha = np.linspace(0.2, 2.8, 15)

    for player in range(2):
        print(f"  Computing gradient field for {CLASS_LABELS[player]}...")
        grad_grid = np.zeros((len(g_beta), len(g_alpha), 2))

        for i, b in enumerate(g_beta):
            for j, a in enumerate(g_alpha):
                U_expr = U_func({'alpha': a, 'beta': b})
                try:
                    grad = ccg_gradient(T, U_TRUE, U_expr, Y, P_INIT, player=player)
                    # grad[player, :] = d(payoff)/d(U_expressed[player,:])
                    # For Section 6 param: alpha controls U[0,0], beta controls U[1,0]
                    # Player 0's relevant gradient component = grad[0, 0] (d/d_alpha)
                    # Player 1's relevant gradient component = grad[1, 0] (d/d_beta)
                    # plot_gradient_field expects: [0]=d/d_param1(beta/vertical), [1]=d/d_param2(alpha/horizontal)
                    if player == 0:
                        grad_grid[i, j, 0] = 0.0          # player 0 doesn't control beta (vertical)
                        grad_grid[i, j, 1] = grad[0, 0]   # d_payoff/d_alpha (horizontal)
                    else:
                        grad_grid[i, j, 0] = grad[1, 0]   # d_payoff/d_beta (vertical)
                        grad_grid[i, j, 1] = 0.0          # player 1 doesn't control alpha (horizontal)
                except Exception:
                    pass

        # Interpolate zone_grid onto the coarser gradient grid for background
        from scipy.interpolate import RegularGridInterpolator
        unique_zones = sorted(set(zone_grid.ravel()))
        zone_to_code = {z: i for i, z in enumerate(unique_zones)}
        code_grid = np.array([[zone_to_code[zone_grid[ii, jj]]
                               for jj in range(zone_grid.shape[1])]
                              for ii in range(zone_grid.shape[0])], dtype=float)
        interp = RegularGridInterpolator(
            (beta_grid, alpha_grid), code_grid, method='nearest',
            bounds_error=False, fill_value=0)
        P1, P2 = np.meshgrid(g_beta, g_alpha, indexing='ij')
        points = np.stack([P1.ravel(), P2.ravel()], axis=-1)
        coarse_code = interp(points).reshape(len(g_beta), len(g_alpha))
        code_to_zone = {v: k for k, v in zone_to_code.items()}
        coarse_zone = np.empty_like(coarse_code, dtype=object)
        for ii in range(len(g_beta)):
            for jj in range(len(g_alpha)):
                coarse_zone[ii, jj] = code_to_zone.get(int(coarse_code[ii, jj]), 'UNK')

        fig, _ = plot_gradient_field(
            grad_grid, g_beta, g_alpha,
            param1_name=r'$\beta$', param2_name=r'$\alpha$',
            zone_grid=coarse_zone,
            title=f'Incentive Gradient: {CLASS_LABELS[player]}',
            output_file=savepath(outdir, f'05_gradient_{CLASS_LABELS[player].split()[0].lower()}.png'))
        plt.close(fig)


def step6_nash(outdir):
    """Step 6: Nash equilibrium search."""
    separator("STEP 6: NASH EQUILIBRIUM SEARCH")

    print("  Running multi-start Nash search (5 restarts)...")
    candidates = find_nash_candidates(
        T, U_TRUE, Y, P_INIT,
        n_restarts=5, max_iter=50, lr=0.1, tol=1e-4, verbose=True)

    print(f"\n  Top Nash candidates:")
    for k, c in enumerate(candidates[:3]):
        print(f"\n  Candidate {k+1}:")
        print(f"    U_expressed =\n{c['U_expressed']}")
        print(f"    Payoffs:    {c['payoffs']}")
        print(f"    Conv. gap:  {c['convergence_gap']:.6f}")
        print(f"    Converged:  {c['converged']}")

    # Plot convergence for the best candidate run from truth
    print("\n  Running detailed Nash iteration from honest play...")
    result = nash_iteration(
        T, U_TRUE, U_TRUE.copy(), Y, P_INIT,
        max_iter=50, lr=0.1, tol=1e-4, verbose=True)

    if result['payoffs'].shape[0] > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        iters = np.arange(result['payoffs'].shape[0])
        for i in range(2):
            ax.plot(iters, result['payoffs'][:, i], 'o-', label=CLASS_LABELS[i], linewidth=2)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Payoff', fontsize=12)
        ax.set_title('Nash Convergence from Honest Play', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.savefig(savepath(outdir, '06_nash_convergence.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Forest at Nash equilibrium
    best = candidates[0] if candidates else None
    if best is not None:
        U_nash = best['U_expressed']
        try:
            payoffs_n, _, wages_n, prices_n, _, X_n, zone_n = \
                ccg_payoff_detailed(T, U_TRUE, U_nash, Y, P_INIT)
            desc_n = describe_forest(zone_n['I'], zone_n['J'], zone_n['F'], m=2)
            print(f"\n  Nash equilibrium forest: {desc_n['label']}")
            print(f"  Roles: {desc_n['roles']}")

            fig, _ = plot_forest_diagram(
                X_n, zone_n['I'], zone_n['J'], CLASS_LABELS, GOOD_LABELS,
                title=f'Nash Equilibrium Forest\n{desc_n["label"]}',
                output_file=savepath(outdir, '06_nash_forest.png'))
            plt.close(fig)

            fig, _ = plot_allocation_pattern(
                X_n, CLASS_LABELS, GOOD_LABELS,
                title='Nash Equilibrium Allocation',
                output_file=savepath(outdir, '06_nash_allocation.png'))
            plt.close(fig)
        except Exception as e:
            print(f"  Nash forest error: {e}")


def step7_verify(outdir, major_grid, forest_grid, alpha_grid, beta_grid):
    """Step 7: Verify zone structure against paper's analytical predictions.

    major_grid contains structural labels from step2.
    This function maps them to paper zones for comparison.
    """
    separator("STEP 7: VERIFICATION AGAINST PAPER")

    print("  Mapping computed structural zones to paper's zone names for comparison...")
    print()

    # Build structural-to-paper mapping for this 2x2 economy
    # (paper_zone_name does this mapping)
    def paper_zone_name(I, J, forest):
        """Map solver output to the paper's 2×2 zone name (F1–F4, Z5, CY)."""
        desc = describe_forest(I, J, forest, m=2)
        if desc['label'] == 'ERR':
            return 'ERR'
        if desc['n_active_classes'] < 2:
            return 'Z5'
        return PAPER_ZONE_MAP.get(desc['label'], desc['label'])

    # Convert major_grid (structural labels) to paper zone names for comparison
    n1, n2 = major_grid.shape
    computed_zone = np.empty((n1, n2), dtype='U5')
    for i in range(n1):
        for j in range(n2):
            struct_label = major_grid[i, j]
            if struct_label == 'ERR':
                computed_zone[i, j] = 'ERR'
            elif struct_label in PAPER_ZONE_MAP:
                computed_zone[i, j] = PAPER_ZONE_MAP[struct_label]
            elif struct_label.startswith('['):
                # Single-class zone (e.g. "[C1]_C1:{g0,g1}") → Z5 in paper
                computed_zone[i, j] = 'Z5'
            else:
                computed_zone[i, j] = 'OTH'

    # Paper's predicted zones (from Section 6, verified conditions)
    # NOTE: major_grid is indexed [beta_idx, alpha_idx], so iterate accordingly
    predicted_zone = np.empty((n1, n2), dtype='U5')
    for i, b in enumerate(beta_grid):
        for j, a in enumerate(alpha_grid):
            if b <= 0.25:
                predicted_zone[i, j] = 'Z5'
            elif a >= b and b > 0.25:
                predicted_zone[i, j] = 'F1'
            elif b >= a and a > 0.5:
                predicted_zone[i, j] = 'F2'
            elif 0.25 < b < 0.5 and b >= a:
                predicted_zone[i, j] = 'F3'
            elif a <= 0.5 and b >= 0.5:
                predicted_zone[i, j] = 'F4'
            else:
                predicted_zone[i, j] = 'UNK'

    # Match analysis
    matches = (computed_zone == predicted_zone)

    # Per-zone match rates
    zone_desc = {
        'F1': 'C0→g0, C1→both  (α≥β>¼)',
        'F2': 'C0→both, C1→g0  (β≥α>½)',
        'F3': 'C0→g1, C1→both  (¼<β<½,β≥α)',
        'F4': 'C0→g1, C1→g0    (α≤½,β≥½)',
        'Z5': 'Only C1 active   (β≤¼)',
    }
    print(f"  {'Zone':<6} {'Match':>7} {'Total':>7} {'Rate':>7}  Description")
    print(f"  {'-'*65}")
    for zn in ['F1', 'F2', 'F3', 'Z5', 'F4']:
        pred_mask = (predicted_zone == zn)
        zmatch = (computed_zone[pred_mask] == zn).sum()
        ztotal = pred_mask.sum()
        rate = 100 * zmatch / ztotal if ztotal > 0 else 0
        print(f"  {zn:<6} {zmatch:>7} {ztotal:>7} {rate:>6.1f}%  {zone_desc[zn]}")

    # Overall match rate
    total_points = matches.size
    total_matches = matches.sum()
    total_pct = 100 * total_matches / total_points
    print(f"\n  Overall match:                    {total_matches}/{total_points} = {total_pct:.1f}%")

    # Diagonal (α≈β) is a non-generic boundary — exclude for clean rate
    diag_mask = np.zeros((n1, n2), dtype=bool)
    da = alpha_grid[1] - alpha_grid[0] if len(alpha_grid) > 1 else 0.1
    for i, b in enumerate(beta_grid):
        for j, a in enumerate(alpha_grid):
            if abs(a - b) < da + 0.01:
                diag_mask[i, j] = True

    clean_mask = ~diag_mask
    clean_matches = matches[clean_mask].sum()
    clean_total = clean_mask.sum()
    clean_pct = 100 * clean_matches / clean_total if clean_total > 0 else 0

    print(f"  Excluding α≈β diagonal:           {clean_matches}/{clean_total} = {clean_pct:.1f}%")

    # 3-panel visualization
    zone_colors = {
        'F1': '#ff6666', 'F2': '#6699ff', 'F3': '#66ff66', 'F4': '#ffcc66',
        'CY': '#cc66ff', 'Z5': '#ffff66', 'ERR': '#333333', 'UNK': '#cccccc',
        'OTH': '#cccccc',
    }

    def make_rgb(grid, colors):
        h, w = grid.shape
        rgb = np.zeros((h, w, 3))
        for code, color in colors.items():
            mask = (grid == code)
            if mask.any():
                r = int(color[1:3], 16) / 255
                g = int(color[3:5], 16) / 255
                b = int(color[5:7], 16) / 255
                rgb[mask] = [r, g, b]
        return rgb

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    a_fine = np.linspace(alpha_grid[0], alpha_grid[-1], 500)

    # Panel 1: Paper's theoretical zones
    ax = axes[0]
    ax.imshow(make_rgb(predicted_zone, zone_colors),
              origin='lower', extent=[alpha_grid[0], alpha_grid[-1],
              beta_grid[0], beta_grid[-1]], aspect='auto', interpolation='nearest')
    ax.plot(a_fine, a_fine, 'k-', linewidth=2, alpha=0.8)
    ax.axhline(y=0.25, color='black', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(r'$\alpha$', fontsize=13)
    ax.set_ylabel(r'$\beta$', fontsize=13)
    ax.set_title("Analytical Zone Predictions", fontsize=14)
    ax.text(2.0, 1.0, 'F1', fontsize=14, ha='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8))
    ax.text(1.0, 2.0, 'F2', fontsize=14, ha='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8))
    ax.text(0.3, 0.38, 'F3', fontsize=12, ha='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.25, 1.5, 'F4', fontsize=12, ha='center', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7))
    ax.text(1.5, 0.12, 'Z5', fontsize=12, ha='center', fontweight='bold')

    # Panel 2: Computed zones (no hardcoded boundary lines — show only computed data)
    ax = axes[1]
    ax.imshow(make_rgb(computed_zone, zone_colors),
              origin='lower', extent=[alpha_grid[0], alpha_grid[-1],
              beta_grid[0], beta_grid[-1]], aspect='auto', interpolation='nearest')
    ax.set_xlabel(r'$\alpha$', fontsize=13)
    ax.set_ylabel(r'$\beta$', fontsize=13)
    ax.set_title("Computed Zones (scm library)", fontsize=14)

    # Panel 3: Match map
    ax = axes[2]
    match_rgb = np.zeros((n1, n2, 3))
    matches = (computed_zone == predicted_zone)
    diag_mask = np.zeros((n1, n2), dtype=bool)
    da = alpha_grid[1] - alpha_grid[0] if len(alpha_grid) > 1 else 0.1
    for i, b in enumerate(beta_grid):
        for j, a in enumerate(alpha_grid):
            if abs(a - b) < da + 0.01:
                diag_mask[i, j] = True

    match_rgb[matches] = [0.2, 0.8, 0.2]               # green = match
    match_rgb[~matches & ~diag_mask] = [1.0, 0.2, 0.2]  # red = mismatch
    match_rgb[~matches & diag_mask] = [1.0, 0.6, 0.0]   # orange = diagonal boundary
    ax.imshow(match_rgb,
              origin='lower', extent=[alpha_grid[0], alpha_grid[-1],
              beta_grid[0], beta_grid[-1]], aspect='auto', interpolation='nearest')
    ax.set_xlabel(r'$\alpha$', fontsize=13)
    ax.set_ylabel(r'$\beta$', fontsize=13)

    # Compute clean match rate (excluding non-generic diagonal)
    clean_mask = ~diag_mask
    clean_matches = matches[clean_mask].sum()
    clean_total = clean_mask.sum()
    clean_pct = 100 * clean_matches / clean_total if clean_total > 0 else 0
    overall_pct = 100 * matches.sum() / matches.size
    ax.set_title(f"Match: {overall_pct:.0f}% overall, {clean_pct:.0f}% excl. diagonal", fontsize=13)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6666', label='F1: C0→g0, C1→both'),
        Patch(facecolor='#6699ff', label='F2: C0→both, C1→g0'),
        Patch(facecolor='#66ff66', label='F3: C0→g1, C1→both'),
        Patch(facecolor='#ffcc66', label='F4: C0→g1, C1→g0'),
        Patch(facecolor='#ffff66', label='Z5: only C1 active'),
        Patch(facecolor='#cc66ff', label='CY: both buy both (α≈β)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(savepath(outdir, '07_zone_verification.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Reproduce paper 2×2 analysis')
    parser.add_argument('--outdir', default=None,
                        help='Output directory for figures')
    args = parser.parse_args()

    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'paper_2x2_figures')

    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(outdir)}")

    separator("PAPER 2×2 ECONOMY REPRODUCTION")
    print(f"  T      = {T.tolist()}")
    print(f"  U_true = {U_TRUE.tolist()}")
    print(f"  Y      = {Y.tolist()}")
    print(f"  Parameterization: U_expressed = [[alpha, 1], [beta, 1]]")

    # Run all steps
    baseline = step1_baseline(outdir)
    zone_grid, payoff_grid, wage_grid, forest_grid, major_grid, alpha_grid, beta_grid = \
        step2_zone_map(outdir)
    step3_forests(outdir)
    step4_sweeps(outdir)
    step5_gradients(outdir, zone_grid, beta_grid, alpha_grid)
    step6_nash(outdir)
    step7_verify(outdir, major_grid, forest_grid, alpha_grid, beta_grid)

    separator("COMPLETE")
    print(f"\n  All figures saved to: {os.path.abspath(outdir)}")
    print(f"  Total figures: {len([f for f in os.listdir(outdir) if f.endswith('.png')])}")


if __name__ == '__main__':
    main()
