"""
full_demo.py — Complete CCG walkthrough for 2×2 and 3×3 economies.

Generates paper-style figures: zone maps, payoff sweeps, Fisher forests,
gradient fields, and Nash search results.

Usage:
    python examples/full_demo.py
"""

import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scm import solve_robust, check_scm_equilibrium
from scm.ccg import (ccg_payoff, ccg_payoff_detailed, ccg_sweep,
                     ccg_gradient, ccg_zone_map, extract_forest, zone_label)
from scm.nash import nash_iteration, find_nash_candidates
from scm.visualize import (
    plot_zone_map, plot_zone_map_with_payoff,
    plot_payoff_trajectory, plot_wage_trajectory, plot_price_trajectory,
    plot_allocation_pattern, plot_forest_diagram, plot_gradient_field,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'docs', 'figures', 'demo')
os.makedirs(OUTDIR, exist_ok=True)


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ======================================================================
# PART A: 2×2 SOAP MARKET (from the paper)
# ======================================================================

def run_2x2():
    separator("PART A: 2×2 SOAP MARKET ECONOMY")

    # Economy definition (from FeigningU.m / paper Section 6)
    T = np.array([[0.2501, 0.0],
                   [0.25,   1.0]])
    U_true = np.array([[1.0, 1.0],
                        [1.0, 1.0]])
    Y = np.array([2.0, 4.0])
    p_init = np.array([2.0, 3.0])
    class_labels = ['Skilled (L0)', 'Unskilled (L1)']
    good_labels = ['Soap (g0)', 'Haircuts (g1)']

    print("Economy definition:")
    print(f"  T (technology):\n{T}")
    print(f"  U_true (true utilities):\n{U_true}")
    print(f"  Y (labour endowments): {Y}")
    print(f"  p_init (price guess): {p_init}")
    print()
    print("  Interpretation:")
    print("    - Good 0 (Soap): needs 0.2501 units of L0, 0.25 units of L1")
    print("    - Good 1 (Haircuts): needs 0 units of L0, 1 unit of L1")
    print("    - Both classes value both goods equally (U_true = all 1s)")
    print("    - L0 has 2 units of labour, L1 has 4 units")

    # ---- Step 1: Baseline equilibrium (honest play) ----
    separator("A.1: BASELINE EQUILIBRIUM (honest play, U_expressed = U_true)")

    result = solve_robust(T, U_true, Y, p_init)
    print(f"  Solver status:  {result['status']}")
    print(f"  Method used:    {result.get('method', 'standard')}")
    print(f"  Fixed-pt error: {result.get('fp_error', 'N/A')}")
    print(f"  Iterations:     {result.get('iterations', 'N/A')}")
    print()
    print(f"  Equilibrium prices:    p = {result['p']}")
    print(f"  Production:            q = {result['q']}")
    print(f"  Wages (income):        W = {result['W']}")
    print(f"  Active labour (I):     {result['I']}")
    print(f"  Active goods (J):      {result['J']}")
    print()
    print(f"  Allocation matrix X (rows=classes, cols=goods):")
    X = result['X']
    print(f"    {'':15s}  {good_labels[0]:>12s}  {good_labels[1]:>12s}")
    for i in range(2):
        print(f"    {class_labels[i]:15s}  {X[i,0]:12.4f}  {X[i,1]:12.4f}")

    # Verify equilibrium
    checks, all_pass = check_scm_equilibrium(result, T, U_true, Y)
    print(f"\n  Equilibrium verification: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    for name, passed in checks.items():
        print(f"    {name}: {'PASS' if passed else 'FAIL'}")

    # Forest at baseline
    forest, bpb = extract_forest(U_true, result['p'], X, result['I'], result['J'])
    zlabel = zone_label(result['I'], result['J'], forest)
    print(f"\n  Zone label: {zlabel}")
    print(f"  Fisher forest: {forest}")
    for k, i in enumerate(result['I']):
        bpb_str = ', '.join(f"g{j}:{r:.3f}" for j, r in bpb[k])
        print(f"    {class_labels[i]} BPB: [{bpb_str}]")

    # Payoffs at honest play
    payoffs_honest, _ = ccg_payoff(T, U_true, U_true, Y, p_init)
    print(f"\n  CCG Payoffs (honest): {payoffs_honest}")

    # Plots
    fig1, ax1 = plot_allocation_pattern(
        X, class_labels, good_labels,
        title='2×2 Soap Market: Honest Allocation',
        output_file=os.path.join(OUTDIR, '2x2_baseline_allocation.png'))
    plt.close(fig1)

    fig2, ax2 = plot_forest_diagram(
        X, result['I'], result['J'], class_labels, good_labels,
        title='2×2 Soap Market: Fisher Forest (Honest)',
        output_file=os.path.join(OUTDIR, '2x2_baseline_forest.png'))
    plt.close(fig2)

    # ---- Step 2: CCG parameterization and zone map ----
    separator("A.2: ZONE MAP (strategic play)")

    print("  Strategy parameterization (Section 6 convention):")
    print("    U_expressed = [[alpha, 1], [beta, 1]]")
    print("    alpha = Class 0's expressed preference for good 0 (rel. to good 1)")
    print("    beta  = Class 1's expressed preference for good 0 (rel. to good 1)")
    print("    alpha=1, beta=1 → honest play")
    print("    alpha>1 → Class 0 overstates desire for soap")
    print("    beta<1  → Class 1 understates desire for soap")

    def U_func_2x2(params):
        a = params.get('alpha', 1.0)
        b = params.get('beta', 1.0)
        return np.array([[a, 1.0], [b, 1.0]])

    alpha_range = np.linspace(0.1, 3.0, 40)
    beta_range = np.linspace(0.1, 3.0, 40)

    print(f"\n  Computing zone map over {len(alpha_range)}×{len(beta_range)} grid...")
    zone_grid, payoff_grid, wage_grid, forest_grid = ccg_zone_map(
        T, U_true, Y, p_init, U_func_2x2,
        alpha_range, beta_range,
        param1_name='alpha', param2_name='beta',
        verbose=False)

    unique_zones = sorted(set(zone_grid.ravel()))
    print(f"  Found {len(unique_zones)} distinct zones:")
    for z in unique_zones:
        count = np.sum(zone_grid == z)
        pct = 100.0 * count / zone_grid.size
        print(f"    {z}  ({count} pts, {pct:.1f}%)")

    # Zone map plot
    fig3, ax3 = plot_zone_map(
        zone_grid, alpha_range, beta_range,
        param1_name='alpha (L0 pref for g0)',
        param2_name='beta (L1 pref for g0)',
        title='2×2 Soap Market: Zone Structure (I, J, F)',
        output_file=os.path.join(OUTDIR, '2x2_zone_map.png'))
    plt.close(fig3)

    # Zone + payoff for each player
    for player in range(2):
        fig, axes = plot_zone_map_with_payoff(
            zone_grid, payoff_grid, alpha_range, beta_range,
            player=player,
            param1_name='alpha', param2_name='beta',
            title=f'2×2 Soap Market: Zones + {class_labels[player]} Payoff',
            output_file=os.path.join(OUTDIR, f'2x2_zone_payoff_class{player}.png'))
        plt.close(fig)

    # ---- Step 3: Payoff sweeps (1D, like FeigningU.m) ----
    separator("A.3: PAYOFF SWEEPS (1D cross-sections)")

    # Sweep beta at fixed alpha=1 (Class 1 varies strategy, Class 0 honest)
    print("  Sweep 1: beta varies (0.1→3.0), alpha=1.0 fixed")
    grid_beta = [{'alpha': 1.0, 'beta': b} for b in beta_range]
    results_beta = ccg_sweep(T, U_true, Y, p_init, U_func_2x2, grid_beta)

    payoff_beta = np.array([r['payoffs'] for r in results_beta])
    wage_beta = np.array([r['wages'] for r in results_beta])
    price_beta = np.array([r['prices'] for r in results_beta])
    zlabels_beta = [r['zone_label'] for r in results_beta]

    # Find best strategy for each player
    best_beta_for_p0 = beta_range[np.argmax(payoff_beta[:, 0])]
    best_beta_for_p1 = beta_range[np.argmax(payoff_beta[:, 1])]
    print(f"    Best beta for L0's payoff: {best_beta_for_p0:.2f} (payoff={payoff_beta[:, 0].max():.4f})")
    print(f"    Best beta for L1's payoff: {best_beta_for_p1:.2f} (payoff={payoff_beta[:, 1].max():.4f})")
    print(f"    Honest beta=1.0: payoffs = {payoff_beta[np.argmin(np.abs(beta_range-1.0))]}")

    fig4, ax4 = plot_payoff_trajectory(
        beta_range, payoff_beta, class_labels,
        param_name='beta (L1 pref for g0)',
        title='2×2 Soap: Payoffs vs beta (alpha=1)',
        zone_labels=zlabels_beta,
        output_file=os.path.join(OUTDIR, '2x2_payoff_vs_beta.png'))
    plt.close(fig4)

    fig5, ax5 = plot_wage_trajectory(
        beta_range, wage_beta, class_labels,
        param_name='beta (L1 pref for g0)',
        title='2×2 Soap: Wages vs beta (alpha=1)',
        zone_labels=zlabels_beta,
        output_file=os.path.join(OUTDIR, '2x2_wage_vs_beta.png'))
    plt.close(fig5)

    fig5b, ax5b = plot_price_trajectory(
        beta_range, price_beta, good_labels,
        param_name='beta (L1 pref for g0)',
        title='2×2 Soap: Prices vs beta (alpha=1)',
        zone_labels=zlabels_beta,
        output_file=os.path.join(OUTDIR, '2x2_price_vs_beta.png'))
    plt.close(fig5b)

    # Sweep alpha at fixed beta=1 (Class 0 varies strategy, Class 1 honest)
    print("\n  Sweep 2: alpha varies (0.1→3.0), beta=1.0 fixed")
    grid_alpha = [{'alpha': a, 'beta': 1.0} for a in alpha_range]
    results_alpha = ccg_sweep(T, U_true, Y, p_init, U_func_2x2, grid_alpha)

    payoff_alpha = np.array([r['payoffs'] for r in results_alpha])
    zlabels_alpha = [r['zone_label'] for r in results_alpha]

    best_alpha_for_p0 = alpha_range[np.argmax(payoff_alpha[:, 0])]
    best_alpha_for_p1 = alpha_range[np.argmax(payoff_alpha[:, 1])]
    print(f"    Best alpha for L0's payoff: {best_alpha_for_p0:.2f} (payoff={payoff_alpha[:, 0].max():.4f})")
    print(f"    Best alpha for L1's payoff: {best_alpha_for_p1:.2f} (payoff={payoff_alpha[:, 1].max():.4f})")

    fig6, ax6 = plot_payoff_trajectory(
        alpha_range, payoff_alpha, class_labels,
        param_name='alpha (L0 pref for g0)',
        title='2×2 Soap: Payoffs vs alpha (beta=1)',
        zone_labels=zlabels_alpha,
        output_file=os.path.join(OUTDIR, '2x2_payoff_vs_alpha.png'))
    plt.close(fig6)

    # ---- Step 4: Forest diagrams at interesting points ----
    separator("A.4: FISHER FORESTS AT KEY POINTS")

    interesting_points = [
        {'alpha': 1.0, 'beta': 1.0, 'label': 'honest'},
        {'alpha': 0.3, 'beta': 1.0, 'label': 'L0_understates_g0'},
        {'alpha': 2.0, 'beta': 1.0, 'label': 'L0_overstates_g0'},
        {'alpha': 1.0, 'beta': 0.3, 'label': 'L1_understates_g0'},
        {'alpha': 1.0, 'beta': 2.0, 'label': 'L1_overstates_g0'},
    ]

    for pt in interesting_points:
        U_expr = U_func_2x2(pt)
        payoffs, pm, wages, prices, quant, X_pt, zone = \
            ccg_payoff_detailed(T, U_true, U_expr, Y, p_init)
        zlbl = zone_label(zone['I'], zone['J'], zone['F'])
        print(f"  Point: alpha={pt['alpha']}, beta={pt['beta']} ({pt['label']})")
        print(f"    Zone:    {zlbl}")
        print(f"    Payoffs: {payoffs}")
        print(f"    Wages:   {wages}")
        print(f"    Prices:  {prices}")
        print(f"    X:\n{X_pt}")
        print()

        fig, ax = plot_forest_diagram(
            X_pt, zone['I'], zone['J'], class_labels, good_labels,
            title=f'Forest: {pt["label"]} (a={pt["alpha"]}, b={pt["beta"]})',
            output_file=os.path.join(OUTDIR, f'2x2_forest_{pt["label"]}.png'))
        plt.close(fig)

    # ---- Step 5: Gradient field ----
    separator("A.5: GRADIENT ANALYSIS")

    print("  Computing gradient at honest play...")
    J_mat = ccg_gradient(T, U_true, U_true, Y, p_init)
    for player in range(2):
        grad = J_mat[player]
        own = grad[player]
        mag = np.linalg.norm(own)
        print(f"  {class_labels[player]}:")
        print(f"    Own-row gradient: [{own[0]:+.6f}, {own[1]:+.6f}]  |mag|={mag:.6f}")
        if mag > 1e-6:
            print(f"    Direction: {own/mag}")
            print(f"    → This player can improve by moving in that direction")
        else:
            print(f"    → At local optimum (no unilateral improvement)")

    # Gradient field over coarser grid
    print("\n  Computing gradient field over 12×12 grid...")
    a_coarse = np.linspace(0.2, 2.5, 12)
    b_coarse = np.linspace(0.2, 2.5, 12)
    grad_grid_p0 = np.zeros((12, 12, 2))
    grad_grid_p1 = np.zeros((12, 12, 2))
    zone_grid_coarse = np.empty((12, 12), dtype=object)

    for i, a in enumerate(a_coarse):
        for j, b in enumerate(b_coarse):
            U_expr = U_func_2x2({'alpha': a, 'beta': b})
            payoffs, _, _, _, _, _, zone = \
                ccg_payoff_detailed(T, U_true, U_expr, Y, p_init)
            zone_grid_coarse[i, j] = zone_label(zone['I'], zone['J'], zone['F'])

            J_loc = ccg_gradient(T, U_true, U_expr, Y, p_init)
            # Player 0's gradient w.r.t. alpha (own row, good 1)
            grad_grid_p0[i, j, 0] = J_loc[0, 0, 1]  # ∂payoff_0/∂alpha (param1 direction)
            grad_grid_p0[i, j, 1] = 0  # Class 0 doesn't control beta
            # Player 1's gradient w.r.t. beta (own row, good 0)
            grad_grid_p1[i, j, 0] = 0  # Class 1 doesn't control alpha
            grad_grid_p1[i, j, 1] = J_loc[1, 1, 0]  # ∂payoff_1/∂beta (param2 direction)

    fig7, ax7 = plot_gradient_field(
        grad_grid_p0, a_coarse, b_coarse,
        param1_name='alpha', param2_name='beta',
        zone_grid=zone_grid_coarse,
        title='2×2 Soap: L0 Gradient (∂payoff₀/∂alpha)',
        output_file=os.path.join(OUTDIR, '2x2_gradient_p0.png'))
    plt.close(fig7)

    fig8, ax8 = plot_gradient_field(
        grad_grid_p1, a_coarse, b_coarse,
        param1_name='alpha', param2_name='beta',
        zone_grid=zone_grid_coarse,
        title='2×2 Soap: L1 Gradient (∂payoff₁/∂beta)',
        output_file=os.path.join(OUTDIR, '2x2_gradient_p1.png'))
    plt.close(fig8)

    # ---- Step 6: Nash equilibrium search ----
    separator("A.6: NASH EQUILIBRIUM SEARCH")

    print("  Running multi-start Nash search (5 restarts, up to 50 iters each)...")
    candidates = find_nash_candidates(
        T, U_true, Y, p_init,
        n_restarts=5, max_iter=50, lr=0.08, tol=1e-4,
        verbose=False)

    for i, c in enumerate(candidates):
        print(f"\n  Candidate {i+1}:")
        print(f"    Converged:     {c['converged']}")
        print(f"    Gap:           {c['convergence_gap']:.8f}")
        print(f"    Payoffs:       {c['payoffs']}")
        print(f"    U_expressed:\n{c['U_expressed']}")

    # Run one iteration with verbose to show trajectory
    print("\n  Detailed Nash iteration from honest play:")
    result_nash = nash_iteration(
        T, U_true, U_true, Y, p_init,
        max_iter=30, lr=0.08, tol=1e-4, verbose=True)

    # Plot Nash convergence
    fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(14, 5))
    payoffs_traj = result_nash['payoffs']
    for p_idx in range(2):
        ax9a.plot(payoffs_traj[:, p_idx], 'o-', label=class_labels[p_idx])
    ax9a.set_xlabel('Iteration')
    ax9a.set_ylabel('Payoff')
    ax9a.set_title('Nash Iteration: Payoff Convergence')
    ax9a.legend()
    ax9a.grid(True, alpha=0.3)

    mags_traj = result_nash['magnitudes']
    for p_idx in range(2):
        ax9b.plot(mags_traj[:, p_idx], 's-', label=class_labels[p_idx])
    ax9b.set_xlabel('Iteration')
    ax9b.set_ylabel('Gradient Magnitude')
    ax9b.set_title('Nash Iteration: Gradient Convergence')
    ax9b.legend()
    ax9b.grid(True, alpha=0.3)
    ax9b.set_yscale('log')

    fig9.tight_layout()
    fig9.savefig(os.path.join(OUTDIR, '2x2_nash_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig9)

    print(f"\n  Nash result: converged={result_nash['converged']}, iters={result_nash['n_iter']}")

    return {
        'payoffs_honest': payoffs_honest,
        'zone_count': len(unique_zones),
        'candidates': candidates,
    }


# ======================================================================
# PART B: 3×3 ECONOMY
# ======================================================================

def run_3x3():
    separator("PART B: 3×3 ECONOMY (3 classes, 3 goods)")

    # A richer economy: Tech workers, Service workers, Manual workers
    # Goods: Electronics, Services, Food
    T = np.array([
        [1.0,  0.0,  0.0],   # Electronics: only Tech labour
        [0.3,  1.0,  0.0],   # Services: some Tech + Service labour
        [0.0,  0.2,  1.0],   # Food: some Service + Manual labour
    ])
    U_true = np.array([
        [0.5, 1.0, 0.8],   # Tech workers: prefer services
        [0.8, 0.5, 1.0],   # Service workers: prefer food
        [1.0, 0.8, 0.5],   # Manual workers: prefer electronics
    ])
    Y = np.array([3.0, 5.0, 4.0])
    p_init = np.array([1.0, 1.0, 1.0])

    class_labels = ['Tech (L0)', 'Service (L1)', 'Manual (L2)']
    good_labels = ['Electronics (g0)', 'Services (g1)', 'Food (g2)']

    print("Economy definition:")
    print(f"  T (technology):\n{T}")
    print(f"  U_true (true utilities):\n{U_true}")
    print(f"  Y (labour): {Y}")
    print()
    print("  Structure:")
    print("    - Electronics: produced only by Tech workers")
    print("    - Services: produced by Tech + Service workers")
    print("    - Food: produced by Service + Manual workers")
    print("    - Each class prefers a good that another class produces")
    print("    - This creates interdependence and strategic tension")

    # ---- Baseline equilibrium ----
    separator("B.1: BASELINE EQUILIBRIUM (3×3)")

    result = solve_robust(T, U_true, Y, p_init)
    print(f"  Status:       {result['status']}")
    print(f"  Prices:       {result['p']}")
    print(f"  Production:   {result['q']}")
    print(f"  Wages:        {result['W']}")
    print(f"  Active I:     {result['I']}")
    print(f"  Active J:     {result['J']}")

    X = result['X']
    print(f"\n  Allocation matrix:")
    header = f"  {'':15s}" + ''.join(f"  {g:>14s}" for g in good_labels)
    print(header)
    for i in range(3):
        row = f"  {class_labels[i]:15s}" + ''.join(f"  {X[i,j]:14.4f}" for j in range(3))
        print(row)

    forest, bpb = extract_forest(U_true, result['p'], X, result['I'], result['J'])
    zlabel = zone_label(result['I'], result['J'], forest)
    print(f"\n  Zone: {zlabel}")

    payoffs_honest, _ = ccg_payoff(T, U_true, U_true, Y, p_init)
    print(f"  Payoffs (honest): {payoffs_honest}")

    fig10, ax10 = plot_allocation_pattern(
        X, class_labels, good_labels,
        title='3×3 Economy: Honest Allocation',
        output_file=os.path.join(OUTDIR, '3x3_baseline_allocation.png'))
    plt.close(fig10)

    fig11, ax11 = plot_forest_diagram(
        X, result['I'], result['J'], class_labels, good_labels,
        title='3×3 Economy: Fisher Forest (Honest)',
        output_file=os.path.join(OUTDIR, '3x3_baseline_forest.png'))
    plt.close(fig11)

    # ---- Zone map: vary two players' strategies ----
    separator("B.2: ZONE MAP (3×3) — Tech vs Service strategies")

    # Parameterize: Tech scales preference for electronics (alpha),
    # Service scales preference for electronics (beta)
    # This captures competition for electronics
    def U_func_3x3(params):
        a = params.get('alpha', 1.0)
        b = params.get('beta', 1.0)
        U = U_true.copy()
        U[0, 0] = a * U_true[0, 0]  # Tech's expressed pref for electronics
        U[1, 0] = b * U_true[1, 0]  # Service's expressed pref for electronics
        return U

    alpha_range_3 = np.linspace(0.1, 3.0, 30)
    beta_range_3 = np.linspace(0.1, 3.0, 30)

    print(f"  Computing zone map over {len(alpha_range_3)}×{len(beta_range_3)} grid...")
    print("  alpha = Tech's scaling of pref for electronics")
    print("  beta  = Service's scaling of pref for electronics")

    zone_grid_3, payoff_grid_3, wage_grid_3, forest_grid_3 = ccg_zone_map(
        T, U_true, Y, p_init, U_func_3x3,
        alpha_range_3, beta_range_3,
        param1_name='alpha', param2_name='beta',
        verbose=False)

    unique_3 = sorted(set(zone_grid_3.ravel()))
    print(f"\n  Found {len(unique_3)} distinct zones:")
    for z in unique_3:
        count = np.sum(zone_grid_3 == z)
        pct = 100.0 * count / zone_grid_3.size
        print(f"    {z}  ({count} pts, {pct:.1f}%)")

    fig12, ax12 = plot_zone_map(
        zone_grid_3, alpha_range_3, beta_range_3,
        param1_name='alpha (Tech pref for electronics)',
        param2_name='beta (Service pref for electronics)',
        title='3×3 Economy: Zone Structure',
        output_file=os.path.join(OUTDIR, '3x3_zone_map.png'))
    plt.close(fig12)

    for player in range(3):
        fig, axes = plot_zone_map_with_payoff(
            zone_grid_3, payoff_grid_3, alpha_range_3, beta_range_3,
            player=player,
            param1_name='alpha (Tech)', param2_name='beta (Service)',
            title=f'3×3: Zones + {class_labels[player]} Payoff',
            output_file=os.path.join(OUTDIR, f'3x3_zone_payoff_class{player}.png'))
        plt.close(fig)

    # ---- 1D sweep: Service varies, others honest ----
    separator("B.3: PAYOFF SWEEP (3×3) — Service varies strategy")

    grid_3_beta = [{'alpha': 1.0, 'beta': b} for b in beta_range_3]
    results_3_beta = ccg_sweep(T, U_true, Y, p_init, U_func_3x3, grid_3_beta)

    payoff_3_beta = np.array([r['payoffs'] for r in results_3_beta])
    wage_3_beta = np.array([r['wages'] for r in results_3_beta])
    zlabels_3_beta = [r['zone_label'] for r in results_3_beta]

    for p_idx in range(3):
        best_b = beta_range_3[np.argmax(payoff_3_beta[:, p_idx])]
        best_pay = payoff_3_beta[:, p_idx].max()
        print(f"  {class_labels[p_idx]}: best at beta={best_b:.2f}, payoff={best_pay:.4f}")

    fig13, ax13 = plot_payoff_trajectory(
        beta_range_3, payoff_3_beta, class_labels,
        param_name='beta (Service pref for electronics)',
        title='3×3: Payoffs vs beta (alpha=1)',
        zone_labels=zlabels_3_beta,
        output_file=os.path.join(OUTDIR, '3x3_payoff_vs_beta.png'))
    plt.close(fig13)

    fig14, ax14 = plot_wage_trajectory(
        beta_range_3, wage_3_beta, class_labels,
        param_name='beta (Service pref for electronics)',
        title='3×3: Wages vs beta (alpha=1)',
        zone_labels=zlabels_3_beta,
        output_file=os.path.join(OUTDIR, '3x3_wage_vs_beta.png'))
    plt.close(fig14)

    # ---- Forest at selected points ----
    separator("B.4: FISHER FORESTS (3×3)")

    points_3 = [
        {'alpha': 1.0, 'beta': 1.0, 'label': 'honest'},
        {'alpha': 2.0, 'beta': 0.5, 'label': 'Tech_high_Serv_low'},
        {'alpha': 0.5, 'beta': 2.0, 'label': 'Tech_low_Serv_high'},
    ]

    for pt in points_3:
        U_expr = U_func_3x3(pt)
        payoffs, _, wages, prices, _, X_pt, zone = \
            ccg_payoff_detailed(T, U_true, U_expr, Y, p_init)
        zlbl = zone_label(zone['I'], zone['J'], zone['F'])
        print(f"  {pt['label']} (a={pt['alpha']}, b={pt['beta']}):")
        print(f"    Zone: {zlbl}")
        print(f"    Payoffs: {payoffs}")
        print(f"    Prices:  {prices}")
        print()

        fig, ax = plot_forest_diagram(
            X_pt, zone['I'], zone['J'], class_labels, good_labels,
            title=f'3×3 Forest: {pt["label"]}',
            output_file=os.path.join(OUTDIR, f'3x3_forest_{pt["label"]}.png'))
        plt.close(fig)

    # ---- Nash search ----
    separator("B.5: NASH EQUILIBRIUM SEARCH (3×3)")

    print("  Running Nash search (3 restarts)...")
    candidates_3 = find_nash_candidates(
        T, U_true, Y, p_init,
        n_restarts=3, max_iter=30, lr=0.05, tol=1e-3,
        verbose=False)

    for i, c in enumerate(candidates_3):
        print(f"\n  Candidate {i+1}:")
        print(f"    Converged: {c['converged']}")
        print(f"    Gap:       {c['convergence_gap']:.6f}")
        print(f"    Payoffs:   {c['payoffs']}")

    return {
        'payoffs_honest': payoffs_honest,
        'zone_count': len(unique_3),
    }


# ======================================================================
# SUMMARY
# ======================================================================

def print_summary(r2, r3):
    separator("SUMMARY: ALL FIGURES GENERATED")

    files = sorted(os.listdir(OUTDIR))
    print(f"  Output directory: {OUTDIR}")
    print(f"  Total figures: {len([f for f in files if f.endswith('.png')])}")
    print()
    for f in files:
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(OUTDIR, f))
            print(f"    {f}  ({size//1024} KB)")


if __name__ == '__main__':
    r2 = run_2x2()
    r3 = run_3x3()
    print_summary(r2, r3)
