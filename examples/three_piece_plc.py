"""
Three-Piece PLC Example: "Developing Industrial Economy"
=========================================================

3 classes × 3 goods with 3-segment piecewise-linear-concave utilities.

Economic narrative
------------------
  Goods:   0 = Food,  1 = Housing,  2 = Manufacturing
  Classes: 0 = Farmers, 1 = Builders, 2 = Workers

Each class has diminishing marginal utility with 3 tiers:
  Segment 1 (Essential): High utility — basic needs, small capacity
  Segment 2 (Comfortable): Medium utility — comfortable living, medium capacity
  Segment 3 (Luxury): Low utility — excess/luxury, unlimited

Comparative advantages in production:
  - Farmers produce Food cheaply (low T[0,0])
  - Builders produce Housing cheaply (low T[1,1])
  - Workers produce Manufacturing cheaply (low T[2,2])

Consumption preferences (cross-class demand creates trade):
  - Farmers want Food (own) and Housing
  - Builders want Housing (own) and Manufacturing
  - Workers want Manufacturing (own) and Food
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scm.scm_round_splc import scm_round_splc
from scm.equilibrium_splc import compute_equilibrium_splc, print_equilibrium_splc

# =========================================================================
#  Economy parameters
# =========================================================================

# Technology matrix T[i,j]: units of class-i labour per unit of good j
# Low diagonal = comparative advantage, but not too extreme
T = np.array([
    [0.8, 1.8, 1.5],   # Farmers: cheapest food, costly housing/mfg
    [1.5, 0.7, 1.3],   # Builders: cheapest housing
    [1.3, 1.6, 0.6],   # Workers: cheapest manufacturing
])

# Labour availability
Y = np.array([10.0, 10.0, 10.0])

# Utility matrices: U[i,j,s] for segment s ∈ {0,1,2}
# Shape: (3, 3, 3)
#
# Segment 0 (Essential): highest marginal utility — basic needs
# Segment 1 (Comfortable): medium — comfortable living
# Segment 2 (Luxury): lowest — excess consumption
#
# Key design: every class has high essential utility for ALL goods,
# creating cross-class demand. The differentiation is in how quickly
# utility falls off (which goods hit comfortable/luxury tiers faster).
#
# Farmers  → strong Food preference, moderate Housing, some Mfg
# Builders → strong Housing, moderate Mfg, some Food
# Workers  → strong Mfg, moderate Food, some Housing

U = np.array([
    # Farmers: food-loving, need housing too, use some mfg
    [[6.0, 4.0, 1.5],    # Food:    essential=6, comfort=4, luxury=1.5
     [4.0, 2.5, 0.8],    # Housing: essential=4, comfort=2.5, luxury=0.8
     [3.0, 1.5, 0.5]],   # Mfg:     essential=3, comfort=1.5, luxury=0.5

    # Builders: housing-focused, like manufacturing, need food
    [[3.0, 1.5, 0.5],    # Food:    essential=3, comfort=1.5, luxury=0.5
     [6.0, 4.0, 1.2],    # Housing: essential=6, comfort=4, luxury=1.2
     [4.5, 2.5, 0.8]],   # Mfg:     essential=4.5, comfort=2.5, luxury=0.8

    # Workers: manufacturing-focused, like food, need housing
    [[4.5, 2.5, 0.8],    # Food:    essential=4.5, comfort=2.5, luxury=0.8
     [3.0, 1.5, 0.5],    # Housing: essential=3, comfort=1.5, luxury=0.5
     [6.0, 4.0, 1.5]],   # Mfg:     essential=6, comfort=4, luxury=1.5
])

# Segment capacity limits: L[i,j,s]
# Segment 0: small (essential needs — first 1-2 units)
# Segment 1: medium (comfortable — next 2-3 units)
# Segment 2: unlimited (luxury — everything beyond)
BIG = 1e6

L = np.array([
    # Farmers
    [[1.5, 3.0, BIG],    # Food: 1.5 essential, 3 comfortable, rest luxury
     [1.0, 2.0, BIG],    # Housing: 1 essential, 2 comfortable
     [0.5, 1.5, BIG]],   # Mfg: 0.5 essential, 1.5 comfortable

    # Builders
    [[0.5, 1.5, BIG],    # Food
     [1.5, 3.0, BIG],    # Housing
     [1.0, 2.0, BIG]],   # Mfg

    # Workers
    [[1.0, 2.0, BIG],    # Food
     [0.5, 1.5, BIG],    # Housing
     [1.5, 3.0, BIG]],   # Mfg
])


# =========================================================================
#  Plot utility functions
# =========================================================================

def plot_utilities(U, L, save_path=None):
    """
    Plot the 3-piece PLC utility curves V_ij(x) for each class and good.
    """
    class_names = ['Farmers (Class 0)', 'Builders (Class 1)', 'Workers (Class 2)']
    good_names = ['Food', 'Housing', 'Manufacturing']
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('3-Piece PLC Utility Functions by Class',
                 fontsize=14, fontweight='bold', y=1.02)

    for i, (ax, cname) in enumerate(zip(axes, class_names)):
        ax.set_title(cname, fontsize=12)
        ax.set_xlabel('Units consumed (x)', fontsize=10)
        ax.set_ylabel('Utility V(x)', fontsize=10)

        for j, (gname, color) in enumerate(zip(good_names, colors)):
            # Build the piecewise linear curve
            breakpoints_x = [0]
            breakpoints_v = [0]
            cumulative_v = 0

            for s in range(3):
                cap = L[i, j, s]
                slope = U[i, j, s]

                if s < 2:
                    # Bounded segment
                    seg_len = cap
                else:
                    # Last segment: extend to show some luxury consumption
                    seg_len = max(3.0, breakpoints_x[-1] * 0.5)

                x_end = breakpoints_x[-1] + seg_len
                v_end = cumulative_v + slope * seg_len

                breakpoints_x.append(x_end)
                breakpoints_v.append(v_end)
                cumulative_v = v_end

            ax.plot(breakpoints_x, breakpoints_v, color=color,
                    linewidth=2, label=gname)

            # Mark segment boundaries with dots
            for s in range(2):
                idx = s + 1
                ax.plot(breakpoints_x[idx], breakpoints_v[idx],
                        'o', color=color, markersize=5, zorder=5)

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved utility plot to: {save_path}")
    plt.close()


# =========================================================================
#  Main: plot + solve
# =========================================================================

if __name__ == '__main__':
    # --- Plot utilities ---
    plot_path = os.path.join(os.path.dirname(__file__), '..', 'docs',
                             'three_piece_plc_utilities.png')
    plot_utilities(U, L, save_path=plot_path)

    # --- Solve economy ---
    print("=" * 60)
    print("  3-Piece PLC Economy: Developing Industrial Economy")
    print("=" * 60)
    print()
    print("Technology matrix T:")
    print(T)
    print(f"\nLabour Y = {Y}")
    print()

    p_init = np.array([1.0, 1.0, 1.0])

    # Try damped tatonnement with normalisation (PLC economies often diverge
    # under standard tatonnement due to zone-boundary discontinuities)
    for alpha in [0.3, 0.1, 0.05]:
        iters = {0.3: 1000, 0.1: 2000, 0.05: 3000}[alpha]
        print(f"\nAttempt: Damped tatonnement (α={alpha}, normalised, {iters} iters)")
        result = compute_equilibrium_splc(
            T, U, L, Y, p_init,
            max_iter=iters, tol=1e-6,
            cycle_window=10,
            damped=True, alpha=alpha, normalise=True
        )
        print_equilibrium_splc(result, label=f"Damped α={alpha}")

        if result['status'] == 'converged':
            break

    # --- Verify fixed point ---
    print("\nFixed-point verification:")
    p_eq = result['p']
    p_next, q, W, X_units, I, J = scm_round_splc(T, U, L, Y, p_eq)
    fp_err = float(np.max(np.abs(p_next / p_next.mean() - p_eq / p_eq.mean())))
    print(f"  max|G(p)/mean - p/mean| = {fp_err:.2e}")
    print(f"  p · q = {p_eq @ q:.6f}")
    print(f"  Σ W   = {W.sum():.6f}")

    # --- Consumption summary ---
    print("\nConsumption Summary (units per class per good):")
    class_names = ['Farmers', 'Builders', 'Workers']
    good_names = ['Food', 'Housing', 'Mfg']
    X_total = result['X_total']

    for i, cn in enumerate(class_names):
        print(f"\n  {cn} (wage income = {result['W'][i]:.4f}):")
        for j, gn in enumerate(good_names):
            total_u = sum(result['X_units'][i, j, s] for s in range(3))
            segs = [f"s{s+1}={result['X_units'][i,j,s]:.3f}" for s in range(3)]
            print(f"    {gn}: {total_u:.3f} units  ({', '.join(segs)})")
