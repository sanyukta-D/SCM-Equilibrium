"""
scm  –  Simple Closed Model equilibrium computation

Compute SM equilibria for economies with linear or piecewise-linear-concave
(PLC) utilities, using the tatonnement algorithm from Deshpande & Sohoni
(arXiv:2109.09248).

Quick start
-----------
    import numpy as np
    from scm import compute_equilibrium, check_scm_equilibrium

    T = np.array([[1.0, 0.0],
                   [1.0, 1.0]])
    U = np.array([[1.0, 0.8],
                   [0.8, 1.0]])
    Y = np.array([2.0, 4.0])
    p_init = np.array([1.0, 1.0])

    result = compute_equilibrium(T, U, Y, p_init)
    checks, ok = check_scm_equilibrium(result, T, U, Y)
"""

# Core solvers
from .production       import solve_production, wages_from_prices
from .fisher_market    import solve_fisher
from .fisher_market_plc import solve_fisher_plc, solve_fisher_plc_3d
from .fisher_market_splc import solve_fisher_splc

# Single SCM round
from .scm_round     import scm_round
from .scm_round_plc import scm_round_plc
from .scm_round_splc import scm_round_splc

# Tatonnement equilibrium computation
from .equilibrium     import compute_equilibrium, print_equilibrium
from .equilibrium_plc import compute_equilibrium_plc, print_equilibrium_plc
from .equilibrium_splc import compute_equilibrium_splc, print_equilibrium_splc

# Equilibrium verification
from .verify import check_scm_equilibrium, check_plc_equilibrium

# Robust solvers (Task 2: alternative methods)
from .solvers import solve_robust, solve_damped, solve_broyden

# Consumer Choice Game (Task 3: strategic preference expression)
from .ccg import (ccg_payoff, ccg_payoff_detailed, ccg_sweep, ccg_gradient,
                  ccg_zone_map, extract_forest, zone_label,
                  describe_forest, classify_zone)

# Nash equilibrium finder
from .nash import best_response_direction, nash_iteration, find_nash_candidates

# Visualization (optional — requires matplotlib)
try:
    from .visualize import (
        plot_zone_map, plot_zone_map_with_payoff,
        plot_payoff_trajectory, plot_wage_trajectory, plot_price_trajectory,
        plot_allocation_pattern, plot_forest_diagram, plot_gradient_field,
    )
except ImportError:
    pass  # matplotlib not installed

__version__ = "0.4.0"
