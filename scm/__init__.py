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

# Single SCM round
from .scm_round     import scm_round
from .scm_round_plc import scm_round_plc

# Tatonnement equilibrium computation
from .equilibrium     import compute_equilibrium, print_equilibrium
from .equilibrium_plc import compute_equilibrium_plc, print_equilibrium_plc

# Equilibrium verification
from .verify import check_scm_equilibrium, check_plc_equilibrium

__version__ = "0.1.0"
