"""
scm_round.py - One full production-consumption round of the SCM

Mirrors fm.m exactly for the linear-utility case:

  1. Production LP  →  q (quantities), I (active labour), J (active goods)
  2. Wages          →  w[I] = p[J] @ inv(T[I,J]),  W[I] = w[I] * Y[I]
  3. Normalise      →  m = W / sum(W)               (relative budgets, sum=1)
  4. Fisher market  →  new prices p_new[J], allocations X
  5. Price update for inactive goods J̄ using bang-per-buck rule from fm.m
  6. Return         →  p_new, q, W (absolute wages), X_units, I, J
"""

import numpy as np
from .production  import solve_production, wages_from_prices
from .fisher_market import solve_fisher


def scm_round(T, U, Y, p, tol_active=1e-5):
    """
    One round of the SCM: prices in → prices out.

    Parameters
    ----------
    T : (m, n)   Technology matrix.
    U : (m, n)   Utility matrix (linear).
    Y : (m,)     Labour availability.
    p : (n,)     Current price vector.

    Returns
    -------
    p_new   : (n,)    Updated price vector.
    q       : (n,)    Production quantities.
    W       : (m,)    Absolute wage income per class (0 for surplus labour).
    X_units : (m, n)  Goods allocation (units).
    I       : array   Indices of active labour classes.
    J       : array   Indices of active goods.
    """
    T = np.array(T, dtype=float)
    U = np.array(U, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p = np.array(p, dtype=float).ravel()
    m, n = T.shape

    # ------------------------------------------------------------------ #
    #  Step 1 – Production LP                                              #
    # ------------------------------------------------------------------ #
    q, w_lp, wages_lp, J, I, _revenue = solve_production(T, Y, p, tol_active)

    if len(J) == 0:
        raise ValueError("Production LP produced no active goods. Check T, Y, p.")
    if len(I) == 0:
        raise ValueError("No binding labour constraints. Check T, Y, p.")

    # ------------------------------------------------------------------ #
    #  Step 2 – Wages from active submatrix  (matches fm.m exactly)       #
    # ------------------------------------------------------------------ #
    T_sub = T[np.ix_(I, J)]          # (|I| × |J|) active submatrix
    p_sub = p[J]
    Y_sub = Y[I]

    W_active = wages_from_prices(T_sub, p_sub, Y_sub)

    W = np.zeros(m)
    W[I] = W_active

    # ------------------------------------------------------------------ #
    #  Step 3 – Normalise budgets (sum → 1, matching fm.m / fisherm.m)    #
    # ------------------------------------------------------------------ #
    total_W = W_active.sum()
    if total_W < 1e-12:
        raise ValueError("Total wage income is zero; cannot run Fisher market.")
    m_norm = W_active / total_W      # normalised budgets for active classes

    # ------------------------------------------------------------------ #
    #  Step 4 – Fisher market on the reduced economy                       #
    # ------------------------------------------------------------------ #
    U_sub = U[np.ix_(I, J)]          # (|I| × |J|) utility submatrix
    q_sub = q[J]

    prices_sub, X_money_sub, X_units_sub, bpb_sub = solve_fisher(
        U_sub, q_sub, m_norm
    )

    # De-normalise prices back to absolute scale:
    #   normalised Fisher market has sum(budgets)=1, so p·q_sub = 1
    #   absolute scale has sum(budgets)=total_W, so p·q_sub = total_W
    prices_abs = prices_sub * total_W

    # ------------------------------------------------------------------ #
    #  Step 5 – Update inactive-good prices  (bang-per-buck rule, fm.m)   #
    # ------------------------------------------------------------------ #
    p_new = p.copy()
    p_new[J] = prices_abs                       # active goods: Fisher output

    # For each active class, compute its max bang-per-buck at NEW prices
    #   ratio[i] = max_j { U[I[i], J[j]] / p_new[J[j]] }
    ratio = np.zeros(len(I))
    for idx_i, i in enumerate(I):
        bpb_i = U[i, J] / prices_abs            # bang-per-buck for active goods
        ratio[idx_i] = bpb_i.max()

    # Inactive goods: price = max over active classes of U[i,j] / ratio[i]
    J_set = set(J)
    for j in range(n):
        if j not in J_set:
            candidates = []
            for idx_i, i in enumerate(I):
                if ratio[idx_i] > 1e-12:
                    candidates.append(U[i, j] / ratio[idx_i])
            if candidates:
                p_new[j] = max(candidates)

    # ------------------------------------------------------------------ #
    #  Step 6 – Full allocation matrix                                     #
    # ------------------------------------------------------------------ #
    X_units = np.zeros((m, n))
    X_units[np.ix_(I, J)] = X_units_sub

    return p_new, q, W, X_units, I, J
