"""
scm_round_splc.py - One full production-consumption round of the SCM with
                    general SPLC (S-piece PLC) utilities.

Extends scm_round_plc.py from 2 segments to arbitrary S ≥ 1.

  1. Production LP  →  q, I, J
  2. Wages          →  W[I]
  3. Normalise      →  m = W / sum(W)
  4. SPLC Fisher    →  new prices, allocations X_units (m, n, S)
  5. Price update for inactive goods (segment-1 BPB rule)
  6. Return
"""

import numpy as np
from .production          import solve_production, wages_from_prices
from .fisher_market_splc  import solve_fisher_splc


def scm_round_splc(T, U, L, Y, p, tol_active=1e-5):
    """
    One round of the SCM with SPLC utilities: prices in → prices out.

    Parameters
    ----------
    T    : (m, n)      Technology matrix.
    U    : (m, n, S)   Utility per unit in each segment.
    L    : (m, n, S)   Segment capacity limits (last segment's limit ignored).
    Y    : (m,)        Labour availability.
    p    : (n,)        Current price vector.

    Returns
    -------
    p_new    : (n,)        Updated price vector.
    q        : (n,)        Production quantities.
    W        : (m,)        Absolute wage income per class.
    X_units  : (m, n, S)   Goods allocation per segment (units).
    I        : array       Indices of active labour classes.
    J        : array       Indices of active goods.
    """
    T = np.array(T, dtype=float)
    U = np.array(U, dtype=float)
    L = np.array(L, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p = np.array(p, dtype=float).ravel()
    m, n = T.shape
    S = U.shape[2]

    # Step 1 – Production LP
    q, w_lp, wages_lp, J, I, _revenue = solve_production(T, Y, p, tol_active)

    if len(J) == 0:
        raise ValueError("Production LP produced no active goods.")
    if len(I) == 0:
        raise ValueError("No binding labour constraints.")

    # Step 2 – Wages
    T_sub = T[np.ix_(I, J)]
    W = np.zeros(m)
    W[I] = wages_from_prices(T_sub, p[J], Y[I])

    # Step 3 – Normalise budgets
    total_W = W[I].sum()
    if total_W < 1e-12:
        raise ValueError("Total wage income is zero.")
    m_norm = W[I] / total_W

    # Step 4 – SPLC Fisher market on the active sub-economy
    U_sub = U[np.ix_(I, J)]          # (|I|, |J|, S)
    L_sub = L[np.ix_(I, J)]          # (|I|, |J|, S)
    q_sub = q[J]

    prices_sub, X_money_sub, X_units_sub, bpb_sub = solve_fisher_splc(
        U_sub, L_sub, q_sub, m_norm
    )

    # De-normalise prices
    prices_abs = prices_sub * total_W

    # Step 5 – Update inactive-good prices (segment-1 BPB rule)
    p_new = p.copy()
    p_new[J] = prices_abs

    ratio = np.zeros(len(I))
    for idx_i, i in enumerate(I):
        bpb_seg1 = U[i, J, 0] / prices_abs
        ratio[idx_i] = bpb_seg1.max()

    J_set = set(J.tolist())
    for j in range(n):
        if j not in J_set:
            candidates = []
            for idx_i, i in enumerate(I):
                if ratio[idx_i] > 1e-12:
                    candidates.append(U[i, j, 0] / ratio[idx_i])
            if candidates:
                p_new[j] = max(candidates)

    # Step 6 – Full allocation
    X_units = np.zeros((m, n, S))
    X_units[np.ix_(I, J)] = X_units_sub

    return p_new, q, W, X_units, I, J
