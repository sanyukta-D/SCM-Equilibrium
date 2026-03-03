"""
scm_round_plc.py - One full production-consumption round of the SCM with PLC utilities

Extends scm_round.py for Piecewise-Linear-Concave (PLC) utility functions.

Mirrors plcmarket.m / plcm.m for the 2-segment case:

  1. Production LP  →  q (quantities), I (active labour), J (active goods)
  2. Wages          →  w[I] = p[J] @ inv(T[I,J]),  W[I] = w[I] * Y[I]
  3. Normalise      →  m = W / sum(W)               (relative budgets, sum=1)
  4. PLC Fisher     →  new prices p_new[J], allocations X1, X2
  5. Price update for inactive goods (bang-per-buck rule, segment 1)
  6. Return         →  p_new, q, W, X1_units, X2_units, I, J

The L matrix gives per-class per-good first-segment capacity limits.  Its shape
is (m, n) with absolute units (same scale as the production quantities).
"""

import numpy as np
from .production      import solve_production, wages_from_prices
from .fisher_market_plc import solve_fisher_plc


def scm_round_plc(T, U1, U2, L1_full, Y, p, tol_active=1e-5):
    """
    One round of the SCM with PLC utilities: prices in → prices out.

    Parameters
    ----------
    T        : (m, n)   Technology matrix.
    U1       : (m, n)   Segment-1 utility matrix.
    U2       : (m, n)   Segment-2 utility matrix. U2[i,j] ≤ U1[i,j].
    L1_full  : (m, n)   Segment-1 capacity limits (absolute units).
    Y        : (m,)     Labour availability.
    p        : (n,)     Current price vector.

    Returns
    -------
    p_new    : (n,)     Updated price vector.
    q        : (n,)     Production quantities.
    W        : (m,)     Absolute wage income per class.
    X1_units : (m, n)   Segment-1 goods allocation (units).
    X2_units : (m, n)   Segment-2 goods allocation (units).
    I        : array    Indices of active labour classes.
    J        : array    Indices of active goods.
    """
    T      = np.array(T,      dtype=float)
    U1     = np.array(U1,     dtype=float)
    U2     = np.array(U2,     dtype=float)
    L1_full= np.array(L1_full,dtype=float)
    Y      = np.array(Y,      dtype=float).ravel()
    p      = np.array(p,      dtype=float).ravel()
    m, n   = T.shape

    # ------------------------------------------------------------------ #
    #  Step 1 – Production LP                                              #
    # ------------------------------------------------------------------ #
    q, w_lp, wages_lp, J, I, _revenue = solve_production(T, Y, p, tol_active)

    if len(J) == 0:
        raise ValueError("Production LP produced no active goods.")
    if len(I) == 0:
        raise ValueError("No binding labour constraints.")

    # ------------------------------------------------------------------ #
    #  Step 2 – Wages from active submatrix                               #
    # ------------------------------------------------------------------ #
    T_sub = T[np.ix_(I, J)]
    W     = np.zeros(m)
    W[I]  = wages_from_prices(T_sub, p[J], Y[I])

    # ------------------------------------------------------------------ #
    #  Step 3 – Normalise budgets                                          #
    # ------------------------------------------------------------------ #
    total_W = W[I].sum()
    if total_W < 1e-12:
        raise ValueError("Total wage income is zero.")
    m_norm = W[I] / total_W

    # ------------------------------------------------------------------ #
    #  Step 4 – PLC Fisher market on the active sub-economy               #
    # ------------------------------------------------------------------ #
    U1_sub = U1[np.ix_(I, J)]
    U2_sub = U2[np.ix_(I, J)]
    L1_sub = L1_full[np.ix_(I, J)]
    q_sub  = q[J]

    prices_sub, X_money_sub, X1_sub, X2_sub, bpb1_sub, bpb2_sub = solve_fisher_plc(
        U1_sub, U2_sub, L1_sub, q_sub, m_norm
    )

    # De-normalise prices back to absolute scale
    prices_abs = prices_sub * total_W

    # ------------------------------------------------------------------ #
    #  Step 5 – Update inactive-good prices (segment-1 BPB rule)          #
    # ------------------------------------------------------------------ #
    p_new      = p.copy()
    p_new[J]   = prices_abs

    # For each active class, compute best marginal BPB at new prices
    ratio = np.zeros(len(I))
    for idx_i, i in enumerate(I):
        bpb_seg1 = U1[i, J] / prices_abs          # segment-1 BPB for active goods
        bpb_seg2 = U2[i, J] / prices_abs          # segment-2 BPB
        # best available is the higher of the two (seg1 if not yet at limit)
        ratio[idx_i] = bpb_seg1.max()

    J_set = set(J.tolist())
    for j in range(n):
        if j not in J_set:
            candidates = []
            for idx_i, i in enumerate(I):
                if ratio[idx_i] > 1e-12:
                    candidates.append(U1[i, j] / ratio[idx_i])
            if candidates:
                p_new[j] = max(candidates)

    # ------------------------------------------------------------------ #
    #  Step 6 – Full allocation matrices                                   #
    # ------------------------------------------------------------------ #
    X1_units = np.zeros((m, n))
    X2_units = np.zeros((m, n))
    X1_units[np.ix_(I, J)] = X1_sub
    X2_units[np.ix_(I, J)] = X2_sub

    return p_new, q, W, X1_units, X2_units, I, J
