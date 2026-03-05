"""
fisher_market_splc.py  –  Fisher Market with general SPLC (S-piece PLC) utilities
                          via the Eisenberg-Gale Convex Program

Generalises fisher_market_plc.py from k=2 segments to arbitrary k ≥ 1.

Theory
------
For S segments, the PLC utility of class i consuming x_j units of good j is:

    V_ij(x) = Σ_{s=1}^{S}  U[i,j,s-1] · (amount consumed in segment s)

where consumption fills segments in order: segment 1 first (up to capacity
L[i,j,0]), then segment 2 (up to capacity L[i,j,1]), etc. The last segment
has no capacity limit (runs to the end of supply).

Concavity requires:  U[i,j,0] ≥ U[i,j,1] ≥ ... ≥ U[i,j,S-1] ≥ 0
(diminishing marginal utility across segments).

SPLC Eisenberg-Gale program:
    max   Σ_i  b_i · log( Σ_j Σ_s U[i,j,s] · x_s[i,j] )
    s.t.  Σ_i Σ_s x_s[i,j] = q[j]       ∀j         (dual → price p_j)
          x_s[i,j] ≤ L[i,j,s]            ∀(i,j,s) for s < S  (capacity)
          x_s[i,j] ≥ 0
"""

import numpy as np
import cvxpy as cp


def solve_fisher_splc(U, L, q, budgets, tol=1e-8):
    """
    Solve an S-segment PLC Fisher market.

    Parameters
    ----------
    U       : array (m, n, S)  Utility per unit in each segment.
              U[i,j,s] ≥ U[i,j,s+1] ≥ 0  (concavity).
    L       : array (m, n, S)  Segment capacity limits.
              L[i,j,s] for s < S-1 = max units in that segment.
              L[i,j,S-1] is ignored (last segment is unbounded).
    q       : array (n,)       Total quantity of each good available.
    budgets : array (m,)       Budget (wage income) of each class.

    Returns
    -------
    prices   : (n,)       Equilibrium prices.  p · q = Σ budgets.
    X_money  : (m, n)     Total money allocation.
    X_units  : (m, n, S)  Units consumed in each segment.
    bpb      : (m, n, S)  Marginal bang-per-buck per segment: U[i,j,s] / p[j].
    """
    U       = np.array(U,       dtype=float)
    L       = np.array(L,       dtype=float)
    q       = np.array(q,       dtype=float).ravel()
    budgets = np.array(budgets, dtype=float).ravel()

    if U.ndim != 3:
        raise ValueError("U must be 3-D: (m, n, S).")
    m, n, S = U.shape
    if L.shape != (m, n, S):
        raise ValueError(f"L must have shape {(m, n, S)}, got {L.shape}.")

    # ------------------------------------------------------------------ #
    #  Active buyers only (positive budget)                                #
    # ------------------------------------------------------------------ #
    active = budgets > tol
    if not np.any(active):
        raise ValueError("All budgets are zero – market cannot be solved.")

    U_a = U[active]
    L_a = L[active]
    b_a = budgets[active]
    m_a = int(np.sum(active))

    # ------------------------------------------------------------------ #
    #  cvxpy variables: X_s[i,j] for each segment s                       #
    # ------------------------------------------------------------------ #
    # X_vars[s] is (m_a, n) for segment s
    X_vars = [cp.Variable((m_a, n), nonneg=True) for _ in range(S)]

    # Utility value per active class
    v = []
    for i in range(m_a):
        v_i = 0
        for s in range(S):
            v_i = v_i + cp.sum(cp.multiply(U_a[i, :, s], X_vars[s][i]))
        v.append(v_i)

    # EG objective
    obj = cp.Maximize(sum(b_a[i] * cp.log(v[i]) for i in range(m_a)))

    # Market-clearing: total across all segments and all buyers = q[j]
    mc = []
    for j in range(n):
        total_j = 0
        for s in range(S):
            total_j = total_j + cp.sum(X_vars[s][:, j])
        mc.append(total_j == q[j])

    # Segment capacity constraints (all segments except last)
    seg_caps = []
    for s in range(S - 1):
        for i in range(m_a):
            for j in range(n):
                seg_caps.append(X_vars[s][i, j] <= L_a[i, j, s])

    prob = cp.Problem(obj, mc + seg_caps)
    prob.solve(solver=cp.CLARABEL, verbose=False,
               tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)

    if X_vars[0].value is None:
        # Fallback to SCS
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-9)

    if X_vars[0].value is None:
        raise ValueError(
            f"SPLC Fisher market solve failed (status: {prob.status}).\n"
            "Check that U, L, q, budgets are non-negative and consistent."
        )

    # Extract solutions
    X_a_all = np.zeros((m_a, n, S))
    for s in range(S):
        X_a_all[:, :, s] = np.maximum(X_vars[s].value, 0.0)

    # ------------------------------------------------------------------ #
    #  Prices from dual variables of market-clearing constraints           #
    # ------------------------------------------------------------------ #
    raw = np.array([c.dual_value for c in mc], dtype=float)
    if raw.mean() < 0:
        raw = -raw
    prices = np.maximum(raw, 0.0)

    # Enforce money conservation: p·q = Σ budgets
    total_budget = float(np.sum(budgets))
    pq = float(prices @ q)
    if pq > tol:
        prices = prices * (total_budget / pq)

    # ------------------------------------------------------------------ #
    #  Reconstruct full (m × n × S) arrays                                #
    # ------------------------------------------------------------------ #
    X_units = np.zeros((m, n, S))
    X_units[active] = X_a_all

    X_money = np.zeros((m, n))
    for j in range(n):
        X_money[:, j] = X_units[:, j, :].sum(axis=1) * prices[j]

    bpb = np.zeros((m, n, S))
    for j in range(n):
        if prices[j] > tol:
            for s in range(S):
                bpb[:, j, s] = U[:, j, s] / prices[j]

    return prices, X_money, X_units, bpb
