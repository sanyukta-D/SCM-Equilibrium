"""
fisher_market_plc.py  –  Fisher Market with Piecewise-Linear-Concave (PLC) utilities
                          via the Eisenberg-Gale Convex Program

Theory (Paper §G.1)
-------------------
For k = 2 segments the PLC utility of class i consuming x_j units of good j is:

    V_ij(x_j) = U1[i,j] · min(x_j, L[i,j])  +  U2[i,j] · max(x_j − L[i,j], 0)

with U1[i,j] ≥ U2[i,j] ≥ 0  (concavity / diminishing marginal utility).

Equivalently, split each allocation into two variables:

    x_j  =  x1[i,j] + x2[i,j],    x1[i,j] ≤ L[i,j]

PLC Eisenberg-Gale program:
    max   Σ_i  b_i · log( Σ_j [ U1[i,j]·x1[i,j]  +  U2[i,j]·x2[i,j] ] )
    s.t.  Σ_i (x1[i,j] + x2[i,j]) = q[j]    ∀j         (dual → price p_j)
          x1[i,j] ≤ L[i,j]                   ∀(i,j)
          x1[i,j], x2[i,j] ≥ 0

At optimum the standard Fisher-market properties hold:
  •  Budget exhaustion :  Σ_j p_j (x1[i,j]+x2[i,j]) = b_i  for active buyers
  •  Money conservation:  p · q = Σ_i b_i
  •  BPB optimality    :  buyer i fills segment 1 of good j before segment 2,
                          and spends only on goods with maximum marginal BPB.
"""

import numpy as np
import cvxpy as cp


def solve_fisher_plc(U1, U2, L1, q, budgets, tol=1e-8):
    """
    Solve a 2-segment PLC Fisher market.

    Parameters
    ----------
    U1      : array (m, n)  Utility / unit in segment 1.  U1[i,j] ≥ U2[i,j] ≥ 0.
    U2      : array (m, n)  Utility / unit in segment 2.
    L1      : array (m, n)  Segment-1 upper bound: max units of good j that class i
                            can consume at the higher (segment-1) rate.
                            L1 is in *absolute units* (same scale as q).
    q       : array (n,)    Total quantity of each good available.
    budgets : array (m,)    Budget (wage income) of each class.

    Returns
    -------
    prices   : (n,)    Equilibrium prices.  p · q = Σ budgets.
    X_money  : (m, n)  Total money allocation.  Row sums ≈ budgets.
    X1_units : (m, n)  Units consumed in segment 1.  X1_units ≤ L1 element-wise.
    X2_units : (m, n)  Units consumed in segment 2.
    bpb1     : (m, n)  Marginal bang-per-buck, segment 1: U1 / prices.
    bpb2     : (m, n)  Marginal bang-per-buck, segment 2: U2 / prices.
    """
    U1      = np.array(U1,      dtype=float)
    U2      = np.array(U2,      dtype=float)
    L1      = np.array(L1,      dtype=float)
    q       = np.array(q,       dtype=float).ravel()
    budgets = np.array(budgets, dtype=float).ravel()
    m, n    = U1.shape

    if U1.shape != U2.shape or U1.shape != L1.shape:
        raise ValueError("U1, U2, L1 must all have shape (m, n).")

    # ------------------------------------------------------------------ #
    #  Active buyers only (positive budget)                                #
    # ------------------------------------------------------------------ #
    active = budgets > tol
    if not np.any(active):
        raise ValueError("All budgets are zero – market cannot be solved.")

    U1_a = U1[active];   U2_a = U2[active]
    L1_a = L1[active];   b_a  = budgets[active]
    m_a  = int(np.sum(active))

    # ------------------------------------------------------------------ #
    #  cvxpy variables                                                     #
    # ------------------------------------------------------------------ #
    X1 = cp.Variable((m_a, n), nonneg=True)   # segment-1 units
    X2 = cp.Variable((m_a, n), nonneg=True)   # segment-2 units

    # Utility value per active class
    v = [cp.sum(cp.multiply(U1_a[i], X1[i]) + cp.multiply(U2_a[i], X2[i]))
         for i in range(m_a)]

    # EG objective
    obj = cp.Maximize(sum(b_a[i] * cp.log(v[i]) for i in range(m_a)))

    # Market-clearing constraints (duals become prices)
    mc = [cp.sum(X1[:, j] + X2[:, j]) == q[j] for j in range(n)]

    # Segment-1 capacity constraints
    seg_caps = [X1[i, j] <= L1_a[i, j]
                for i in range(m_a) for j in range(n)]

    prob = cp.Problem(obj, mc + seg_caps)
    prob.solve(solver=cp.CLARABEL, verbose=False,
               tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)

    if X1.value is None:
        # Fallback to SCS with tight tolerance
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-9)

    if X1.value is None:
        raise ValueError(
            f"PLC Fisher market solve failed (status: {prob.status}).\n"
            "Check that U1, U2, L1, q, budgets are non-negative and consistent."
        )

    X1_a = np.maximum(X1.value, 0.0)
    X2_a = np.maximum(X2.value, 0.0)

    # ------------------------------------------------------------------ #
    #  Prices from dual variables of market-clearing constraints.          #
    #                                                                      #
    #  At the EG optimum the dual of the market-clearing equality          #
    #    Σ_i (x1[i,j] + x2[i,j]) = q_j                                   #
    #  is exactly the Walrasian shadow price for good j.                   #
    #  We enforce money conservation (p·q = Σ budgets) to bring the       #
    #  numerically recovered duals onto the exact Walrasian price scale.   #
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
    #  Reconstruct full (m × n) matrices                                   #
    # ------------------------------------------------------------------ #
    X1_units = np.zeros((m, n));  X1_units[active] = X1_a
    X2_units = np.zeros((m, n));  X2_units[active] = X2_a

    X_money = np.zeros((m, n))
    for j in range(n):
        X_money[:, j] = (X1_units[:, j] + X2_units[:, j]) * prices[j]

    bpb1 = np.zeros((m, n))
    bpb2 = np.zeros((m, n))
    for j in range(n):
        if prices[j] > tol:
            bpb1[:, j] = U1[:, j] / prices[j]
            bpb2[:, j] = U2[:, j] / prices[j]

    return prices, X_money, X1_units, X2_units, bpb1, bpb2


# ---------------------------------------------------------------------------
# Convenience wrapper: accept the 3-D arrays used in the paper notation
# U3d[i, j, 0] = segment-1 utility  ;  U3d[i, j, 1] = segment-2 utility
# L3d[i, j, 0] = segment-1 limit    ;  L3d[i, j, 1] = segment-2 limit (unused)
# ---------------------------------------------------------------------------

def solve_fisher_plc_3d(U3d, L3d, q, budgets, tol=1e-8):
    """
    Convenience wrapper accepting 3-D arrays (m × n × k), k ≥ 2.

    U3d[:, :, 0] = segment-1 utilities
    U3d[:, :, 1] = segment-2 utilities
    L3d[:, :, 0] = segment-1 capacity limits  (absolute units)
    L3d[:, :, 1] = segment-2 limits (ignored – last segment runs to end of supply)
    """
    U3d = np.array(U3d, dtype=float)
    L3d = np.array(L3d, dtype=float)
    if U3d.ndim != 3 or U3d.shape[2] < 2:
        raise ValueError("U3d must be (m, n, k) with k ≥ 2.")
    return solve_fisher_plc(U3d[:, :, 0], U3d[:, :, 1], L3d[:, :, 0], q, budgets, tol)
