"""
fisher_market.py - Fisher Market via the Eisenberg-Gale Convex Program

Replaces adplc + fisherm.m for the linear-utility case.

Eisenberg-Gale program (linear utilities):
    max   Σ_i  b_i · log( Σ_j  u_ij · x_ij )
    s.t.  Σ_i  x_ij  =  q_j     ∀j      (dual → prices p_j)
          x_ij ≥ 0

At optimum:
  • All budgets are exhausted:  Σ_j p_j x_ij = b_i  (Fisher market property)
  • Bang-per-buck is equalised:  u_ij / p_j = const  for purchased goods
  • Revenue = Σ budgets:         p · q = Σ_i b_i    (money conservation)
"""

import numpy as np
import cvxpy as cp


def solve_fisher(U, q, budgets, tol=1e-8):
    """
    Solve a Fisher market with linear utilities.

    Parameters
    ----------
    U       : array (m, n)  Utility matrix. U[i,j] = utility of class i
                            per unit of good j. Must be non-negative;
                            at least one positive entry per row.
    q       : array (n,)    Quantity of each good available (production).
    budgets : array (m,)    Budget (wage income) of each class.
                            Classes with zero budget are excluded.

    Returns
    -------
    prices   : (n,)    Equilibrium prices.  p · q = Σ budgets.
    X_money  : (m, n)  Money allocation.    X_money[i,j] = spending of class i
                       on good j.  Row sums = budgets.
    X_units  : (m, n)  Goods allocation.    X_units[i,j] = units of good j
                       received by class i.  Column sums = q.
    bpb      : (m, n)  Bang-per-buck ratios = U / prices (for diagnostics).
    """
    U       = np.array(U,       dtype=float)
    q       = np.array(q,       dtype=float).ravel()
    budgets = np.array(budgets, dtype=float).ravel()
    m, n    = U.shape

    # ------------------------------------------------------------------ #
    #  Only buyers with positive budget participate                        #
    # ------------------------------------------------------------------ #
    active = budgets > tol
    if not np.any(active):
        raise ValueError("All budgets are zero – market cannot be solved.")

    U_a   = U[active]
    b_a   = budgets[active]
    m_a   = int(np.sum(active))

    # ------------------------------------------------------------------ #
    #  Eisenberg-Gale convex program                                       #
    # ------------------------------------------------------------------ #
    X_a = cp.Variable((m_a, n), nonneg=True)

    # Utility value for each active class
    v = [U_a[i] @ X_a[i] for i in range(m_a)]

    # Objective: weighted sum of logs
    obj = cp.Maximize(sum(b_a[i] * cp.log(v[i]) for i in range(m_a)))

    # Market clearing: all goods sold (equality so dual = price)
    mc = [cp.sum(X_a[:, j]) == q[j] for j in range(n)]

    prob = cp.Problem(obj, mc)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if X_a.value is None:
        raise ValueError(
            f"Fisher market solve failed (status: {prob.status}). "
            "Check that U has positive entries and q, budgets are positive."
        )

    # X_a is the EG variable x_ij = units of good j allocated to class i.
    # The constraint Σ_i x_ij = q_j is in units, so X_a.value is in units.
    X_a_val = np.maximum(X_a.value, 0.0)   # shape (m_a, n) — UNITS

    # ------------------------------------------------------------------ #
    #  Extract prices from dual variables of market-clearing constraints   #
    # ------------------------------------------------------------------ #
    raw = np.array([c.dual_value for c in mc], dtype=float)

    # CVXPY sign convention for maximisation + equality:
    # dual_value of  Σ_i x_ij = q_j  gives the shadow price p_j (≥ 0).
    # Flip sign if solver returns them negated.
    if raw.mean() < 0:
        raw = -raw
    prices = np.maximum(raw, 0.0)

    # Normalise so that  p · q = Σ budgets  (money conservation).
    # At an exact EG optimum this holds automatically; we enforce it
    # numerically to remove rounding drift.
    total_budget = float(np.sum(budgets))
    pq = float(prices @ q)
    if pq > tol:
        prices = prices * (total_budget / pq)

    # ------------------------------------------------------------------ #
    #  Reconstruct full (m × n) allocation matrices                        #
    # ------------------------------------------------------------------ #
    # X_units: the EG variable x_ij already gives units.
    # X_money: money spent = units × price (column-wise).
    X_units = np.zeros((m, n))
    X_units[active] = X_a_val             # units received — direct from EG

    X_money = np.zeros((m, n))
    for j in range(n):
        X_money[:, j] = X_units[:, j] * prices[j]   # money = units × price

    # Bang-per-buck ratios (for zone / forest diagnostics)
    bpb = np.zeros((m, n))
    for j in range(n):
        if prices[j] > tol:
            bpb[:, j] = U[:, j] / prices[j]

    return prices, X_money, X_units, bpb
