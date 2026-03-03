"""
production.py - Production LP for the Simple Closed Model (SCM)

Solves: max p · q   s.t. T q ≤ Y, q ≥ 0

Mirrors the LP in fm.m (fmincon call) and fisherm.m (inv(T)*Y for square T).
Uses CVXPY / CLARABEL for exact dual variable access (wages).
"""

import numpy as np
import cvxpy as cp


def solve_production(T, Y, p, tol_active=1e-5):
    """
    Solve the production LP.

    Parameters
    ----------
    T   : array (m, n)  Technology matrix. T[i,j] = units of labor class i
                        needed per unit of good j.
    Y   : array (m,)    Labor availability per class.
    p   : array (n,)    Current price vector.
    tol_active : float  Threshold for classifying goods/labor as active.

    Returns
    -------
    q      : (n,)   Production quantities.
    w      : (m,)   Wage rate per unit labor (dual of T q ≤ Y).
                    w[i] = 0 for surplus labor (complementary slackness).
    wages  : (m,)   Total wage income = w * Y.
    J      : (k,)   Indices of active goods  (q[j] > tol_active).
    I      : (r,)   Indices of active labor  (binding constraint: T q ≈ Y).
    revenue: float  Total revenue p · q (= total wages by money conservation).
    """
    T = np.array(T, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p = np.array(p, dtype=float).ravel()
    m, n = T.shape

    # --- Solve LP ---
    q_var = cp.Variable(n, nonneg=True)
    objective = cp.Maximize(p @ q_var)
    constraints = [T @ q_var <= Y]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if q_var.value is None:
        raise ValueError(
            f"Production LP failed (status: {prob.status}). "
            "Check that T, Y, p are well-formed."
        )

    q = np.maximum(q_var.value, 0.0)

    # --- Dual variables = wage rates ---
    # For max p·q s.t. T q ≤ Y, the dual of T q ≤ Y satisfies T^T w ≥ p
    # with equality for active goods (complementary slackness).
    # CLARABEL returns the dual for the inequality as the KKT multiplier.
    w_raw = np.array(constraints[0].dual_value).ravel()
    w = np.maximum(w_raw, 0.0)   # clip tiny negatives from numerics
    wages = w * Y

    # --- Active sets ---
    J = np.where(q > tol_active)[0]                          # positive production
    slack = Y - T @ q
    I = np.where(np.abs(slack) < tol_active * (1.0 + Y))[0] # binding labor

    revenue = float(p @ q)

    return q, w, wages, J, I, revenue


def wages_from_prices(T_active, p_active, Y_active):
    """
    Compute wages directly from the active submatrix:
        w = p_active @ inv(T_active),   W_i = w_i * Y_i

    This matches fm.m's computation exactly for the square invertible case.
    For non-square T_active, falls back to least-squares solve.

    Parameters
    ----------
    T_active : (r, k) submatrix of T restricted to active rows and columns.
    p_active : (k,)   prices for active goods.
    Y_active : (r,)   labor availability for active classes.

    Returns
    -------
    wages : (r,) total wage income for each active class.
    """
    T_active = np.array(T_active, dtype=float)
    p_active = np.array(p_active, dtype=float).ravel()
    Y_active = np.array(Y_active, dtype=float).ravel()
    r, k = T_active.shape

    if r == k:
        # Square: exact solution (matches MATLAB inv(T))
        w_rates = p_active @ np.linalg.inv(T_active)
    else:
        # Non-square: least-squares (T^T w = p)
        w_rates, _, _, _ = np.linalg.lstsq(T_active.T, p_active, rcond=None)

    wages = w_rates * Y_active
    return np.maximum(wages, 0.0)
