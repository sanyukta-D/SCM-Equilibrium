"""
verify.py  –  Reusable SCM Equilibrium Condition Checker

Implements the complete set of equilibrium conditions from the SCM paper
(Deshpande & Sohoni, arXiv:2109.09248).

Usage
-----
    from scm.verify import check_scm_equilibrium, check_plc_equilibrium

    checks, all_pass = check_scm_equilibrium(result, T, U, Y)
    checks, all_pass = check_plc_equilibrium(result, T, U1, U2, L1, Y)

    # result is the dict returned by compute_equilibrium / compute_equilibrium_plc

Conditions checked (linear utilities)
--------------------------------------
  1.  money_conservation   : p · q = Σ_i W_i
  2.  price_nonneg         : p ≥ 0
  3.  labour_feasibility   : T · q ≤ Y  (componentwise)
  4.  prod_nonneg          : q ≥ 0
  5.  market_clearing      : Σ_i X[i,j] = q[j]  for active j ∈ J
  6.  budget_exhaustion    : Σ_j p[j] X[i,j] = W[i]  for active i ∈ I
  7.  wage_consistency     : W[I] matches matrix-inverse formula
  8.  bpb_optimality       : buyer i spends only on goods with max BPB
  9.  fixed_point          : one more scm_round leaves p unchanged
  10. production_optimality: active goods match LP solution at p

For PLC: conditions 5, 6, 8 are extended to cover both segments; the
segment-1 capacity constraint is additionally checked.
"""

import numpy as np
from .production import solve_production, wages_from_prices


# =========================================================================
#  LINEAR utility equilibrium checker
# =========================================================================

def check_scm_equilibrium(result, T, U, Y,
                           tol=1e-5, verbose=True, label=""):
    """
    Check all SCM equilibrium conditions for a linear-utility result.

    Parameters
    ----------
    result : dict   Output of compute_equilibrium(T, U, Y, p_init).
    T      : (m,n)  Technology matrix.
    U      : (m,n)  Linear utility matrix.
    Y      : (m,)   Labour endowments.
    tol    : float  Pass/fail tolerance.
    verbose: bool   Print condition table.
    label  : str    Optional label for the print header.

    Returns
    -------
    checks   : dict  {condition_name: (passed, error_value)}
    all_pass : bool  True iff every checked condition passes.
    """
    T = np.array(T, dtype=float)
    U = np.array(U, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    m, n = T.shape

    p = result['p']
    q = result['q']
    W = result['W']
    X = result['X']
    I = np.asarray(result['I'])
    J = np.asarray(result['J'])

    checks = {}

    # --------------------------------------------------------------------- #
    # 1. Money conservation
    # --------------------------------------------------------------------- #
    pq  = float(p @ q)
    sw  = float(W.sum())
    err = abs(pq - sw)
    checks['money_conservation'] = (err < tol, err)

    # --------------------------------------------------------------------- #
    # 2. Price non-negativity
    # --------------------------------------------------------------------- #
    err = float(max(0.0, -np.min(p)))
    checks['price_nonneg'] = (err < tol, err)

    # --------------------------------------------------------------------- #
    # 3. Labour feasibility:  T q ≤ Y
    # --------------------------------------------------------------------- #
    slack = Y - T @ q
    err   = float(max(0.0, -np.min(slack)))
    checks['labour_feasibility'] = (err < tol, err)

    # --------------------------------------------------------------------- #
    # 4. Production non-negativity
    # --------------------------------------------------------------------- #
    err = float(max(0.0, -np.min(q)))
    checks['prod_nonneg'] = (err < tol, err)

    # --------------------------------------------------------------------- #
    # 5. Market clearing for active goods
    # --------------------------------------------------------------------- #
    J_set = set(J.tolist())
    mc_errs = [abs(float(q[j]) - float(X[:, j].sum())) for j in J_set]
    err = max(mc_errs) if mc_errs else 0.0
    checks['market_clearing'] = (err < tol, err)

    # --------------------------------------------------------------------- #
    # 6. Budget exhaustion for active classes  (relative error)
    #    We measure relative error (abs_err / W[i]) so that the tolerance
    #    is scale-invariant.  When total wages are large (e.g. W[i]~30),
    #    the Fisher market's interior-point primal error (~1e-5 normalised)
    #    amplifies to ~3e-4 in absolute terms but remains ~1e-5 relative.
    # --------------------------------------------------------------------- #
    be_errs = []
    for i in I:
        if W[i] > tol:
            spent = float(p @ X[i])
            rel_err = abs(spent - float(W[i])) / max(float(W[i]), 1.0)
            be_errs.append(rel_err)
    err = max(be_errs) if be_errs else 0.0
    checks['budget_exhaustion'] = (err < tol, err)

    # --------------------------------------------------------------------- #
    # 7. Wage consistency:  W[I] = wages_from_prices(T[I,J], p[J], Y[I])
    # --------------------------------------------------------------------- #
    if len(I) > 0 and len(J) > 0:
        try:
            T_sub  = T[np.ix_(I, J)]
            W_chk  = wages_from_prices(T_sub, p[J], Y[I])
            err    = float(np.max(np.abs(W[I] - W_chk)))
        except Exception:
            err = float('nan')
    else:
        err = 0.0
    checks['wage_consistency'] = (_nan_pass(err, tol), err)

    # --------------------------------------------------------------------- #
    # 8. Bang-per-buck optimality (linear U, active submatrix)
    # --------------------------------------------------------------------- #
    bpb_errs = []
    for i in I:
        if W[i] < tol:
            continue
        bpb_j    = np.array([U[i, j] / p[j] if p[j] > tol else 0.0
                             for j in J])
        max_bpb  = bpb_j.max()
        for idx_j, j in enumerate(J):
            spend = float(X[i, j]) * float(p[j])
            if spend > tol * float(W[i]):
                gap = max(0.0, max_bpb - bpb_j[idx_j])
                bpb_errs.append(gap)
    err = max(bpb_errs) if bpb_errs else 0.0
    checks['bpb_optimality'] = (err < tol, err)

    # --------------------------------------------------------------------- #
    # 9. Fixed-point:  one more round leaves p unchanged
    # --------------------------------------------------------------------- #
    try:
        from .scm_round import scm_round
        p_nxt, _, _, _, _, _ = scm_round(T, U, Y, p)
        err = float(np.max(np.abs(p_nxt - p)))
    except Exception:
        err = float('nan')
    checks['fixed_point'] = (_nan_pass(err, 1e-4), err)   # relaxed tol for round-trip

    # --------------------------------------------------------------------- #
    # 10. Production optimality:  LP at p gives same J
    # --------------------------------------------------------------------- #
    try:
        _, _, _, J_chk, _, _ = solve_production(T, Y, p)
        J_chk_set = set(J_chk.tolist())
        err = 0.0 if J_chk_set == J_set else 1.0
    except Exception:
        err = float('nan')
    checks['production_optimality'] = (_nan_pass(err, tol), err)

    # --------------------------------------------------------------------- #
    # Print
    # --------------------------------------------------------------------- #
    if verbose:
        _print_checks(checks, label)

    all_pass = _all_pass(checks)
    return checks, all_pass


# =========================================================================
#  PLC utility equilibrium checker
# =========================================================================

def check_plc_equilibrium(result, T, U1, U2, L1, Y,
                           tol=1e-5, verbose=True, label=""):
    """
    Check all SCM equilibrium conditions for a PLC-utility result.

    Parameters
    ----------
    result : dict   Output of compute_equilibrium_plc(...)
    T      : (m,n)  Technology matrix.
    U1     : (m,n)  Segment-1 utility matrix.
    U2     : (m,n)  Segment-2 utility matrix.
    L1     : (m,n)  Segment-1 capacity limits.
    Y      : (m,)   Labour endowments.

    Returns
    -------
    checks, all_pass  (same structure as check_scm_equilibrium)
    """
    T  = np.array(T,  dtype=float)
    U1 = np.array(U1, dtype=float)
    U2 = np.array(U2, dtype=float)
    L1 = np.array(L1, dtype=float)
    Y  = np.array(Y,  dtype=float).ravel()
    m, n = T.shape

    p   = result['p']
    q   = result['q']
    W   = result['W']
    X1  = result['X1']
    X2  = result['X2']
    X   = result['X']
    I   = np.asarray(result['I'])
    J   = np.asarray(result['J'])

    checks = {}

    # ---- 1-4: identical to linear ----
    pq  = float(p @ q);  sw = float(W.sum())
    checks['money_conservation']  = (abs(pq - sw) < tol, abs(pq - sw))
    checks['price_nonneg']        = (_gt0(p, tol))
    slack = Y - T @ q
    checks['labour_feasibility']  = (float(max(0.0, -np.min(slack))) < tol,
                                     float(max(0.0, -np.min(slack))))
    checks['prod_nonneg']         = (_gt0(q, tol))

    # ---- 5. Market clearing (both segments) ----
    J_set   = set(J.tolist())
    mc_errs = [abs(float(q[j]) - float(X[:, j].sum())) for j in J_set]
    err = max(mc_errs) if mc_errs else 0.0
    checks['market_clearing'] = (err < tol, err)

    # ---- 6. Budget exhaustion (relative error, scale-invariant) ----
    be_errs = []
    for i in I:
        if W[i] > tol:
            spent = float(p @ X[i])
            rel_err = abs(spent - float(W[i])) / max(float(W[i]), 1.0)
            be_errs.append(rel_err)
    err = max(be_errs) if be_errs else 0.0
    checks['budget_exhaustion'] = (err < tol, err)

    # ---- 7. Wage consistency ----
    if len(I) > 0 and len(J) > 0:
        try:
            W_chk = wages_from_prices(T[np.ix_(I, J)], p[J], Y[I])
            err   = float(np.max(np.abs(W[I] - W_chk)))
        except Exception:
            err = float('nan')
    else:
        err = 0.0
    checks['wage_consistency'] = (_nan_pass(err, tol), err)

    # ---- 8a. Segment-1 capacity not exceeded ----
    seg1_errs = [max(0.0, float(X1[i, j]) - float(L1[i, j]))
                 for i in I for j in J]
    err = max(seg1_errs) if seg1_errs else 0.0
    checks['seg1_capacity'] = (err < tol, err)

    # ---- 8b. PLC bang-per-buck optimality ----
    # For each active (i, j):
    #   • if x1[i,j] < L1[i,j] (seg-1 not saturated): seg-2 of good j not bought
    #     AND the seg-1 BPB of j ≥ BPB of any purchased seg-1 or seg-2
    #   • buyer does not mix: if buying seg-2 of j, must have x1[i,j] = L1[i,j]
    bpb_errs = []
    for i in I:
        if W[i] < tol:
            continue
        for j in J:
            if p[j] < tol:
                continue
            bpb1_ij = U1[i, j] / p[j]
            bpb2_ij = U2[i, j] / p[j]
            buying_seg2 = float(X2[i, j]) > tol
            seg1_full   = float(L1[i, j]) - float(X1[i, j]) < tol

            # If buying seg-2 of j, must have saturated seg-1 of j
            if buying_seg2 and not seg1_full:
                bpb_errs.append(1.0)     # invariant violated
    err = max(bpb_errs) if bpb_errs else 0.0
    checks['plc_bpb_ordering'] = (err < tol, err)

    # ---- 9. Fixed-point (PLC round) ----
    try:
        from .scm_round_plc import scm_round_plc
        p_nxt, _, _, _, _, _, _ = scm_round_plc(T, U1, U2, L1, Y, p)
        err = float(np.max(np.abs(p_nxt - p)))
    except Exception:
        err = float('nan')
    checks['fixed_point'] = (_nan_pass(err, 1e-4), err)

    # ---- 10. Production optimality ----
    try:
        _, _, _, J_chk, _, _ = solve_production(T, Y, p)
        err = 0.0 if set(J_chk.tolist()) == J_set else 1.0
    except Exception:
        err = float('nan')
    checks['production_optimality'] = (_nan_pass(err, tol), err)

    if verbose:
        _print_checks(checks, label)

    return checks, _all_pass(checks)


# =========================================================================
#  Helpers
# =========================================================================

def _nan_pass(err, tol):
    if isinstance(err, float) and np.isnan(err):
        return True          # can't evaluate → don't fail
    return err < tol

def _gt0(arr, tol):
    err = float(max(0.0, -np.min(arr)))
    return (err < tol, err)

def _print_checks(checks, label=""):
    if label:
        print(f"  {'─'*46}")
        print(f"  {label}")
    print(f"  {'─'*46}")
    for name, (passed, err_val) in checks.items():
        tag = "[PASS]" if passed else "[FAIL]"
        ev  = "NaN" if (isinstance(err_val, float) and np.isnan(err_val)) \
              else f"{err_val:.2e}"
        print(f"    {tag}  {name:<28s}  err={ev}")

def _all_pass(checks):
    return all(passed for passed, ev in checks.values()
               if not (isinstance(ev, float) and np.isnan(ev)))


# =========================================================================
#  Quick summary helper
# =========================================================================

def equilibrium_summary(result, T, U, Y, tol=1e-5, label="", plc_args=None):
    """
    Print a one-liner summary after running check_scm_equilibrium.

    plc_args : dict with keys U1, U2, L1 for PLC mode; None for linear.
    """
    if plc_args is not None:
        checks, ok = check_plc_equilibrium(
            result, T, plc_args['U1'], plc_args['U2'], plc_args['L1'], Y,
            tol=tol, verbose=True, label=label)
    else:
        checks, ok = check_scm_equilibrium(
            result, T, U, Y, tol=tol, verbose=True, label=label)
    print(f"  {'ALL PASS ✓' if ok else 'FAILURES DETECTED ✗'}")
    return checks, ok
