"""
solvers.py  –  Alternative equilibrium solvers for the SCM (Task 2)

Provides three additional methods beyond standard tatonnement:

  1. Damped tatonnement  –  p_{t+1} = (1-α)p_t + α·G(p_t), with α sweep
  2. Broyden's method    –  quasi-Newton on F(p) = G(p) - p = 0
  3. Cascading solver    –  tries standard → Broyden → damped, returns best

All methods use price normalisation (mean=1) for stability.

Usage
-----
    from scm.solvers import solve_robust, solve_damped, solve_broyden

    result = solve_robust(T, U, Y, p_init)      # best of all methods
    result = solve_damped(T, U, Y, p_init)       # damped only
    result = solve_broyden(T, U, Y, p_init)      # Broyden only
"""

import numpy as np
from .scm_round import scm_round
from .equilibrium import compute_equilibrium
from .verify import check_scm_equilibrium


def _fp_error(T, U, Y, p):
    """Compute normalised fixed-point error: max|G(p̂) - p̂| where p̂ = p/mean(p)."""
    try:
        p_new, _, _, _, _, _ = scm_round(T, U, Y, p)
        p_n = p / p.mean()
        p_new_n = p_new / p_new.mean()
        return float(np.max(np.abs(p_new_n - p_n)))
    except Exception:
        return float('inf')


def _scm_round_safe(T, U, Y, p):
    """Run scm_round, return (p_new, q, W, X, I, J) or None on failure."""
    try:
        return scm_round(T, U, Y, p)
    except Exception:
        return None


def _pack_result(p, T, U, Y, method, n_iter):
    """Run one final scm_round at p and pack into a standard result dict."""
    p = np.array(p, dtype=float)
    res = _scm_round_safe(T, U, Y, p)
    if res is None:
        m, n = np.array(T).shape
        return {
            'p': p, 'q': np.zeros(n), 'W': np.zeros(m),
            'X': np.zeros((m, n)), 'I': np.array([]), 'J': np.array([]),
            'status': 'failed', 'n_iter': n_iter,
            'fp_error': float('inf'), 'method': method,
            'history': []
        }
    p_new, q, W, X, I, J = res
    fp_err = _fp_error(T, U, Y, p)
    status = 'converged' if fp_err < 1e-5 else 'approximate'
    return {
        'p': p, 'q': q, 'W': W, 'X': X, 'I': I, 'J': J,
        'status': status, 'n_iter': n_iter,
        'fp_error': fp_err, 'method': method,
        'history': []
    }


# =========================================================================
# Method 1: Damped tatonnement
# =========================================================================

def solve_damped(T, U, Y, p_init, tol=1e-6, max_iter=1500,
                 alphas=(0.3, 0.1, 0.05)):
    """
    Damped tatonnement with price normalisation and alpha sweep.

    Tries each alpha in sequence. For each alpha, iterates up to max_iter
    steps. Returns the best solution found across all alphas.

    Parameters
    ----------
    T, U, Y, p_init : array_like
        Economy parameters and initial prices.
    tol : float
        Convergence tolerance on normalised fixed-point error.
    max_iter : int
        Maximum iterations per alpha.
    alphas : tuple of float
        Step sizes to try in order.

    Returns
    -------
    result : dict
        Standard equilibrium result dict with additional keys:
        'fp_error' (float), 'method' (str).
    """
    T = np.array(T, dtype=float)
    U = np.array(U, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()
    n = len(p_init)

    best_err = np.inf
    best_p = p_init.copy()
    total_iters = 0

    for alpha in alphas:
        p = p_init.copy() / p_init.mean()

        for it in range(max_iter):
            total_iters += 1
            res = _scm_round_safe(T, U, Y, p)
            if res is None:
                p = np.maximum(p + np.random.randn(n) * 0.01, 1e-6)
                p = p / p.mean()
                continue

            p_scm = res[0]
            p_scm_n = p_scm / p_scm.mean()
            p_n = p / p.mean()
            err = float(np.max(np.abs(p_scm_n - p_n)))

            if err < best_err:
                best_err = err
                best_p = p_n.copy()

            if err < tol:
                return _pack_result(p_n, T, U, Y,
                                    f'damped(α={alpha})', total_iters)

            p = np.maximum((1 - alpha) * p_n + alpha * p_scm_n, 1e-6)

        # Early exit if this alpha already got close
        if best_err < tol * 10:
            break

    return _pack_result(best_p, T, U, Y,
                        f'damped(best α)', total_iters)


# =========================================================================
# Method 2: Broyden's quasi-Newton
# =========================================================================

def solve_broyden(T, U, Y, p_init, tol=1e-6, max_iter=500):
    """
    Broyden's quasi-Newton method on the fixed-point residual F(p) = G(p) - p.

    Uses scipy.optimize.root with method='broyden1'. Operates on normalised
    prices (mean=1) for stability.

    Parameters
    ----------
    T, U, Y, p_init : array_like
        Economy parameters and initial prices.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum function evaluations.

    Returns
    -------
    result : dict
        Standard equilibrium result dict.
    """
    from scipy.optimize import root

    T = np.array(T, dtype=float)
    U = np.array(U, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p0 = np.array(p_init, dtype=float).ravel()
    n = len(p0)
    p0 = p0 / p0.mean()

    eval_count = [0]

    def F(p):
        eval_count[0] += 1
        p_pos = np.maximum(p, 1e-8)
        p_n = p_pos / p_pos.mean()
        res = _scm_round_safe(T, U, Y, p_n)
        if res is None:
            return np.full(n, 1e10)
        p_new = res[0]
        p_new_n = p_new / p_new.mean()
        return p_new_n - p_n

    try:
        sol = root(F, p0, method='broyden1',
                   options={'maxiter': max_iter, 'fatol': tol, 'xatol': tol})
        p_eq = np.maximum(sol.x, 1e-8)
        p_eq = p_eq / p_eq.mean()
        return _pack_result(p_eq, T, U, Y, 'broyden1', eval_count[0])
    except Exception:
        return _pack_result(p0, T, U, Y, 'broyden1(failed)', eval_count[0])


# =========================================================================
# Method 3: Cascading robust solver
# =========================================================================

def solve_robust(T, U, Y, p_init, tol=1e-6, verbose=False):
    """
    Cascading solver: tries multiple methods and returns the best result.

    Order of attempts:
      1. Standard tatonnement (fast, works for ~50% of economies)
      2. Broyden's method (fast, handles many cycling cases)
      3. Damped tatonnement (slower, most robust single method)

    Returns the result with the lowest fixed-point error.

    Parameters
    ----------
    T, U, Y, p_init : array_like
        Economy parameters and initial prices.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress for each method.

    Returns
    -------
    result : dict
        Standard equilibrium result dict. 'method' key indicates which
        method produced the result.
    """
    T = np.array(T, dtype=float)
    U = np.array(U, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()

    candidates = []

    # --- Attempt 1: standard tatonnement ---
    try:
        res = compute_equilibrium(T, U, Y, p_init, max_iter=200, tol=tol)
        if res['status'] == 'converged':
            fp_err = _fp_error(T, U, Y, res['p'])
            res['fp_error'] = fp_err
            res['method'] = 'standard'
            if verbose:
                print(f"  standard: converged, fp_err={fp_err:.2e}")
            if fp_err < tol:
                return res
            candidates.append(res)
    except Exception:
        if verbose:
            print("  standard: exception")

    # --- Attempt 2: Broyden's method ---
    try:
        res = solve_broyden(T, U, Y, p_init, tol=tol, max_iter=500)
        if verbose:
            print(f"  broyden: fp_err={res['fp_error']:.2e}")
        if res['fp_error'] < tol:
            return res
        candidates.append(res)
    except Exception:
        if verbose:
            print("  broyden: exception")

    # --- Attempt 3: damped tatonnement ---
    try:
        res = solve_damped(T, U, Y, p_init, tol=tol, max_iter=1500)
        if verbose:
            print(f"  damped: fp_err={res['fp_error']:.2e}")
        if res['fp_error'] < tol:
            return res
        candidates.append(res)
    except Exception:
        if verbose:
            print("  damped: exception")

    # --- Return best ---
    if candidates:
        best = min(candidates, key=lambda r: r.get('fp_error', float('inf')))
        return best

    # All failed — return a failure result
    return _pack_result(p_init, T, U, Y, 'all_failed', 0)
