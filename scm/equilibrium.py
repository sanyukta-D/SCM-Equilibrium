"""
scm_equilibrium.py - Tâtonnement iterator for the SCM

Iterates scm_round(T, U, Y, p) until prices converge:
    p_{t+1} = scm_round(T, U, Y, p_t)

Mirrors the pattern in equilibrium.m / FeigningU.m.

Also includes cycle detection (the "toggling state" phenomenon documented
in section G.2.2 of the market code documentation).
"""

import numpy as np
from .scm_round import scm_round


def compute_equilibrium(T, U, Y, p_init,
                         max_iter=200, tol=1e-7,
                         cycle_window=4):
    """
    Tâtonnement loop: iterate scm_round until convergence.

    Parameters
    ----------
    T          : (m, n)  Technology matrix.
    U          : (m, n)  Utility matrix (linear).
    Y          : (m,)    Labour availability.
    p_init     : (n,)    Initial price vector.
    max_iter   : int     Maximum iterations before giving up.
    tol        : float   Convergence threshold on max |p_new - p|.
    cycle_window: int    How many recent price vectors to check for cycling.

    Returns
    -------
    result : dict with keys
        'p'          (n,)    Equilibrium (or last) prices.
        'q'          (n,)    Production quantities.
        'W'          (m,)    Absolute wage income per class.
        'X'          (m,n)   Goods allocation (units).
        'I'                  Active labour indices.
        'J'                  Active goods indices.
        'status'     str     'converged' | 'cycling' | 'max_iter'
        'n_iter'     int     Number of iterations run.
        'history'    list    Per-iteration dict {p, W, q, I, J, delta}.
    """
    T      = np.array(T,      dtype=float)
    U      = np.array(U,      dtype=float)
    Y      = np.array(Y,      dtype=float).ravel()
    p      = np.array(p_init, dtype=float).ravel()

    history = []
    recent_prices = []          # ring buffer for cycle detection

    for it in range(max_iter):
        p_new, q, W, X, I, J = scm_round(T, U, Y, p)

        delta = float(np.max(np.abs(p_new - p)))
        history.append({
            'iter': it,
            'p':    p.copy(),
            'W':    W.copy(),
            'q':    q.copy(),
            'I':    I.copy(),
            'J':    J.copy(),
            'delta': delta
        })

        # ---- convergence check ----
        if delta < tol:
            return _pack(p_new, q, W, X, I, J,
                         'converged', it + 1, history)

        # ---- cycle detection: compare p_new to recent vectors ----
        for prev in recent_prices:
            if np.max(np.abs(p_new - prev)) < tol * 10:
                # Found a repeat – we're cycling
                return _pack(p_new, q, W, X, I, J,
                             'cycling', it + 1, history)

        recent_prices.append(p_new.copy())
        if len(recent_prices) > cycle_window:
            recent_prices.pop(0)

        p = p_new

    # Exhausted iterations
    p_new, q, W, X, I, J = scm_round(T, U, Y, p)
    return _pack(p_new, q, W, X, I, J, 'max_iter', max_iter, history)


def _pack(p, q, W, X, I, J, status, n_iter, history):
    return {
        'p':       p,
        'q':       q,
        'W':       W,
        'X':       X,
        'I':       I,
        'J':       J,
        'status':  status,
        'n_iter':  n_iter,
        'history': history
    }


# -------------------------------------------------------------------------
# Convenience: print a summary of the equilibrium result
# -------------------------------------------------------------------------
def print_equilibrium(result, label=""):
    sep = "─" * 50
    print(sep)
    if label:
        print(f"  {label}")
    print(f"  Status   : {result['status']}  ({result['n_iter']} iterations)")
    print(f"  Prices p : {np.round(result['p'], 5)}")
    print(f"  Production q: {np.round(result['q'], 4)}")
    print(f"  Wages W  : {np.round(result['W'], 5)}")
    print(f"  Active goods  J = {result['J']}")
    print(f"  Active labour I = {result['I']}")
    print(f"  Money conservation: p·q = {result['p'] @ result['q']:.6f},  "
          f"ΣW = {result['W'].sum():.6f}")
    print(sep)
