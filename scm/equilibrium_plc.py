"""
scm_equilibrium_plc.py  –  Tâtonnement iterator for the SCM with PLC utilities

Iterates scm_round_plc(T, U1, U2, L1, Y, p) until prices converge.
Mirrors the pattern of plcequilibrium.m with cycle detection added.
"""

import numpy as np
from .scm_round_plc import scm_round_plc


def compute_equilibrium_plc(T, U1, U2, L1, Y, p_init,
                             max_iter=200, tol=1e-7,
                             cycle_window=4):
    """
    Tâtonnement loop for the PLC SCM.

    Parameters
    ----------
    T          : (m, n)  Technology matrix.
    U1         : (m, n)  Segment-1 utility matrix.
    U2         : (m, n)  Segment-2 utility matrix.
    L1         : (m, n)  Segment-1 capacity limits (absolute units).
    Y          : (m,)    Labour availability.
    p_init     : (n,)    Initial price vector.
    max_iter   : int     Maximum iterations.
    tol        : float   Convergence threshold on max |p_new − p|.
    cycle_window: int    History window for cycle detection.

    Returns
    -------
    result : dict with keys
        'p'          (n,)    Equilibrium (or last) prices.
        'q'          (n,)    Production quantities.
        'W'          (m,)    Absolute wage income per class.
        'X1'         (m,n)   Segment-1 allocation (units).
        'X2'         (m,n)   Segment-2 allocation (units).
        'X'          (m,n)   Total allocation  X = X1 + X2.
        'I'                  Active labour indices.
        'J'                  Active goods indices.
        'status'     str     'converged' | 'cycling' | 'max_iter'
        'n_iter'     int     Number of iterations run.
        'history'    list    Per-iteration dicts.
    """
    T   = np.array(T,      dtype=float)
    U1  = np.array(U1,     dtype=float)
    U2  = np.array(U2,     dtype=float)
    L1  = np.array(L1,     dtype=float)
    Y   = np.array(Y,      dtype=float).ravel()
    p   = np.array(p_init, dtype=float).ravel()

    history      = []
    recent_prices = []

    for it in range(max_iter):
        p_new, q, W, X1, X2, I, J = scm_round_plc(T, U1, U2, L1, Y, p)

        delta = float(np.max(np.abs(p_new - p)))
        history.append({
            'iter': it, 'p': p.copy(), 'W': W.copy(),
            'q': q.copy(), 'I': I.copy(), 'J': J.copy(), 'delta': delta
        })

        if delta < tol:
            return _pack_plc(p_new, q, W, X1, X2, I, J,
                             'converged', it + 1, history)

        for prev in recent_prices:
            if np.max(np.abs(p_new - prev)) < tol * 10:
                return _pack_plc(p_new, q, W, X1, X2, I, J,
                                 'cycling', it + 1, history)

        recent_prices.append(p_new.copy())
        if len(recent_prices) > cycle_window:
            recent_prices.pop(0)

        p = p_new

    p_new, q, W, X1, X2, I, J = scm_round_plc(T, U1, U2, L1, Y, p)
    return _pack_plc(p_new, q, W, X1, X2, I, J, 'max_iter', max_iter, history)


def _pack_plc(p, q, W, X1, X2, I, J, status, n_iter, history):
    return {
        'p': p, 'q': q, 'W': W,
        'X1': X1, 'X2': X2, 'X': X1 + X2,
        'I': I, 'J': J,
        'status': status, 'n_iter': n_iter, 'history': history
    }


def print_equilibrium_plc(result, label=""):
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
    pq  = result['p'] @ result['q']
    sw  = result['W'].sum()
    print(f"  Money conservation: p·q = {pq:.6f},  ΣW = {sw:.6f}")
    print(sep)
