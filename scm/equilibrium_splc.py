"""
equilibrium_splc.py  –  Tâtonnement iterator for the SCM with general SPLC utilities

Supports damped tatonnement with price normalisation for hard economies.
Tracks the best fixed-point solution seen (lowest fp_error) and returns it.
"""

import numpy as np
from .scm_round_splc import scm_round_splc


def compute_equilibrium_splc(T, U, L, Y, p_init,
                              max_iter=200, tol=1e-7,
                              cycle_window=4,
                              damped=False, alpha=0.3,
                              normalise=False):
    """
    Tâtonnement loop for the SPLC SCM.

    Parameters
    ----------
    T          : (m, n)      Technology matrix.
    U          : (m, n, S)   Utility per unit per segment.
    L          : (m, n, S)   Segment capacity limits.
    Y          : (m,)        Labour availability.
    p_init     : (n,)        Initial price vector.
    max_iter   : int         Maximum iterations.
    tol        : float       Convergence threshold.
    cycle_window : int       History window for cycle detection.
    damped     : bool        Use damped tatonnement.
    alpha      : float       Step size for damped tatonnement.
    normalise  : bool        Normalise prices to mean=1 each step.

    Returns
    -------
    result : dict with keys
        'p'          (n,)      Equilibrium (or best) prices.
        'q'          (n,)      Production quantities.
        'W'          (m,)      Absolute wage income per class.
        'X_units'    (m,n,S)   Per-segment allocation.
        'X_total'    (m,n)     Total allocation across segments.
        'I', 'J'               Active sets.
        'status'     str       'converged' | 'cycling' | 'max_iter'
        'fp_error'   float     Fixed-point error at returned solution.
        'n_iter'     int
        'history'    list
    """
    T = np.array(T, dtype=float)
    U = np.array(U, dtype=float)
    L = np.array(L, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p = np.array(p_init, dtype=float).ravel()
    n_goods = len(p)
    S = U.shape[2]

    if normalise:
        p = p / p.mean()

    history = []
    recent_prices = []

    # Track best solution seen
    best_fp_err = np.inf
    best_state = None

    for it in range(max_iter):
        try:
            p_new, q, W, X_units, I, J = scm_round_splc(T, U, L, Y, p)
        except (ValueError, np.linalg.LinAlgError):
            p = np.maximum(p + np.random.randn(n_goods) * 0.01, 1e-6)
            if normalise:
                p = p / p.mean()
            continue

        if normalise:
            p_new_n = p_new / p_new.mean()
            p_n = p / p.mean()
        else:
            p_new_n = p_new
            p_n = p

        # Compute fixed-point error (normalised)
        fp_err = float(np.max(np.abs(p_new_n - p_n)))

        if damped:
            p_step = (1 - alpha) * p_n + alpha * p_new_n
            p_step = np.maximum(p_step, 1e-8)
        else:
            p_step = p_new_n

        # Track best
        if fp_err < best_fp_err:
            best_fp_err = fp_err
            best_state = {
                'p': p_n.copy(), 'q': q.copy(), 'W': W.copy(),
                'X_units': X_units.copy(), 'I': I.copy(), 'J': J.copy(),
                'fp_error': fp_err, 'n_iter': it + 1
            }

        history.append({
            'iter': it, 'p': p.copy(), 'W': W.copy(),
            'q': q.copy(), 'I': I.copy(), 'J': J.copy(),
            'delta': fp_err
        })

        if fp_err < tol:
            return _pack(p_n, q, W, X_units, I, J,
                         'converged', fp_err, it + 1, history)

        # Cycle detection
        for prev in recent_prices:
            if np.max(np.abs(p_step - prev)) < tol * 10:
                # Return best seen rather than current cycling point
                if best_state is not None and best_fp_err < fp_err:
                    bs = best_state
                    return _pack(bs['p'], bs['q'], bs['W'], bs['X_units'],
                                 bs['I'], bs['J'],
                                 'cycling', best_fp_err, it + 1, history)
                return _pack(p_step, q, W, X_units, I, J,
                             'cycling', fp_err, it + 1, history)

        recent_prices.append(p_step.copy())
        if len(recent_prices) > cycle_window:
            recent_prices.pop(0)

        p = p_step

    # Return best solution seen
    if best_state is not None:
        bs = best_state
        return _pack(bs['p'], bs['q'], bs['W'], bs['X_units'],
                     bs['I'], bs['J'],
                     'max_iter', best_fp_err, max_iter, history)

    try:
        p_new, q, W, X_units, I, J = scm_round_splc(T, U, L, Y, p)
        fp_err = float(np.max(np.abs(p_new / p_new.mean() - p / p.mean())))
    except (ValueError, np.linalg.LinAlgError):
        return _pack(p, np.zeros(n_goods), np.zeros(T.shape[0]),
                     np.zeros((T.shape[0], n_goods, S)),
                     np.array([]), np.array([]),
                     'diverged', np.inf, max_iter, history)
    return _pack(p, q, W, X_units, I, J, 'max_iter', fp_err, max_iter, history)


def _pack(p, q, W, X_units, I, J, status, fp_error, n_iter, history):
    return {
        'p': p, 'q': q, 'W': W,
        'X_units': X_units,
        'X_total': X_units.sum(axis=2),
        'I': I, 'J': J,
        'status': status, 'fp_error': fp_error,
        'n_iter': n_iter, 'history': history
    }


def print_equilibrium_splc(result, label=""):
    sep = "─" * 55
    print(sep)
    if label:
        print(f"  {label}")
    status = result['status']
    fp_err = result.get('fp_error', None)
    fp_str = f", fp_err={fp_err:.2e}" if fp_err is not None else ""
    print(f"  Status   : {status}  ({result['n_iter']} iters{fp_str})")
    print(f"  Prices p : {np.round(result['p'], 5)}")
    print(f"  Production q: {np.round(result['q'], 4)}")
    print(f"  Wages W  : {np.round(result['W'], 5)}")
    print(f"  Active goods  J = {result['J']}")
    print(f"  Active labour I = {result['I']}")
    pq = result['p'] @ result['q']
    sw = result['W'].sum()
    print(f"  Money conservation: p·q = {pq:.6f},  ΣW = {sw:.6f}")
    S = result['X_units'].shape[2]
    for s in range(S):
        xs = result['X_units'][:, :, s]
        if np.any(xs > 1e-6):
            print(f"  Segment {s+1} allocation (units):")
            print(f"    {np.round(xs, 4)}")
    print(sep)
