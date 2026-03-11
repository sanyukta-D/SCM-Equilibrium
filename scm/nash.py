"""
nash.py  –  Nash equilibrium finder for the Consumer Choice Game (CCG)

Iterative best-response methods for finding Nash equilibrium candidates.
A Nash equilibrium is a utility expression profile U_expressed where no
player can improve their payoff by unilaterally changing their row of
U_expressed.

Usage
-----
    from scm.nash import best_response_direction, nash_iteration

    # Check gradient for one player
    direction, magnitude = best_response_direction(
        T, U_true, U_expressed, Y, p_init, player=0)

    # Iterative best-response to find Nash
    result = nash_iteration(T, U_true, U_init, Y, p_init)
    print(result['converged'], result['payoffs'][-1])
"""

import numpy as np
from .ccg import ccg_payoff, ccg_gradient


def best_response_direction(T, U_true, U_expressed, Y, p_init, player,
                             eps=1e-5, solver='robust', tol=1e-6):
    """
    Compute the best-response gradient direction for one player.

    Returns the direction in U-space that most increases player's payoff,
    considering only that player's own row of U_expressed.

    Parameters
    ----------
    T, U_true, U_expressed, Y, p_init : array_like
        Economy parameters.
    player : int
        Which player.
    eps : float
        Finite difference step.

    Returns
    -------
    direction : (n,) array
        Unit direction in U_expressed[player, :] space for payoff increase.
    magnitude : float
        Gradient magnitude (0 = at local optimum within current zone).
    """
    m, n = np.array(T).shape

    # Full gradient for this player: ∂payoff_player / ∂U_expressed[k, l]
    grad = ccg_gradient(T, U_true, U_expressed, Y, p_init,
                        player=player, eps=eps, solver=solver, tol=tol)

    # Player's own row gradient: how their payoff changes with their own strategy
    own_grad = grad[player, :]  # (n,) — ∂payoff / ∂U_expressed[player, :]

    magnitude = np.linalg.norm(own_grad)
    if magnitude > 1e-10:
        direction = own_grad / magnitude
    else:
        direction = np.zeros(n)

    return direction, magnitude


def best_response_search(T, U_true, U_expressed, Y, p_init, player,
                          n_steps=10, max_step=0.5,
                          solver='robust', tol=1e-6):
    """
    Line search along best-response direction for one player.

    Tries several step sizes along the gradient direction and returns
    the one that maximizes the player's payoff.

    Parameters
    ----------
    T, U_true, U_expressed, Y, p_init : array_like
    player : int
    n_steps : int
        Number of step sizes to try.
    max_step : float
        Maximum step size.

    Returns
    -------
    best_U_row : (n,) array
        Best utility expression found for this player.
    best_payoff : float
    step_size_used : float
    """
    direction, magnitude = best_response_direction(
        T, U_true, U_expressed, Y, p_init, player,
        solver=solver, tol=tol)

    if magnitude < 1e-10:
        # Already at local optimum
        payoffs, _ = ccg_payoff(T, U_true, U_expressed, Y, p_init,
                                 solver=solver, tol=tol)
        return U_expressed[player, :].copy(), payoffs[player], 0.0

    U_expr = np.array(U_expressed, dtype=float)
    best_payoff = -np.inf
    best_row = U_expr[player, :].copy()
    best_step = 0.0

    step_sizes = np.linspace(0, max_step, n_steps + 1)[1:]  # skip 0

    for step in step_sizes:
        U_trial = U_expr.copy()
        U_trial[player, :] += step * direction
        U_trial = np.maximum(U_trial, 0)  # Utilities non-negative

        payoffs, _ = ccg_payoff(T, U_true, U_trial, Y, p_init,
                                 solver=solver, tol=tol)
        if payoffs[player] > best_payoff:
            best_payoff = payoffs[player]
            best_row = U_trial[player, :].copy()
            best_step = step

    return best_row, best_payoff, best_step


def nash_iteration(T, U_true, U_init, Y, p_init,
                   max_iter=50, lr=0.1, tol=1e-4,
                   solver='robust', solver_tol=1e-6, verbose=False):
    """
    Iterative best-response: all players simultaneously update toward
    their best-response utility expression.

    Each iteration:
      1. Compute gradient for each player
      2. Each player takes a step in their gradient direction
      3. Check convergence (max improvement magnitude < tol)

    Parameters
    ----------
    T : (m, n) array
        Technology matrix.
    U_true : (m, n) array
        True utility matrix.
    U_init : (m, n) array
        Starting expressed utility profile.
    Y : (m,) array
        Labour endowments.
    p_init : (n,) array
        Price guess.
    max_iter : int
        Maximum iterations.
    lr : float
        Learning rate (step size per iteration).
    tol : float
        Convergence threshold on max gradient magnitude.
    verbose : bool

    Returns
    -------
    result : dict
        'profiles'    : list of (m, n) arrays — U_expressed at each step
        'payoffs'     : (n_iter, m) array — payoffs at each step
        'magnitudes'  : (n_iter, m) array — gradient magnitude per player
        'converged'   : bool
        'n_iter'      : int
    """
    T = np.array(T, dtype=float)
    U_true = np.array(U_true, dtype=float)
    U_expr = np.array(U_init, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()
    m, n = T.shape

    profiles = [U_expr.copy()]
    payoffs_list = []
    magnitudes_list = []
    converged = False

    for iteration in range(max_iter):
        # Evaluate current payoffs
        payoffs, _ = ccg_payoff(T, U_true, U_expr, Y, p_init,
                                 solver=solver, tol=solver_tol)
        payoffs_list.append(payoffs.copy())

        # Compute gradient direction and magnitude for each player
        mags = np.zeros(m)
        U_new = U_expr.copy()

        for player in range(m):
            direction, mag = best_response_direction(
                T, U_true, U_expr, Y, p_init, player,
                solver=solver, tol=solver_tol)
            mags[player] = mag
            U_new[player, :] += lr * direction

        U_new = np.maximum(U_new, 1e-6)  # Keep utilities positive
        magnitudes_list.append(mags.copy())

        if verbose:
            pay_str = ', '.join(f'{p:.4f}' for p in payoffs)
            mag_str = ', '.join(f'{m:.6f}' for m in mags)
            print(f"  Iter {iteration + 1}: payoffs=[{pay_str}]  "
                  f"grad_mag=[{mag_str}]")

        # Check convergence
        if mags.max() < tol:
            converged = True
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            U_expr = U_new
            profiles.append(U_expr.copy())
            break

        U_expr = U_new
        profiles.append(U_expr.copy())

    # Final payoff evaluation
    payoffs_final, _ = ccg_payoff(T, U_true, U_expr, Y, p_init,
                                   solver=solver, tol=solver_tol)
    payoffs_list.append(payoffs_final)

    return {
        'profiles': profiles,
        'payoffs': np.array(payoffs_list),
        'magnitudes': np.array(magnitudes_list),
        'converged': converged,
        'n_iter': len(profiles) - 1,
    }


def find_nash_candidates(T, U_true, Y, p_init, n_restarts=5,
                          max_iter=50, lr=0.1, tol=1e-4,
                          solver='robust', solver_tol=1e-6,
                          verbose=False):
    """
    Multi-start Nash equilibrium search.

    Runs nash_iteration from several starting points and returns
    candidates ranked by convergence quality (smallest gradient magnitude).

    Parameters
    ----------
    T, U_true, Y, p_init : array_like
    n_restarts : int
        Number of random starting points.
    max_iter : int
    lr : float
    tol : float
    verbose : bool

    Returns
    -------
    candidates : list of dict
        Sorted by convergence quality (best first). Each dict has:
        'U_expressed'     : (m, n) — final strategy profile
        'payoffs'         : (m,)   — payoffs at final profile
        'convergence_gap' : float  — max gradient magnitude (0 = exact Nash)
        'n_iter'          : int    — iterations used
        'converged'       : bool
    """
    T = np.array(T, dtype=float)
    U_true = np.array(U_true, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()
    m, n = T.shape

    candidates = []

    for restart in range(n_restarts):
        if restart == 0:
            # First restart: start at true utilities
            U_init = U_true.copy()
        elif restart == 1:
            # Second restart: start at uniform
            U_init = np.ones((m, n))
        else:
            # Random restarts
            U_init = np.random.uniform(0.3, 2.0, (m, n))

        if verbose:
            print(f"\n--- Restart {restart + 1}/{n_restarts} ---")

        result = nash_iteration(
            T, U_true, U_init, Y, p_init,
            max_iter=max_iter, lr=lr, tol=tol,
            solver=solver, solver_tol=solver_tol, verbose=verbose)

        final_profile = result['profiles'][-1]
        final_payoffs = result['payoffs'][-1]

        if len(result['magnitudes']) > 0:
            convergence_gap = result['magnitudes'][-1].max()
        else:
            convergence_gap = float('inf')

        candidates.append({
            'U_expressed': final_profile,
            'payoffs': final_payoffs,
            'convergence_gap': convergence_gap,
            'n_iter': result['n_iter'],
            'converged': result['converged'],
        })

    # Sort by convergence quality
    candidates.sort(key=lambda x: x['convergence_gap'])

    return candidates
