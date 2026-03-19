"""
ccg.py  –  Consumer Choice Game (CCG) for the Simple Closed Model

The CCG models strategic preference expression in the SCM. Each labour class
(player) has true utilities U_true but can express different preferences
U_expressed. The economy runs at equilibrium under U_expressed, and each
player's payoff is evaluated using U_true on the resulting allocations.

Key concepts:
  - "Friction" = consumers playing U_expressed ≠ U_true (brand loyalty, habit)
  - "AI agents" = forcing U_expressed → U_true (optimal preference expression)
  - Zone structure: each (I, J, F) triple defines a combinatorial zone in
    strategy space. Within a zone, payoffs are smooth algebraic functions
    of U_expressed. Zone boundaries are regime shifts.
  - F = Fisher forest: the spending pattern (who buys what, BPB ordering)

Ported from MATLAB: FeigningU.m, grad*.m, and the CCG framework in
Deshpande & Sohoni (arXiv:2109.09248), Sections 4.3, 6.1.

Usage
-----
    from scm.ccg import ccg_payoff, ccg_sweep, ccg_gradient, extract_forest

    # Single payoff evaluation
    payoffs, result = ccg_payoff(T, U_true, U_expressed, Y, p_init)

    # Sweep over a parameter grid
    table = ccg_sweep(T, U_true, Y, p_init, U_func, param_grid)

    # Numerical gradient (Jacobian)
    J = ccg_gradient(T, U_true, U_expressed, Y, p_init)

    # Extract Fisher forest from equilibrium
    forest, bpb_order = extract_forest(U, p, X, I, J)
"""

import numpy as np
from .solvers import solve_robust
from .scm_round import scm_round


# ======================================================================
# Fisher Forest Extraction
# ======================================================================

def extract_forest(U, p, X, I, J, tol=1e-3):
    """
    Extract the Fisher forest (spending pattern) from equilibrium data.

    The Fisher forest F encodes the bang-per-buck (BPB) structure: for each
    active class, which goods it buys and in what BPB order. Two equilibria
    with the same (I, J) but different F are in different combinatorial zones.

    Parameters
    ----------
    U : (m, n) array
        Utility matrix (the expressed utilities used to solve the equilibrium).
    p : (n,) array
        Equilibrium prices.
    X : (m, n) array
        Allocation matrix (units).
    I : array of int
        Active labour class indices.
    J : array of int
        Active goods indices.
    tol : float
        Relative threshold for considering an allocation positive.
        A class is considered to buy good j if X[i,j] / sum(X[i,:]) > tol.
        Default 1e-3 avoids misclassifying numerical residuals as purchases.

    Returns
    -------
    forest : tuple of tuples
        forest[k] = tuple of good indices that active class I[k] buys,
        sorted by descending BPB ratio.
        Example: ((0, 1), (1,)) means active class 0 buys goods 0 and 1
        (good 0 has higher BPB), active class 1 buys only good 1.
    bpb_ordering : list of list of (good_idx, bpb_ratio)
        For diagnostics: full BPB ranking per active class.
        bpb_ordering[k] = [(j, ratio), ...] sorted descending.
    """
    U = np.asarray(U, dtype=float)
    p = np.asarray(p, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    I = np.asarray(I).ravel()
    J = np.asarray(J).ravel()

    forest_list = []
    bpb_ordering = []

    for i in I:
        # Compute BPB for active goods
        ratios = []
        for j in J:
            if p[j] > 1e-12:
                bpb = U[i, j] / p[j]
            else:
                bpb = float('inf') if U[i, j] > 0 else 0.0
            ratios.append((int(j), bpb))

        # Sort by descending BPB
        ratios.sort(key=lambda x: -x[1])
        bpb_ordering.append(ratios)

        # Forest: goods this class actually buys (relative threshold)
        total_alloc = X[i, :].sum()
        if total_alloc > 1e-12:
            bought = tuple(j for j, _ in ratios
                           if X[i, j] / total_alloc > tol)
        else:
            bought = ()
        forest_list.append(bought)

    return tuple(forest_list), bpb_ordering


def zone_label(I, J, forest=None):
    """
    Generate a readable string label for a combinatorial zone (I, J, F).

    Parameters
    ----------
    I : array-like
        Active labour class indices.
    J : array-like
        Active goods indices.
    forest : tuple of tuples or None
        Fisher forest from extract_forest(). If None, label uses (I,J) only.

    Returns
    -------
    label : str
        E.g. "I={0,1}_J={0,2}" or "I={0,1}_J={0,2}_F={0→0,1|1→1}"
    """
    I = np.asarray(I).ravel()
    J = np.asarray(J).ravel()

    I_str = ','.join(str(int(x)) for x in sorted(I))
    J_str = ','.join(str(int(x)) for x in sorted(J))
    label = f"I={{{I_str}}}_J={{{J_str}}}"

    if forest is not None:
        parts = []
        for k, goods in enumerate(forest):
            if goods:
                goods_str = ','.join(str(int(g)) for g in goods)
                # Use the actual class index (I[k]) in the label
                class_idx = int(I[k]) if k < len(I) else k
                parts.append(f"{class_idx}\u2192{goods_str}")
        if parts:
            label += "_F={" + "|".join(parts) + "}"

    return label


def describe_forest(I, J, forest, m=None):
    """
    General structural description of a Fisher forest from solver output.

    Works for ANY m×n economy. Produces a dict describing each class's role
    (which goods it buys) and a canonical string label. No hardcoded zone
    names — the label is purely derived from the solver's output.

    Parameters
    ----------
    I : array-like
        Active class indices.
    J : array-like
        Active good indices.
    forest : tuple of tuples
        Fisher forest from extract_forest(). forest[k] = goods bought by I[k].
    m : int or None
        Total number of classes in the economy. If None, inferred from max(I)+1.

    Returns
    -------
    desc : dict with keys:
        'n_active_classes' : int
        'n_active_goods'   : int
        'active_classes'   : tuple of int
        'active_goods'     : tuple of int
        'pattern'          : tuple of frozenset
            pattern[k] = frozenset of goods bought by active class I[k].
            Canonical form: order follows I. For inactive classes, empty set.
        'roles'            : dict {class_idx: 'specialist'|'generalist'|'inactive'}
            'specialist' = buys exactly 1 good
            'generalist' = buys 2+ goods
            'inactive'   = not in I (zero allocation)
        'label'            : str
            Canonical human-readable label, e.g. "C0:{g0}|C1:{g0,g1}" or
            "C1_only:{g0,g1}". Encodes the full spending pattern without
            assuming any paper-specific naming convention.
    """
    I = np.asarray(I).ravel()
    J = np.asarray(J).ravel()

    if forest is None:
        return {'n_active_classes': 0, 'n_active_goods': 0,
                'active_classes': (), 'active_goods': (),
                'pattern': (), 'roles': {}, 'label': 'ERR'}

    if m is None:
        m = int(max(I)) + 1 if len(I) > 0 else 0

    n_active = len(I)
    n_goods = len(J)

    # Build pattern and roles
    pattern = []
    roles = {}
    label_parts = []

    for k in range(n_active):
        goods = frozenset(forest[k]) if k < len(forest) else frozenset()
        pattern.append(goods)
        class_idx = int(I[k])

        if len(goods) == 0:
            roles[class_idx] = 'inactive'
        elif len(goods) == 1:
            roles[class_idx] = 'specialist'
        else:
            roles[class_idx] = 'generalist'

        goods_str = ','.join(f'g{g}' for g in sorted(goods))
        label_parts.append(f"C{class_idx}:{{{goods_str}}}")

    # Mark inactive classes
    for i in range(m):
        if i not in [int(x) for x in I]:
            roles[i] = 'inactive'

    # Build canonical label
    if n_active == 0:
        label = 'EMPTY'
    elif n_active < m:
        # Some classes inactive — note which are active
        active_str = ','.join(f'C{int(i)}' for i in I)
        label = f"[{active_str}]_" + '|'.join(label_parts)
    else:
        label = '|'.join(label_parts)

    return {
        'n_active_classes': n_active,
        'n_active_goods': n_goods,
        'active_classes': tuple(int(i) for i in I),
        'active_goods': tuple(int(j) for j in J),
        'pattern': tuple(pattern),
        'roles': roles,
        'label': label,
    }


def classify_zone(I, J, forest, m=None):
    """
    Classify a forest into a canonical structural label.

    Convenience wrapper around describe_forest() that returns just the label.
    Works for any m×n economy — no paper-specific names.

    Parameters
    ----------
    I : array-like
        Active class indices.
    J : array-like
        Active good indices.
    forest : tuple of tuples
        Fisher forest from extract_forest().
    m : int or None
        Total number of classes. If None, inferred from max(I)+1.

    Returns
    -------
    label : str
        Canonical structural label, e.g. "C0:{g0}|C1:{g0,g1}".
    """
    desc = describe_forest(I, J, forest, m=m)
    return desc['label']


# ======================================================================
# CCG Payoff Functions
# ======================================================================

def ccg_payoff(T, U_true, U_expressed, Y, p_init,
               solver='robust', tol=1e-6):
    """
    Compute CCG payoffs: run equilibrium under U_expressed, evaluate at U_true.

    Each class i's payoff is:
        b_i = sum_j U_true[i,j] * X[i,j]

    where X is the equilibrium allocation under U_expressed.

    Parameters
    ----------
    T           : (m, n)  Technology matrix.
    U_true      : (m, n)  True utility matrix.
    U_expressed : (m, n)  Expressed (strategic) utility matrix.
    Y           : (m,)    Labour endowments.
    p_init      : (n,)    Initial price guess.
    solver      : str     'robust' (recommended) or 'standard'.
    tol         : float   Solver tolerance.

    Returns
    -------
    payoffs : (m,) array
        Payoff for each class: b_i = U_true[i,:] · X[i,:].
    result : dict
        Full equilibrium result dict (prices, allocations, etc.)
        under U_expressed.
    """
    T = np.array(T, dtype=float)
    U_true = np.array(U_true, dtype=float)
    U_expressed = np.array(U_expressed, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()
    m, n = T.shape

    # Solve equilibrium under expressed utilities
    if solver == 'robust':
        result = solve_robust(T, U_expressed, Y, p_init, tol=tol)
    else:
        from .equilibrium import compute_equilibrium
        result = compute_equilibrium(T, U_expressed, Y, p_init, tol=tol)

    # Get allocations (units)
    X = result.get('X', np.zeros((m, n)))

    # Compute payoffs using true utilities
    payoffs = np.array([U_true[i, :] @ X[i, :] for i in range(m)])

    return payoffs, result


def ccg_payoff_detailed(T, U_true, U_expressed, Y, p_init,
                        solver='robust', tol=1e-6):
    """
    Detailed CCG payoff computation with per-good breakdown and Fisher forest.

    Returns
    -------
    payoffs     : (m,) array   Total payoff per class.
    payoff_mat  : (m, n) array Per-good payoff contribution: U_true[i,j] * X[i,j].
    wages       : (m,) array   Wage income per class at equilibrium.
    prices      : (n,) array   Equilibrium prices.
    quantities  : (n,) array   Equilibrium production.
    allocations : (m, n) array Allocation matrix (units).
    zone        : dict         Zone data: I, J, F (forest), bpb_ordering, status.
    """
    T = np.array(T, dtype=float)
    U_true = np.array(U_true, dtype=float)
    U_expressed = np.array(U_expressed, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()
    m, n = T.shape

    if solver == 'robust':
        result = solve_robust(T, U_expressed, Y, p_init, tol=tol)
    else:
        from .equilibrium import compute_equilibrium
        result = compute_equilibrium(T, U_expressed, Y, p_init, tol=tol)

    X = result.get('X', np.zeros((m, n)))
    payoff_mat = U_true * X
    payoffs = payoff_mat.sum(axis=1)

    I = result.get('I', np.array([], dtype=int))
    J = result.get('J', np.array([], dtype=int))
    p = result.get('p', np.zeros(n))

    # Extract Fisher forest
    forest, bpb_order = extract_forest(U_expressed, p, X, I, J)

    zone = {
        'I': I,
        'J': J,
        'F': forest,
        'bpb_ordering': bpb_order,
        'status': result.get('status', 'unknown'),
        'method': result.get('method', 'unknown'),
        'fp_error': result.get('fp_error', None),
    }

    return payoffs, payoff_mat, result['W'], result['p'], result['q'], X, zone


# ======================================================================
# Parameter Sweeps
# ======================================================================

def ccg_sweep(T, U_true, Y, p_init, U_func, param_grid,
              solver='robust', tol=1e-6, verbose=False):
    """
    Sweep CCG payoffs over a parameter grid.

    This is the Python equivalent of FeigningU.m: for each parameter value,
    construct U_expressed via U_func, compute equilibrium, evaluate payoffs.

    Parameters
    ----------
    T          : (m, n)  Technology matrix.
    U_true     : (m, n)  True utility matrix.
    Y          : (m,)    Labour endowments.
    p_init     : (n,)    Initial price guess.
    U_func     : callable
        Function (params) -> U_expressed (m, n). Takes a dict of parameter
        values and returns the expressed utility matrix.
    param_grid : list of dict
        Each dict is a set of parameter values passed to U_func.
        Example: [{'alpha': 0.5, 'beta': 0.8}, {'alpha': 1.0, 'beta': 0.8}, ...]
    solver     : str     'robust' or 'standard'.
    tol        : float   Solver tolerance.
    verbose    : bool    Print progress.

    Returns
    -------
    results : list of dict
        One entry per parameter point, with keys:
        'params'      : the parameter dict
        'payoffs'     : (m,) payoff array
        'wages'       : (m,) wage income array
        'prices'      : (n,) equilibrium prices
        'quantities'  : (n,) production quantities
        'allocations' : (m,n) allocation matrix
        'zone_I'      : active labour indices
        'zone_J'      : active goods indices
        'forest'      : Fisher forest tuple
        'zone_label'  : full zone label string
        'status'      : solver status
        'fp_error'    : fixed-point error
    """
    T = np.array(T, dtype=float)
    U_true = np.array(U_true, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()

    results = []

    for idx, params in enumerate(param_grid):
        U_expr = np.array(U_func(params), dtype=float)

        payoffs, payoff_mat, wages, prices, quantities, X, zone = \
            ccg_payoff_detailed(T, U_true, U_expr, Y, p_init,
                                solver=solver, tol=tol)

        zlabel = zone_label(zone['I'], zone['J'], zone['F'])

        entry = {
            'params': params,
            'payoffs': payoffs,
            'payoff_mat': payoff_mat,
            'wages': wages,
            'prices': prices,
            'quantities': quantities,
            'allocations': X,
            'zone_I': zone['I'],
            'zone_J': zone['J'],
            'forest': zone['F'],
            'zone_label': zlabel,
            'status': zone['status'],
            'fp_error': zone['fp_error'],
        }
        results.append(entry)

        if verbose:
            m = len(payoffs)
            pay_str = ', '.join(f'{payoffs[i]:.4f}' for i in range(m))
            print(f"  [{idx+1}/{len(param_grid)}] {params} → "
                  f"payoffs=[{pay_str}]  {zlabel}  {zone['status']}")

    return results


# ======================================================================
# Numerical Gradient (Jacobian)
# ======================================================================

def ccg_gradient(T, U_true, U_expressed, Y, p_init,
                 player=None, eps=1e-5, solver='robust', tol=1e-6):
    """
    Numerical CCG gradient: ∂payoff_i / ∂U_expressed[k,l] via finite differences.

    For each entry (k,l) of U_expressed, perturb it by ±eps and measure the
    change in payoffs. Returns the full Jacobian or a single player's gradient.

    Parameters
    ----------
    T, U_true, U_expressed, Y, p_init : array_like
        Economy parameters.
    player : int or None
        If int, return gradient for that player only (m*n vector reshaped to m×n).
        If None, return full Jacobian (m × m × n): J[i,k,l] = ∂b_i/∂U[k,l].
    eps : float
        Finite difference step size.
    solver : str
        'robust' or 'standard'.
    tol : float
        Solver tolerance.

    Returns
    -------
    If player is None:
        J : (m, m, n) array — J[i,k,l] = ∂payoff_i / ∂U_expressed[k,l]
    If player is int:
        grad : (m, n) array — grad[k,l] = ∂payoff_player / ∂U_expressed[k,l]
    """
    T = np.array(T, dtype=float)
    U_true = np.array(U_true, dtype=float)
    U_expr = np.array(U_expressed, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()
    m, n = T.shape

    # Base payoffs
    payoffs_base, _ = ccg_payoff(T, U_true, U_expr, Y, p_init,
                                  solver=solver, tol=tol)

    J = np.zeros((m, m, n))

    for k in range(m):
        for l in range(n):
            U_plus = U_expr.copy()
            U_plus[k, l] += eps

            payoffs_plus, _ = ccg_payoff(T, U_true, U_plus, Y, p_init,
                                          solver=solver, tol=tol)

            U_minus = U_expr.copy()
            U_minus[k, l] -= eps

            payoffs_minus, _ = ccg_payoff(T, U_true, U_minus, Y, p_init,
                                           solver=solver, tol=tol)

            J[:, k, l] = (payoffs_plus - payoffs_minus) / (2 * eps)

    if player is not None:
        return J[player, :, :]
    return J


# ======================================================================
# Zone Mapping
# ======================================================================

def ccg_zone_map(T, U_true, Y, p_init, U_func, param1_grid, param2_grid,
                 param1_name='alpha', param2_name='beta',
                 solver='robust', tol=1e-6, verbose=False):
    """
    Map the zone structure (I, J, F) across a 2D parameter grid.

    For each (param1, param2) point, compute the equilibrium under
    U_expressed = U_func({param1_name: p1, param2_name: p2}) and record
    which combinatorial zone (I, J, F) is active.

    Useful for identifying zone boundaries and regime shifts.

    Parameters
    ----------
    T, U_true, Y, p_init : array_like
        Economy parameters.
    U_func : callable
        (params_dict) -> U_expressed (m, n).
    param1_grid, param2_grid : 1D arrays
        Grid values for the two parameters.
    param1_name, param2_name : str
        Names used as keys in the params dict.

    Returns
    -------
    zone_grid : (n1, n2) array of str
        Zone label at each grid point. Includes Fisher forest if available.
    payoff_grid : (n1, n2, m) array
        Payoffs at each grid point.
    wage_grid : (n1, n2, m) array
        Wages at each grid point.
    forest_grid : (n1, n2) array of object
        Fisher forest tuple at each grid point.
    """
    T = np.array(T, dtype=float)
    U_true = np.array(U_true, dtype=float)
    Y = np.array(Y, dtype=float).ravel()
    p_init = np.array(p_init, dtype=float).ravel()
    m = T.shape[0]

    n1, n2 = len(param1_grid), len(param2_grid)
    zone_grid = np.empty((n1, n2), dtype=object)
    payoff_grid = np.zeros((n1, n2, m))
    wage_grid = np.zeros((n1, n2, m))
    forest_grid = np.empty((n1, n2), dtype=object)

    total = n1 * n2
    count = 0

    n_errors = 0
    for i, v1 in enumerate(param1_grid):
        for j, v2 in enumerate(param2_grid):
            count += 1
            params = {param1_name: v1, param2_name: v2}
            U_expr = np.array(U_func(params), dtype=float)

            try:
                payoffs, _, wages, _, _, _, zone = \
                    ccg_payoff_detailed(T, U_true, U_expr, Y, p_init,
                                        solver=solver, tol=tol)

                zlabel = zone_label(zone['I'], zone['J'], zone['F'])

                zone_grid[i, j] = zlabel
                forest_grid[i, j] = zone['F']
                payoff_grid[i, j, :] = payoffs
                wage_grid[i, j, :] = wages
            except Exception:
                zone_grid[i, j] = 'ERROR'
                forest_grid[i, j] = None
                n_errors += 1

            if verbose and count % max(1, total // 20) == 0:
                print(f"  [{count}/{total}] ({param1_name}={v1:.2f}, "
                      f"{param2_name}={v2:.2f}) → {zone_grid[i, j]}")

    if n_errors > 0 and verbose:
        print(f"  Warning: {n_errors}/{total} points failed to solve.")

    return zone_grid, payoff_grid, wage_grid, forest_grid
