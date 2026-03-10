#!/usr/bin/env python3
"""
test_task2_methods.py — Benchmark Task 2 alternative equilibrium methods on 35 economies.

Methods tested:
  1. Standard tatonnement (baseline)
  2. Damped tatonnement (α sweep: 0.3, 0.1, 0.05, 0.02)
  3. Anderson acceleration (scipy.optimize.anderson)
  4. Newton/quasi-Newton (scipy.optimize.root with hybr, broyden1)
  5. Combined: damped warmup → Anderson/Newton polish

For each economy, every method is tried and results compared.
"""

import numpy as np
import warnings
import time
from collections import OrderedDict

from scm.scm_round import scm_round
from scm.equilibrium import compute_equilibrium

warnings.filterwarnings("ignore")


# =========================================================================
# Method implementations
# =========================================================================

def fp_error(T, U, Y, p):
    """Compute fixed-point error: max|G(p) - p| with price normalisation."""
    try:
        p_new, _, _, _, _, _ = scm_round(T, U, Y, p)
        p_n = p / p.mean()
        p_new_n = p_new / p_new.mean()
        return float(np.max(np.abs(p_new_n - p_n)))
    except:
        return float('inf')


def method_standard(T, U, Y, p_init, tol=1e-6, max_iter=200):
    """Standard tatonnement."""
    res = compute_equilibrium(T, U, Y, p_init, max_iter=max_iter, tol=tol)
    if res['status'] == 'converged':
        err = fp_error(T, U, Y, res['p'])
        return res['p'], err, res['n_iter']
    return None, float('inf'), max_iter


def method_damped(T, U, Y, p_init, tol=1e-6, max_iter=1000):
    """Damped tatonnement with alpha sweep and price normalisation."""
    n = len(p_init)
    for alpha in [0.3, 0.1, 0.05]:
        p = p_init.copy() / p_init.mean()
        best_err = np.inf
        best_p = p.copy()
        for it in range(max_iter):
            try:
                p_scm, _, _, _, _, _ = scm_round(T, U, Y, p)
                p_scm_n = p_scm / p_scm.mean()
                p_n = p / p.mean()
                err = float(np.max(np.abs(p_scm_n - p_n)))
                if err < best_err:
                    best_err = err
                    best_p = p_n.copy()
                if err < tol:
                    return best_p, best_err, it + 1
                p = np.maximum((1 - alpha) * p_n + alpha * p_scm_n, 1e-6)
            except:
                p = np.maximum(p + np.random.randn(n) * 0.01, 1e-6)
                p = p / p.mean()
        if best_err < tol * 10:
            return best_p, best_err, max_iter
    return best_p, best_err, max_iter


def method_anderson(T, U, Y, p_init, tol=1e-6, max_iter=300):
    """Anderson acceleration via scipy.optimize.anderson."""
    from scipy.optimize import anderson

    n = len(p_init)
    p0 = p_init.copy() / p_init.mean()

    def F(p):
        p_pos = np.maximum(p, 1e-8)
        p_n = p_pos / p_pos.mean()
        try:
            p_new, _, _, _, _, _ = scm_round(T, U, Y, p_n)
            p_new_n = p_new / p_new.mean()
            return p_new_n - p_n
        except:
            return np.zeros(n)

    try:
        p_eq = anderson(F, p0, M=5, maxiter=max_iter,
                        f_tol=tol, f_rtol=1e-10, verbose=False)
        p_eq = np.maximum(p_eq, 1e-8)
        p_eq = p_eq / p_eq.mean()
        err = fp_error(T, U, Y, p_eq)
        return p_eq, err, -1  # anderson doesn't expose iter count easily
    except Exception as e:
        return None, float('inf'), max_iter


def method_anderson_damped_warmup(T, U, Y, p_init, tol=1e-6,
                                    warmup_iter=100, warmup_alpha=0.3,
                                    max_iter=300):
    """Damped warmup (100 iters at α=0.3), then Anderson acceleration."""
    from scipy.optimize import anderson

    n = len(p_init)
    # Warmup phase
    p = p_init.copy() / p_init.mean()
    for it in range(warmup_iter):
        try:
            p_scm, _, _, _, _, _ = scm_round(T, U, Y, p)
            p_scm_n = p_scm / p_scm.mean()
            p_n = p / p.mean()
            err = float(np.max(np.abs(p_scm_n - p_n)))
            if err < tol:
                return p_n, err, it + 1
            p = np.maximum((1 - warmup_alpha) * p_n + warmup_alpha * p_scm_n, 1e-6)
        except:
            p = np.maximum(p + np.random.randn(n) * 0.01, 1e-6)
            p = p / p.mean()

    # Anderson phase
    p0 = p / p.mean()

    def F(p):
        p_pos = np.maximum(p, 1e-8)
        p_n = p_pos / p_pos.mean()
        try:
            p_new, _, _, _, _, _ = scm_round(T, U, Y, p_n)
            p_new_n = p_new / p_new.mean()
            return p_new_n - p_n
        except:
            return np.zeros(n)

    try:
        p_eq = anderson(F, p0, M=5, maxiter=max_iter,
                        f_tol=tol, f_rtol=1e-10, verbose=False)
        p_eq = np.maximum(p_eq, 1e-8)
        p_eq = p_eq / p_eq.mean()
        err = fp_error(T, U, Y, p_eq)
        return p_eq, err, warmup_iter + max_iter
    except:
        # Anderson failed, return best from warmup
        err = fp_error(T, U, Y, p0)
        return p0, err, warmup_iter


def method_newton_hybr(T, U, Y, p_init, tol=1e-6, max_iter=500):
    """Powell's hybrid Newton method (scipy.optimize.root, method='hybr')."""
    from scipy.optimize import root

    n = len(p_init)
    p0 = p_init.copy() / p_init.mean()

    eval_count = [0]

    def F(p):
        eval_count[0] += 1
        p_pos = np.maximum(p, 1e-8)
        p_n = p_pos / p_pos.mean()
        try:
            p_new, _, _, _, _, _ = scm_round(T, U, Y, p_n)
            p_new_n = p_new / p_new.mean()
            return p_new_n - p_n
        except:
            return np.full(n, 1e10)

    try:
        res = root(F, p0, method='hybr',
                   options={'maxfev': max_iter * (n + 1), 'xtol': tol})
        p_eq = np.maximum(res.x, 1e-8)
        p_eq = p_eq / p_eq.mean()
        err = fp_error(T, U, Y, p_eq)
        return p_eq, err, eval_count[0]
    except:
        return None, float('inf'), eval_count[0]


def method_newton_broyden(T, U, Y, p_init, tol=1e-6, max_iter=500):
    """Broyden's quasi-Newton method (scipy.optimize.root, method='broyden1')."""
    from scipy.optimize import root

    n = len(p_init)
    p0 = p_init.copy() / p_init.mean()

    eval_count = [0]

    def F(p):
        eval_count[0] += 1
        p_pos = np.maximum(p, 1e-8)
        p_n = p_pos / p_pos.mean()
        try:
            p_new, _, _, _, _, _ = scm_round(T, U, Y, p_n)
            p_new_n = p_new / p_new.mean()
            return p_new_n - p_n
        except:
            return np.full(n, 1e10)

    try:
        res = root(F, p0, method='broyden1',
                   options={'maxiter': max_iter, 'fatol': tol, 'xatol': tol})
        p_eq = np.maximum(res.x, 1e-8)
        p_eq = p_eq / p_eq.mean()
        err = fp_error(T, U, Y, p_eq)
        return p_eq, err, eval_count[0]
    except:
        return None, float('inf'), eval_count[0]


def method_damped_then_newton(T, U, Y, p_init, tol=1e-6,
                               warmup_iter=200, warmup_alpha=0.3):
    """Damped warmup then Powell's hybrid Newton polish."""
    from scipy.optimize import root

    n = len(p_init)
    # Warmup
    p = p_init.copy() / p_init.mean()
    for it in range(warmup_iter):
        try:
            p_scm, _, _, _, _, _ = scm_round(T, U, Y, p)
            p_scm_n = p_scm / p_scm.mean()
            p_n = p / p.mean()
            err = float(np.max(np.abs(p_scm_n - p_n)))
            if err < tol:
                return p_n, err, it + 1
            p = np.maximum((1 - warmup_alpha) * p_n + warmup_alpha * p_scm_n, 1e-6)
        except:
            p = np.maximum(p + np.random.randn(n) * 0.01, 1e-6)
            p = p / p.mean()

    # Newton polish
    p0 = p / p.mean()
    eval_count = [0]

    def F(p):
        eval_count[0] += 1
        p_pos = np.maximum(p, 1e-8)
        p_n = p_pos / p_pos.mean()
        try:
            p_new, _, _, _, _, _ = scm_round(T, U, Y, p_n)
            p_new_n = p_new / p_new.mean()
            return p_new_n - p_n
        except:
            return np.full(n, 1e10)

    try:
        res = root(F, p0, method='hybr',
                   options={'maxfev': 500 * (n + 1), 'xtol': tol})
        p_eq = np.maximum(res.x, 1e-8)
        p_eq = p_eq / p_eq.mean()
        err = fp_error(T, U, Y, p_eq)
        return p_eq, err, warmup_iter + eval_count[0]
    except:
        err = fp_error(T, U, Y, p0)
        return p0, err, warmup_iter


# =========================================================================
# Economy definitions (same 35 as test_many_economies.py)
# =========================================================================

ECONOMIES = []

# 2x2
ECONOMIES.append({'name': '2x2_diagonal', 'T': [[1, 0], [0, 1]], 'U': [[3, 1], [1, 3]], 'Y': [10, 10], 'p0': [1, 1]})
ECONOMIES.append({'name': '2x2_paper_example', 'T': [[1, 3], [4, 1]], 'U': [[5, 1], [1, 5]], 'Y': [10, 10], 'p0': [1, 1]})
ECONOMIES.append({'name': '2x2_asymmetric', 'T': [[1, 2], [3, 1]], 'U': [[1, 4], [4, 1]], 'Y': [5, 15], 'p0': [1, 1]})
ECONOMIES.append({'name': '2x2_nearly_singular', 'T': [[1, 1.01], [1.01, 1]], 'U': [[2, 1], [1, 2]], 'Y': [10, 10], 'p0': [1, 1]})
ECONOMIES.append({'name': '2x2_extreme_prefs', 'T': [[1, 2], [2, 1]], 'U': [[100, 1], [1, 100]], 'Y': [10, 10], 'p0': [1, 1]})

# 3x3
ECONOMIES.append({'name': '3x3_paper_G22', 'T': [[1.0, 0.4, 0.5], [0.5, 1.5, 0.25], [0.2, 0.35, 0.6]], 'U': [[0.85, 0.5, 0.4], [0.4, 0.9, 0.45], [0.55, 0.4, 0.8]], 'Y': [10, 10, 10], 'p0': [1, 1.2, 1.3]})
ECONOMIES.append({'name': '3x3_paper_A3_toggle', 'T': [[0.05, 1.0, 0.9], [0.5, 0.8, 0.15], [0.4, 0.5, 0.4]], 'U': [[0.2, 0.3, 0.8], [0.9, 0.2, 0.4], [0.25, 0.85, 0.33]], 'Y': [10, 10, 10], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '3x3_random_cycling', 'T': [[4.18, 4.37, 0.53], [2.80, 2.38, 1.50], [1.04, 2.02, 4.74]], 'U': [[3.30, 5.24, 7.06], [3.70, 9.72, 9.63], [2.59, 5.02, 3.08]], 'Y': [7.85, 5.37, 11.10], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '3x3_identity', 'T': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'U': [[3, 1, 1], [1, 3, 1], [1, 1, 3]], 'Y': [10, 10, 10], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '3x3_strong_mismatch', 'T': [[1, 4, 1], [3, 1, 2], [2, 3, 1]], 'U': [[1, 8, 1], [8, 1, 1], [1, 1, 8]], 'Y': [10, 10, 10], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '3x3_sparse_T', 'T': [[1, 0, 2], [0, 1, 0], [1, 1, 1]], 'U': [[2, 1, 3], [1, 2, 1], [3, 1, 2]], 'Y': [10, 5, 15], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '3x3_unequal_Y', 'T': [[1, 2, 1], [2, 1, 2], [1, 2, 1]], 'U': [[5, 1, 2], [2, 5, 1], [1, 2, 5]], 'Y': [1, 10, 100], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '3x3_random_A', 'T': [[2.1, 0.8, 1.5], [1.3, 2.7, 0.4], [0.6, 1.1, 3.2]], 'U': [[1.5, 4.2, 0.8], [3.1, 0.9, 2.6], [0.7, 3.5, 1.2]], 'Y': [8, 12, 6], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '3x3_random_B', 'T': [[0.3, 1.8, 0.7], [1.5, 0.4, 2.1], [0.9, 1.2, 0.5]], 'U': [[4.0, 0.5, 2.0], [0.8, 3.5, 1.5], [2.5, 1.0, 3.0]], 'Y': [15, 8, 12], 'p0': [1, 1, 1]})

# 4x4
ECONOMIES.append({'name': '4x4_identity', 'T': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 'U': [[4, 1, 1, 1], [1, 4, 1, 1], [1, 1, 4, 1], [1, 1, 1, 4]], 'Y': [10, 10, 10, 10], 'p0': [1, 1, 1, 1]})
ECONOMIES.append({'name': '4x4_cyclic_T', 'T': [[1, 3, 0.5, 2], [2, 1, 3, 0.5], [0.5, 2, 1, 3], [3, 0.5, 2, 1]], 'U': [[1, 6, 2, 1], [6, 1, 1, 2], [2, 1, 1, 6], [1, 2, 6, 1]], 'Y': [10, 10, 10, 10], 'p0': [1, 1, 1, 1]})
ECONOMIES.append({'name': '4x4_random_A', 'T': [[1.5, 0.3, 2.1, 0.7], [0.8, 1.9, 0.4, 1.6], [2.0, 1.1, 0.9, 0.5], [0.4, 1.7, 1.3, 2.2]], 'U': [[2.0, 5.0, 1.0, 3.0], [4.0, 1.0, 3.0, 2.0], [1.0, 3.0, 5.0, 1.0], [3.0, 2.0, 1.0, 4.0]], 'Y': [10, 8, 12, 9], 'p0': [1, 1, 1, 1]})
ECONOMIES.append({'name': '4x4_random_B', 'T': [[0.5, 2.3, 1.1, 0.8], [1.7, 0.6, 0.9, 2.0], [1.0, 1.5, 2.4, 0.3], [2.2, 0.4, 0.7, 1.4]], 'U': [[3.0, 1.0, 2.0, 6.0], [1.0, 5.0, 3.0, 1.0], [6.0, 2.0, 1.0, 3.0], [2.0, 3.0, 5.0, 1.0]], 'Y': [7, 14, 5, 11], 'p0': [1, 1, 1, 1]})
ECONOMIES.append({'name': '4x4_extreme', 'T': [[3, 1, 2, 1], [1, 3, 1, 2], [2, 1, 3, 1], [1, 2, 1, 3]], 'U': [[10, 1, 1, 1], [1, 10, 1, 1], [1, 1, 10, 1], [1, 1, 1, 10]], 'Y': [10, 10, 10, 10], 'p0': [1, 1, 1, 1]})

# 5x5
ECONOMIES.append({'name': '5x5_identity', 'T': np.eye(5).tolist(), 'U': (np.eye(5) * 4 + np.ones((5, 5))).tolist(), 'Y': [10, 10, 10, 10, 10], 'p0': [1, 1, 1, 1, 1]})
ECONOMIES.append({'name': '5x5_random', 'T': [[1.2, 0.3, 0.8, 1.5, 0.4], [0.5, 2.1, 0.6, 0.3, 1.3], [1.0, 0.7, 1.8, 0.9, 0.5], [0.3, 1.4, 0.5, 2.0, 0.8], [0.8, 0.5, 1.2, 0.4, 1.9]], 'U': [[5, 2, 1, 3, 1], [1, 5, 3, 1, 2], [2, 1, 5, 1, 3], [3, 1, 1, 5, 2], [1, 3, 2, 2, 5]], 'Y': [10, 8, 12, 7, 11], 'p0': [1, 1, 1, 1, 1]})

# 6x6
ECONOMIES.append({'name': '6x6_block_diagonal', 'T': np.block([
    [np.array([[1, 2], [2, 1]]), np.zeros((2, 2)), np.zeros((2, 2))],
    [np.zeros((2, 2)), np.array([[1, 3], [3, 1]]), np.zeros((2, 2))],
    [np.zeros((2, 2)), np.zeros((2, 2)), np.array([[1, 1.5], [1.5, 1]])]
]).tolist(), 'U': (np.eye(6) * 5 + np.ones((6, 6))).tolist(), 'Y': [10]*6, 'p0': [1]*6})

# Random generated
np.random.seed(2024)
for dim in [3, 4, 5]:
    for trial in range(3):
        T = np.random.uniform(0.2, 3.0, (dim, dim))
        U = np.random.uniform(0.5, 8.0, (dim, dim))
        Y = np.random.uniform(5, 15, dim)
        ECONOMIES.append({'name': f'{dim}x{dim}_rand_seed2024_t{trial}', 'T': np.round(T, 4).tolist(), 'U': np.round(U, 4).tolist(), 'Y': np.round(Y, 4).tolist(), 'p0': np.ones(dim).tolist()})

# Stress tests
ECONOMIES.append({'name': '3x3_competing_circulant', 'T': [[1, 2, 3], [3, 1, 2], [2, 3, 1]], 'U': [[1, 5, 1], [1, 1, 5], [5, 1, 1]], 'Y': [10, 10, 10], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '3x3_near_degenerate', 'T': [[1, 1, 1], [1, 1.001, 1], [1, 1, 1.001]], 'U': [[3, 1, 1], [1, 3, 1], [1, 1, 3]], 'Y': [10, 10, 10], 'p0': [1, 1, 1]})
ECONOMIES.append({'name': '4x4_one_dominant_good', 'T': [[0.1, 1, 1, 1], [0.1, 1, 1, 1], [0.1, 1, 1, 1], [0.1, 1, 1, 1]], 'U': [[10, 1, 1, 1], [10, 1, 1, 1], [10, 1, 1, 1], [10, 1, 1, 1]], 'Y': [10, 10, 10, 10], 'p0': [1, 1, 1, 1]})
ECONOMIES.append({'name': '3x3_very_asymmetric_Y', 'T': [[1, 2, 1], [2, 1, 2], [1, 2, 1]], 'U': [[3, 1, 2], [1, 3, 1], [2, 1, 3]], 'Y': [0.1, 10, 1000], 'p0': [1, 1, 1]})


# =========================================================================
# All methods to test
# =========================================================================
METHODS = OrderedDict([
    ('standard',          method_standard),
    ('damped',            method_damped),
    ('anderson',          method_anderson),
    ('damped+anderson',   method_anderson_damped_warmup),
    ('newton_hybr',       method_newton_hybr),
    ('broyden1',          method_newton_broyden),
    ('damped+newton',     method_damped_then_newton),
])


# =========================================================================
# Main benchmark
# =========================================================================
if __name__ == '__main__':
    TOL = 1e-6

    # Focus on problem economies first, then all
    problem_names = {
        '5x5_rand_seed2024_t0', '5x5_rand_seed2024_t1', '5x5_rand_seed2024_t2',
        '2x2_nearly_singular', '3x3_near_degenerate',
        '4x4_random_A', '4x4_random_B', '6x6_block_diagonal',
        '3x3_competing_circulant',
    }

    # Collect all results
    all_results = {}

    header = f"{'Economy':<35s}"
    for mname in METHODS:
        header += f" {mname:>16s}"
    print(header)
    print("=" * len(header))

    for eco in ECONOMIES:
        T = np.array(eco['T'], dtype=float)
        U = np.array(eco['U'], dtype=float)
        Y = np.array(eco['Y'], dtype=float)
        p0 = np.array(eco['p0'], dtype=float)

        row = f"{eco['name']:<35s}"
        eco_results = {}

        for mname, mfunc in METHODS.items():
            t0 = time.time()
            try:
                p_eq, err, iters = mfunc(T, U, Y, p0, tol=TOL)
                elapsed = time.time() - t0
                if err < 1e-6:
                    tag = f"{err:.0e}"
                elif err < 1e-4:
                    tag = f"{err:.0e}"
                elif err < 1e-2:
                    tag = f"~{err:.0e}"
                else:
                    tag = f"FAIL"
                # timeout guard: if >60s and not great, mark slow
                if elapsed > 30:
                    tag += "/slow"
                row += f" {tag:>16s}"
                eco_results[mname] = {'err': err, 'iters': iters, 'time': elapsed}
            except Exception as e:
                row += f" {'ERR':>16s}"
                eco_results[mname] = {'err': float('inf'), 'iters': 0, 'time': 0}

        all_results[eco['name']] = eco_results
        print(row)

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Best fp_error per economy across all methods")
    print("=" * 80)

    for mname in METHODS:
        n_ok = sum(1 for r in all_results.values() if r[mname]['err'] < 1e-4)
        n_approx = sum(1 for r in all_results.values() if 1e-4 <= r[mname]['err'] < 1e-2)
        n_fail = sum(1 for r in all_results.values() if r[mname]['err'] >= 1e-2)
        print(f"  {mname:<20s}  OK: {n_ok:>2d}  APPROX: {n_approx:>2d}  FAIL: {n_fail:>2d}")

    # Best method per economy
    print("\n" + "-" * 80)
    print("Best method per economy (lowest fp_error):")
    print("-" * 80)
    overall_best_ok = 0
    for ename, eres in all_results.items():
        best_m = min(eres, key=lambda m: eres[m]['err'])
        best_err = eres[best_m]['err']
        status = 'OK' if best_err < 1e-4 else 'APPROX' if best_err < 1e-2 else 'FAIL'
        if best_err < 1e-4:
            overall_best_ok += 1
        if best_err >= 1e-4:  # only show non-trivial
            print(f"  {ename:<35s}  {best_m:<20s}  err={best_err:.2e}  {status}")

    print(f"\nOverall: {overall_best_ok}/{len(all_results)} economies solved to <1e-4 by at least one method")

    # Specifically for the 5 problem cases
    print("\n" + "=" * 80)
    print("DETAILED: Problem economies")
    print("=" * 80)
    for pname in sorted(problem_names):
        if pname in all_results:
            print(f"\n  {pname}:")
            for mname in METHODS:
                r = all_results[pname][mname]
                status = 'OK' if r['err'] < 1e-4 else 'APPROX' if r['err'] < 1e-2 else 'FAIL'
                print(f"    {mname:<20s}  err={r['err']:<10.2e}  time={r['time']:.1f}s  {status}")
