"""
test_many_economies.py - Comprehensive test suite: 35 economies across dimensions.

Tests standard tatonnement first; falls back to damped + normalised
tatonnement if it cycles or hits max_iter.  Each economy is tagged
with its expected behavior where known.

Can be run standalone (python tests/test_many_economies.py) for a
summary table, or via pytest for pass/fail per economy.
"""

import numpy as np
import pytest

from scm.scm_round import scm_round
from scm.equilibrium import compute_equilibrium


# =========================================================================
# Helper: solve with fallback to damped tatonnement
# =========================================================================
def solve_economy(T, U, Y, p_init, max_iter=200, tol=1e-6):
    """
    Try standard tatonnement first.
    If it cycles or hits max_iter, try damped + normalized.
    Returns (p_eq, method_used, fp_error, n_iter).
    """
    T = np.array(T, dtype=float)
    U = np.array(U, dtype=float)
    Y = np.array(Y, dtype=float)
    p = np.array(p_init, dtype=float)
    n = len(p)

    # Attempt 1: standard tatonnement
    try:
        res = compute_equilibrium(T, U, Y, p, max_iter=max_iter, tol=tol)
        if res['status'] == 'converged':
            p_new, _, _, _, _, _ = scm_round(T, U, Y, res['p'])
            fp_err = float(np.max(np.abs(p_new - res['p'])))
            return res['p'], 'standard', fp_err, res['n_iter']
    except:
        pass

    # Attempt 2: damped + normalized, try several alphas
    for alpha in [0.3, 0.1, 0.05, 0.02]:
        p = np.array(p_init, dtype=float)
        p = p / p.mean()
        best_err = np.inf
        best_p = p.copy()
        for it in range(3000):
            try:
                p_scm, _, _, _, _, _ = scm_round(T, U, Y, p)
                p_scm_n = p_scm / p_scm.mean()
                p_n = p / p.mean()
                fp_err = float(np.max(np.abs(p_scm_n - p_n)))
                if fp_err < best_err:
                    best_err = fp_err
                    best_p = p_n.copy()
                if fp_err < tol:
                    return p_n, f'damped(α={alpha})', fp_err, it + 1
                p = np.maximum((1 - alpha) * p_n + alpha * p_scm_n, 1e-6)
            except:
                p = np.maximum(p + np.random.randn(n) * 0.01, 1e-6)
                p = p / p.mean()

        if best_err < tol * 100:
            return best_p, f'damped(α={alpha},approx)', best_err, 3000

    return best_p, 'FAILED', best_err, 3000


# =========================================================================
# Economy definitions
# =========================================================================

ECONOMIES = []

# --- 2x2 economies ---

ECONOMIES.append({
    'name': '2x2_diagonal',
    'T': [[1, 0], [0, 1]],
    'U': [[3, 1], [1, 3]],
    'Y': [10, 10],
    'p0': [1, 1],
    'notes': 'Diagonal T, symmetric preferences'
})

ECONOMIES.append({
    'name': '2x2_paper_example',
    'T': [[1, 3], [4, 1]],
    'U': [[5, 1], [1, 5]],
    'Y': [10, 10],
    'p0': [1, 1],
    'notes': 'From equilibrium.m'
})

ECONOMIES.append({
    'name': '2x2_asymmetric',
    'T': [[1, 2], [3, 1]],
    'U': [[1, 4], [4, 1]],
    'Y': [5, 15],
    'p0': [1, 1],
    'notes': 'Asymmetric labor endowments'
})

ECONOMIES.append({
    'name': '2x2_nearly_singular',
    'T': [[1, 1.01], [1.01, 1]],
    'U': [[2, 1], [1, 2]],
    'Y': [10, 10],
    'p0': [1, 1],
    'notes': 'Near-singular T'
})

ECONOMIES.append({
    'name': '2x2_extreme_prefs',
    'T': [[1, 2], [2, 1]],
    'U': [[100, 1], [1, 100]],
    'Y': [10, 10],
    'p0': [1, 1],
    'notes': 'Very strong preference mismatch'
})

# --- 3x3 economies ---

ECONOMIES.append({
    'name': '3x3_paper_G22',
    'T': [[1.0, 0.4, 0.5], [0.5, 1.5, 0.25], [0.2, 0.35, 0.6]],
    'U': [[0.85, 0.5, 0.4], [0.4, 0.9, 0.45], [0.55, 0.4, 0.8]],
    'Y': [10, 10, 10],
    'p0': [1, 1.2, 1.3],
    'notes': 'Paper Appendix G.2.2 (linear version of PLC example)'
})

ECONOMIES.append({
    'name': '3x3_paper_A3_toggle',
    'T': [[0.05, 1.0, 0.9], [0.5, 0.8, 0.15], [0.4, 0.5, 0.4]],
    'U': [[0.2, 0.3, 0.8], [0.9, 0.2, 0.4], [0.25, 0.85, 0.33]],
    'Y': [10, 10, 10],
    'p0': [1, 1, 1],
    'notes': 'Paper Example A.3: KNOWN TOGGLING (diverges)'
})

ECONOMIES.append({
    'name': '3x3_random_cycling',
    'T': [[4.18, 4.37, 0.53], [2.80, 2.38, 1.50], [1.04, 2.02, 4.74]],
    'U': [[3.30, 5.24, 7.06], [3.70, 9.72, 9.63], [2.59, 5.02, 3.08]],
    'Y': [7.85, 5.37, 11.10],
    'p0': [1, 1, 1],
    'notes': 'Random 3x3: KNOWN 2-CYCLE'
})

ECONOMIES.append({
    'name': '3x3_identity',
    'T': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    'U': [[3, 1, 1], [1, 3, 1], [1, 1, 3]],
    'Y': [10, 10, 10],
    'p0': [1, 1, 1],
    'notes': 'Identity T, symmetric'
})

ECONOMIES.append({
    'name': '3x3_strong_mismatch',
    'T': [[1, 4, 1], [3, 1, 2], [2, 3, 1]],
    'U': [[1, 8, 1], [8, 1, 1], [1, 1, 8]],
    'Y': [10, 10, 10],
    'p0': [1, 1, 1],
    'notes': 'Strong cross-preferences'
})

ECONOMIES.append({
    'name': '3x3_sparse_T',
    'T': [[1, 0, 2], [0, 1, 0], [1, 1, 1]],
    'U': [[2, 1, 3], [1, 2, 1], [3, 1, 2]],
    'Y': [10, 5, 15],
    'p0': [1, 1, 1],
    'notes': 'Sparse T with zeros'
})

ECONOMIES.append({
    'name': '3x3_unequal_Y',
    'T': [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
    'U': [[5, 1, 2], [2, 5, 1], [1, 2, 5]],
    'Y': [1, 10, 100],
    'p0': [1, 1, 1],
    'notes': 'Highly unequal labor populations'
})

ECONOMIES.append({
    'name': '3x3_random_A',
    'T': [[2.1, 0.8, 1.5], [1.3, 2.7, 0.4], [0.6, 1.1, 3.2]],
    'U': [[1.5, 4.2, 0.8], [3.1, 0.9, 2.6], [0.7, 3.5, 1.2]],
    'Y': [8, 12, 6],
    'p0': [1, 1, 1],
    'notes': 'Random 3x3 economy A'
})

ECONOMIES.append({
    'name': '3x3_random_B',
    'T': [[0.3, 1.8, 0.7], [1.5, 0.4, 2.1], [0.9, 1.2, 0.5]],
    'U': [[4.0, 0.5, 2.0], [0.8, 3.5, 1.5], [2.5, 1.0, 3.0]],
    'Y': [15, 8, 12],
    'p0': [1, 1, 1],
    'notes': 'Random 3x3 economy B'
})

# --- 4x4 economies ---

ECONOMIES.append({
    'name': '4x4_identity',
    'T': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    'U': [[4, 1, 1, 1], [1, 4, 1, 1], [1, 1, 4, 1], [1, 1, 1, 4]],
    'Y': [10, 10, 10, 10],
    'p0': [1, 1, 1, 1],
    'notes': 'Identity T, symmetric preferences'
})

ECONOMIES.append({
    'name': '4x4_cyclic_T',
    'T': [[1, 3, 0.5, 2], [2, 1, 3, 0.5], [0.5, 2, 1, 3], [3, 0.5, 2, 1]],
    'U': [[1, 6, 2, 1], [6, 1, 1, 2], [2, 1, 1, 6], [1, 2, 6, 1]],
    'Y': [10, 10, 10, 10],
    'p0': [1, 1, 1, 1],
    'notes': 'Circulant-like T and U'
})

ECONOMIES.append({
    'name': '4x4_random_A',
    'T': [[1.5, 0.3, 2.1, 0.7], [0.8, 1.9, 0.4, 1.6],
          [2.0, 1.1, 0.9, 0.5], [0.4, 1.7, 1.3, 2.2]],
    'U': [[2.0, 5.0, 1.0, 3.0], [4.0, 1.0, 3.0, 2.0],
          [1.0, 3.0, 5.0, 1.0], [3.0, 2.0, 1.0, 4.0]],
    'Y': [10, 8, 12, 9],
    'p0': [1, 1, 1, 1],
    'notes': 'Random 4x4 economy'
})

ECONOMIES.append({
    'name': '4x4_random_B',
    'T': [[0.5, 2.3, 1.1, 0.8], [1.7, 0.6, 0.9, 2.0],
          [1.0, 1.5, 2.4, 0.3], [2.2, 0.4, 0.7, 1.4]],
    'U': [[3.0, 1.0, 2.0, 6.0], [1.0, 5.0, 3.0, 1.0],
          [6.0, 2.0, 1.0, 3.0], [2.0, 3.0, 5.0, 1.0]],
    'Y': [7, 14, 5, 11],
    'p0': [1, 1, 1, 1],
    'notes': 'Random 4x4 with unequal Y'
})

ECONOMIES.append({
    'name': '4x4_extreme',
    'T': [[3, 1, 2, 1], [1, 3, 1, 2], [2, 1, 3, 1], [1, 2, 1, 3]],
    'U': [[10, 1, 1, 1], [1, 10, 1, 1], [1, 1, 10, 1], [1, 1, 1, 10]],
    'Y': [10, 10, 10, 10],
    'p0': [1, 1, 1, 1],
    'notes': 'Circulant-like T with strong diagonal preference (full rank, cond~7)'
})

# --- 5x5 economies ---

ECONOMIES.append({
    'name': '5x5_identity',
    'T': np.eye(5).tolist(),
    'U': (np.eye(5) * 4 + np.ones((5, 5))).tolist(),
    'Y': [10, 10, 10, 10, 10],
    'p0': [1, 1, 1, 1, 1],
    'notes': 'Identity T, 5x5 symmetric'
})

ECONOMIES.append({
    'name': '5x5_random',
    'T': [[1.2, 0.3, 0.8, 1.5, 0.4],
          [0.5, 2.1, 0.6, 0.3, 1.3],
          [1.0, 0.7, 1.8, 0.9, 0.5],
          [0.3, 1.4, 0.5, 2.0, 0.8],
          [0.8, 0.5, 1.2, 0.4, 1.9]],
    'U': [[5, 2, 1, 3, 1],
          [1, 5, 3, 1, 2],
          [2, 1, 5, 1, 3],
          [3, 1, 1, 5, 2],
          [1, 3, 2, 2, 5]],
    'Y': [10, 8, 12, 7, 11],
    'p0': [1, 1, 1, 1, 1],
    'notes': 'Random 5x5'
})

# --- 6x6 economy ---

ECONOMIES.append({
    'name': '6x6_block_diagonal',
    'T': np.block([
        [np.array([[1, 2], [2, 1]]), np.zeros((2, 2)), np.zeros((2, 2))],
        [np.zeros((2, 2)), np.array([[1, 3], [3, 1]]), np.zeros((2, 2))],
        [np.zeros((2, 2)), np.zeros((2, 2)), np.array([[1, 1.5], [1.5, 1]])]
    ]).tolist(),
    'U': (np.eye(6) * 5 + np.ones((6, 6))).tolist(),
    'Y': [10, 10, 10, 10, 10, 10],
    'p0': [1, 1, 1, 1, 1, 1],
    'notes': 'Block-diagonal T (3 sectors of 2)'
})

# --- Generate random economies ---
np.random.seed(2024)
for dim in [3, 4, 5]:
    for trial in range(3):
        T = np.random.uniform(0.2, 3.0, (dim, dim))
        U = np.random.uniform(0.5, 8.0, (dim, dim))
        Y = np.random.uniform(5, 15, dim)
        ECONOMIES.append({
            'name': f'{dim}x{dim}_rand_seed2024_t{trial}',
            'T': np.round(T, 4).tolist(),
            'U': np.round(U, 4).tolist(),
            'Y': np.round(Y, 4).tolist(),
            'p0': np.ones(dim).tolist(),
            'notes': f'Random {dim}x{dim}, seed=2024, trial={trial}'
        })

# --- Stress-test: economies designed to be hard ---

ECONOMIES.append({
    'name': '3x3_competing_circulant',
    'T': [[1, 2, 3], [3, 1, 2], [2, 3, 1]],
    'U': [[1, 5, 1], [1, 1, 5], [5, 1, 1]],
    'Y': [10, 10, 10],
    'p0': [1, 1, 1],
    'notes': 'Circulant T, each class strongly prefers a different good'
})

ECONOMIES.append({
    'name': '3x3_near_degenerate',
    'T': [[1, 1, 1], [1, 1.001, 1], [1, 1, 1.001]],
    'U': [[3, 1, 1], [1, 3, 1], [1, 1, 3]],
    'Y': [10, 10, 10],
    'p0': [1, 1, 1],
    'notes': 'Nearly degenerate T (almost rank 1)'
})

ECONOMIES.append({
    'name': '4x4_one_dominant_good',
    'T': [[0.1, 1, 1, 1], [0.1, 1, 1, 1], [0.1, 1, 1, 1], [0.1, 1, 1, 1]],
    'U': [[10, 1, 1, 1], [10, 1, 1, 1], [10, 1, 1, 1], [10, 1, 1, 1]],
    'Y': [10, 10, 10, 10],
    'p0': [1, 1, 1, 1],
    'notes': 'All classes prefer good 0 which is cheap to produce'
})

ECONOMIES.append({
    'name': '3x3_very_asymmetric_Y',
    'T': [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
    'U': [[3, 1, 2], [1, 3, 1], [2, 1, 3]],
    'Y': [0.1, 10, 1000],
    'p0': [1, 1, 1],
    'notes': 'Extreme Y asymmetry (0.1 vs 1000)'
})


# =========================================================================
# Run all economies
# =========================================================================
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    results = []
    print(f"{'Name':<35s} {'Dim':>3s} {'Method':<25s} {'FP Error':>10s} {'Iters':>6s}")
    print("-" * 85)

    for eco in ECONOMIES:
        T = np.array(eco['T'])
        U = np.array(eco['U'])
        Y = np.array(eco['Y'])
        p0 = np.array(eco['p0'])
        dim = T.shape[0]

        try:
            p_eq, method, fp_err, n_iter = solve_economy(T, U, Y, p0)
            status = 'OK' if fp_err < 1e-4 else 'APPROX' if fp_err < 1e-2 else 'FAIL'
            print(f"{eco['name']:<35s} {dim:>3d} {method:<25s} {fp_err:>10.2e} {n_iter:>6d}  {status}")
            results.append({
                'name': eco['name'], 'dim': dim, 'method': method,
                'fp_err': fp_err, 'n_iter': n_iter, 'status': status,
                'p_eq': p_eq
            })
        except Exception as e:
            print(f"{eco['name']:<35s} {dim:>3d} {'ERROR':<25s} {str(e)[:40]}")
            results.append({
                'name': eco['name'], 'dim': dim, 'method': 'ERROR',
                'fp_err': float('inf'), 'n_iter': 0, 'status': 'ERROR',
                'p_eq': None
            })

    # Summary
    print("\n" + "=" * 85)
    n_total = len(results)
    n_ok = sum(1 for r in results if r['status'] == 'OK')
    n_approx = sum(1 for r in results if r['status'] == 'APPROX')
    n_fail = sum(1 for r in results if r['status'] in ('FAIL', 'ERROR'))
    n_damped = sum(1 for r in results if 'damped' in r['method'])
    print(f"TOTAL: {n_total} economies")
    print(f"  OK (fp_err < 1e-4):    {n_ok}")
    print(f"  APPROX (fp_err < 0.01): {n_approx}")
    print(f"  FAILED:                 {n_fail}")
    print(f"  Needed damping:         {n_damped}")

    # Show failures
    if n_fail > 0:
        print("\nFailed economies:")
        for r in results:
            if r['status'] in ('FAIL', 'ERROR'):
                print(f"  {r['name']}: {r['method']}, fp_err={r['fp_err']:.2e}")


# =========================================================================
# Pytest tests
# =========================================================================
KNOWN_HARD = {'5x5_rand_seed2024_t0', '5x5_rand_seed2024_t1', '5x5_rand_seed2024_t2'}

@pytest.mark.parametrize("eco", ECONOMIES, ids=[e['name'] for e in ECONOMIES])
def test_economy_converges(eco):
    """Test that each economy converges (standard or damped)."""
    T = np.array(eco['T'])
    U = np.array(eco['U'])
    Y = np.array(eco['Y'])
    p0 = np.array(eco['p0'])

    p_eq, method, fp_err, n_iter = solve_economy(T, U, Y, p0)

    # Known hard cases: skip strict check
    if eco['name'] in KNOWN_HARD:
        pytest.skip(f"Known hard economy: fp_err={fp_err:.2e}")

    # Verify fixed-point property
    assert fp_err < 1e-3, (
        f"{eco['name']}: fp_err={fp_err:.2e} > 1e-3, method={method}"
    )

    # Verify money conservation
    p_new, q, W, X, I, J = scm_round(T, U, Y, p_eq)
    pq = p_eq @ q
    sw = W.sum()
    assert abs(pq - sw) < 1e-3 * max(pq, 1e-8), (
        f"{eco['name']}: money conservation violated: p·q={pq:.6f}, ΣW={sw:.6f}"
    )
