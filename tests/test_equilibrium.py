"""Tests for tatonnement equilibrium computation (linear utilities)."""

import numpy as np
import pytest
from scm import compute_equilibrium, scm_round, check_scm_equilibrium


# ── Test 6: 2x2 equilibrium convergence ──

def test_equilibrium_2x2_converges():
    """2x2 economy from equilibrium.m converges and satisfies structural conditions."""
    T = np.array([[1.0, 0.0], [1.0, 1.0]])
    U = np.array([[1.0, 0.8], [0.8, 1.0]])
    Y = np.array([2.0, 4.0])

    result = compute_equilibrium(T, U, Y, np.array([1.0, 1.0]),
                                  max_iter=200, tol=1e-7)

    assert result['status'] in ('converged', 'cycling')
    assert abs(result['p'] @ result['q'] - result['W'].sum()) < 0.01
    spent = np.array([result['p'] @ result['X'][i] for i in range(2)])
    np.testing.assert_allclose(spent, result['W'], rtol=0.02, atol=0.05)


# ── Test 7: G.2.2 example 3 structural invariants ──

def test_equilibrium_G22_ex3_structural():
    """G.2.2 example 3 satisfies all structural invariants under linear U."""
    T = np.array([
        [1.00, 0.10, 0.50],
        [0.50, 0.80, 0.25],
        [0.20, 0.35, 0.60],
    ])
    U = np.array([
        [0.85, 0.30, 0.40],
        [0.40, 0.90, 0.35],
        [0.30, 0.40, 0.80],
    ])
    Y = np.array([10.0, 10.0, 10.0])

    result = compute_equilibrium(T, U, Y, np.array([1.0, 1.2, 1.3]),
                                  max_iter=200, tol=1e-7)

    assert result['status'] in ('converged', 'cycling')

    p, q, W, X = result['p'], result['q'], result['W'], result['X']
    I, J = result['I'], result['J']

    # Money conservation
    assert abs(p @ q - W.sum()) < 0.01

    # Market clearing (active goods)
    np.testing.assert_allclose(X.sum(axis=0)[J], q[J], atol=1e-2)

    # Budget exhaustion (active classes)
    spent = np.array([p @ X[i] for i in range(3)])
    np.testing.assert_allclose(spent[I], W[I], rtol=0.02, atol=0.05)

    # Labour feasibility
    assert np.all(Y - T @ q >= -1e-4)

    # Fixed point
    p_fp, _, _, _, _, _ = scm_round(T, U, Y, p)
    assert np.max(np.abs(p_fp - p)) < 1e-4


# ── Test 10: Exhaustive 10-condition check (2x2) ──

def test_equilibrium_conditions_2x2():
    """All 10 SM equilibrium conditions hold for the 2x2 economy."""
    T = np.array([[1.0, 0.0], [1.0, 1.0]])
    U = np.array([[1.0, 0.8], [0.8, 1.0]])
    Y = np.array([2.0, 4.0])

    result = compute_equilibrium(T, U, Y, np.array([1.0, 1.0]),
                                  max_iter=200, tol=1e-7)
    checks, all_pass = check_scm_equilibrium(result, T, U, Y,
                                               tol=1e-4, verbose=False)
    assert all_pass, f"Failed conditions: {[k for k, (p, _) in checks.items() if not p]}"


# ── Test 11: Exhaustive conditions (G.2.2 ex3) ──

def test_equilibrium_conditions_G22():
    """All 10 conditions hold for G.2.2 example 3 (linear U)."""
    T = np.array([
        [1.00, 0.10, 0.50],
        [0.50, 0.80, 0.25],
        [0.20, 0.35, 0.60],
    ])
    U = np.array([
        [0.85, 0.30, 0.40],
        [0.40, 0.90, 0.35],
        [0.30, 0.40, 0.80],
    ])
    Y = np.array([10.0, 10.0, 10.0])

    result = compute_equilibrium(T, U, Y, np.array([1.0, 1.2, 1.3]),
                                  max_iter=200, tol=1e-7)
    checks, all_pass = check_scm_equilibrium(result, T, U, Y,
                                               tol=1e-3, verbose=False)
    assert all_pass, f"Failed conditions: {[k for k, (p, _) in checks.items() if not p]}"


# ── Test 12: Synthetic diagonal (known exact solution) ──

def test_equilibrium_diagonal_exact():
    """Diagonal T=I, U=I with equal Y gives equal prices and self-allocation."""
    T = np.eye(2)
    U = np.eye(2)
    Y = np.array([5.0, 5.0])

    result = compute_equilibrium(T, U, Y, np.array([1.0, 1.0]),
                                  max_iter=100, tol=1e-8)
    p = result['p']

    assert abs(p @ result['q'] - result['W'].sum()) < 1e-4
    assert abs(p[0] - p[1]) < 0.01 * max(p)

    checks, all_pass = check_scm_equilibrium(result, T, U, Y,
                                               tol=1e-4, verbose=False)
    assert all_pass


# ── Test 13: Synthetic 3x3 (generated, exhaustive verify) ──

def test_equilibrium_synthetic_3x3():
    """A custom 3x3 economy passes all equilibrium conditions."""
    T = np.array([
        [2.00, 0.20, 0.10],
        [0.10, 1.50, 0.20],
        [0.05, 0.10, 1.00],
    ])
    U = np.array([
        [0.80, 0.20, 0.10],
        [0.15, 0.75, 0.25],
        [0.10, 0.20, 0.85],
    ])
    Y = np.array([8.0, 6.0, 5.0])

    result = compute_equilibrium(T, U, Y, np.array([1.0, 1.0, 1.0]),
                                  max_iter=300, tol=1e-7)
    checks, all_pass = check_scm_equilibrium(result, T, U, Y,
                                               tol=1e-3, verbose=False)
    assert all_pass


# ── Test 17: Batch of 5 synthetic economies ──

BATCH_CASES = [
    ("2x2 dense",
     np.array([[1.0, 0.3], [0.4, 0.9]]),
     np.array([[0.7, 0.4], [0.3, 0.8]]),
     np.array([5.0, 6.0]),
     np.array([1.0, 1.0])),
    ("3x3 sparse-ish",
     np.array([[2.0, 0.5, 0.2], [0.3, 1.5, 0.4], [0.1, 0.2, 1.0]]),
     np.array([[0.8, 0.2, 0.1], [0.2, 0.7, 0.3], [0.1, 0.3, 0.9]]),
     np.array([8.0, 7.0, 5.0]),
     np.array([1.0, 1.2, 0.8])),
    ("2x2 near-diagonal",
     np.array([[1.0, 0.1], [0.2, 1.0]]),
     np.array([[0.9, 0.1], [0.1, 0.9]]),
     np.array([3.0, 8.0]),
     np.array([0.5, 1.5])),
    ("3x3 circulant",
     np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]),
     np.array([[0.6, 0.3, 0.1], [0.1, 0.6, 0.3], [0.3, 0.1, 0.6]]),
     np.array([6.0, 6.0, 6.0]),
     np.array([1.0, 1.0, 1.0])),
    ("G.2.2-variant",
     np.array([[1.0, 0.0, 0.5], [0.5, 1.5, 0.25], [0.2, 0.35, 0.6]]),
     np.array([[0.85, 0.50, 0.40], [0.40, 0.90, 0.45], [0.55, 0.40, 0.80]]),
     np.array([10.0, 10.0, 10.0]),
     np.array([1.0, 1.2, 1.3])),
]


@pytest.mark.parametrize("label,T,U,Y,p_init", BATCH_CASES,
                          ids=[c[0] for c in BATCH_CASES])
def test_batch_synthetic(label, T, U, Y, p_init):
    """Batch synthetic economy passes all equilibrium conditions."""
    result = compute_equilibrium(T, U, Y, p_init, max_iter=300, tol=1e-7)
    checks, all_pass = check_scm_equilibrium(result, T, U, Y,
                                               tol=1e-3, verbose=False)
    assert all_pass, (
        f"{label}: Failed conditions: "
        f"{[k for k, (p, _) in checks.items() if not p]}"
    )
