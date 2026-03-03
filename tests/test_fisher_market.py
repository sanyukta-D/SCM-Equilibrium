"""Tests for the linear-utility Fisher market solver."""

import numpy as np
import pytest
from scm import solve_fisher


# ── Test 2: Diagonal utility (analytical solution) ──

def test_fisher_diagonal():
    """Each class only values its own good; prices = budget / quantity."""
    U       = np.array([[2.0, 0.0], [0.0, 3.0]])
    q       = np.array([10.0, 10.0])
    budgets = np.array([6.0, 4.0])

    prices, X_money, X_units, bpb = solve_fisher(U, q, budgets)

    np.testing.assert_allclose(prices, [0.6, 0.4], atol=1e-3)
    assert X_units[0, 0] > 9.9 and X_units[0, 1] < 0.01
    assert X_units[1, 0] < 0.01 and X_units[1, 1] > 9.9
    assert abs(prices @ q - budgets.sum()) < 1e-3
    np.testing.assert_allclose(X_units @ prices, budgets, atol=1e-3)


# ── Test 3: Symmetric 2x2 (analytical solution) ──

def test_fisher_symmetric():
    """Symmetric utilities and equal budgets yield equal prices."""
    U       = np.array([[2.0, 1.0], [1.0, 2.0]])
    q       = np.array([1.0, 1.0])
    budgets = np.array([1.0, 1.0])

    prices, X_money, X_units, bpb = solve_fisher(U, q, budgets)

    np.testing.assert_allclose(prices, [1.0, 1.0], atol=1e-2)
    assert abs(prices @ q - budgets.sum()) < 1e-3
    np.testing.assert_allclose(X_units.sum(axis=0), q, atol=1e-3)
    np.testing.assert_allclose(X_units @ prices, budgets, atol=1e-2)


# ── Test 4: 3-class x 4-good from paper section G.1.3 (linear U) ──

def test_fisher_G13_structural():
    """Fisher market structural properties hold for the G.1.3 economy."""
    U = np.array([
        [0.80, 0.40, 0.50, 0.45],
        [0.30, 0.75, 0.20, 0.50],
        [0.25, 0.40, 0.80, 0.35],
    ])
    q       = np.array([12.0, 20.0, 25.0, 18.0])
    budgets = np.array([16.0, 7.0, 10.0])

    prices, X_money, X_units, bpb = solve_fisher(U, q, budgets)

    # Money conservation
    assert abs(prices @ q - budgets.sum()) < 1e-3

    # Market clearing
    np.testing.assert_allclose(X_units.sum(axis=0), q, atol=1e-2)

    # Budget exhaustion
    spent = np.array([prices @ X_units[i] for i in range(3)])
    np.testing.assert_allclose(spent, budgets, rtol=0.01, atol=0.1)

    # BPB optimality: each class buys only max bang-per-buck goods
    for i in range(3):
        best_bpb = bpb[i].max()
        for j in range(4):
            if X_units[i, j] > 0.1:
                assert abs(bpb[i, j] - best_bpb) < 1e-2 * best_bpb
