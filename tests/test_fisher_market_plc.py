"""Tests for the PLC Fisher market solver."""

import numpy as np
import pytest
from scm import solve_fisher_plc, solve_fisher_plc_3d


# ── Test 9: PLC Fisher market, paper section G.1.3 (exact match) ──

def test_plc_fisher_G13():
    """PLC Fisher market prices and allocations match paper section G.1.3."""
    U3d = np.array([
        [[0.80, 0.60], [0.40, 0.25], [0.50, 0.20], [0.45, 0.30]],
        [[0.30, 0.20], [0.75, 0.40], [0.20, 0.10], [0.50, 0.40]],
        [[0.25, 0.15], [0.40, 0.36], [0.80, 0.40], [0.35, 0.20]],
    ])
    L3d = np.array([
        [[3,   6  ], [4,   4  ], [7,  3  ], [4,   3.5]],
        [[2.5, 2  ], [3.8, 5  ], [6,  8  ], [0.5, 4  ]],
        [[9,   8  ], [6,   5  ], [1.5,4  ], [3,   8  ]],
    ])
    q       = np.array([12.0, 20.0, 25.0, 18.0])
    budgets = np.array([16.0, 7.0, 10.0])
    p_expected = np.array([0.735, 0.368, 0.408, 0.368])

    prices, X_money, X1_units, X2_units, bpb1, bpb2 = solve_fisher_plc_3d(
        U3d, L3d, q, budgets
    )

    # Prices match paper
    np.testing.assert_allclose(prices, p_expected, rtol=0.02, atol=0.005)

    # Money conservation
    assert abs(prices @ q - budgets.sum()) < 0.05

    # Market clearing
    X_total = X1_units + X2_units
    np.testing.assert_allclose(X_total.sum(axis=0), q, atol=0.05)

    # Segment-1 capacity not exceeded
    L1 = L3d[:, :, 0]
    assert np.all(X1_units <= L1 + 1e-5)

    # Segment-1 allocations match paper
    X1_expected = np.array([
        [3.0, 4.0, 7.0, 4.0],
        [0.0, 3.8, 0.0, 0.5],
        [0.0, 6.0, 1.5, 0.0],
    ])
    np.testing.assert_allclose(X1_units, X1_expected, rtol=0.02, atol=0.1)

    # BPB ordering: seg-2 only after seg-1 is full
    for i in range(3):
        for j in range(4):
            if X2_units[i, j] > 0.01:
                assert X1_units[i, j] >= L1[i, j] - 0.05


# ── Test 14: PLC Fisher, 2-class diagonal (analytical) ──

def test_plc_fisher_diagonal():
    """Diagonal PLC utilities give the same result as the linear case."""
    U1 = np.array([[1.0, 0.001], [0.001, 1.0]])
    U2 = np.array([[0.5, 0.001], [0.001, 0.5]])
    L1 = np.array([[5.0, 0.0], [0.0, 5.0]])
    q       = np.array([10.0, 10.0])
    budgets = np.array([6.0, 4.0])

    prices, X_money, X1_units, X2_units, bpb1, bpb2 = solve_fisher_plc(
        U1, U2, L1, q, budgets
    )

    np.testing.assert_allclose(prices, [0.6, 0.4], atol=0.01)
    assert abs(prices @ q - budgets.sum()) < 0.01

    X_total = X1_units + X2_units
    np.testing.assert_allclose(X_total.sum(axis=0), q, atol=0.01)
    assert X_total[0, 0] > 9.8 and X_total[1, 1] > 9.8
    assert np.all(X1_units <= L1 + 1e-5)
