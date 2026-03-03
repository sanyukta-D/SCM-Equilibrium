#!/usr/bin/env python3
"""
cli.py  –  Command-line interface for the SCM equilibrium solver

Usage:
    python cli.py examples/economy_2x2_linear.json
    python cli.py examples/economy_3x4_plc.json --max-iter 300 --tol 1e-8

JSON format (linear):
    {
        "type": "linear",
        "T": [[1, 0], [1, 1]],
        "U": [[1, 0.8], [0.8, 1]],
        "Y": [2, 4],
        "p_init": [1, 1]
    }

JSON format (PLC):
    {
        "type": "plc",
        "T":  [[1, 0], [1, 1]],
        "U1": [[1, 0.8], [0.8, 1]],
        "U2": [[0.5, 0.4], [0.4, 0.5]],
        "L1": [[1, 1], [1, 1]],
        "Y":  [2, 4],
        "p_init": [1, 1]
    }
"""

import argparse
import json
import numpy as np

from scm import (
    compute_equilibrium, print_equilibrium, check_scm_equilibrium,
    compute_equilibrium_plc, print_equilibrium_plc, check_plc_equilibrium,
)


def load_economy(path):
    """Load economy parameters from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    eco = {
        "type":   data.get("type", "linear"),
        "T":      np.array(data["T"], dtype=float),
        "Y":      np.array(data["Y"], dtype=float),
        "p_init": np.array(data["p_init"], dtype=float),
    }

    if eco["type"] == "linear":
        eco["U"] = np.array(data["U"], dtype=float)
    elif eco["type"] == "plc":
        eco["U1"] = np.array(data["U1"], dtype=float)
        eco["U2"] = np.array(data["U2"], dtype=float)
        eco["L1"] = np.array(data["L1"], dtype=float)
    else:
        raise ValueError(f"Unknown economy type: {eco['type']!r}. Use 'linear' or 'plc'.")

    return eco


def main():
    parser = argparse.ArgumentParser(
        description="Solve an SM equilibrium from a JSON economy file."
    )
    parser.add_argument("economy_file", help="Path to JSON economy file")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Maximum tatonnement iterations (default: 200)")
    parser.add_argument("--tol", type=float, default=1e-7,
                        help="Convergence tolerance (default: 1e-7)")
    parser.add_argument("--verify-tol", type=float, default=1e-3,
                        help="Tolerance for equilibrium condition checks (default: 1e-3)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip equilibrium verification")
    args = parser.parse_args()

    # Load
    eco = load_economy(args.economy_file)
    m, n = eco["T"].shape

    print("=" * 60)
    print("  SCM Equilibrium Solver")
    print("=" * 60)
    print(f"\n  Economy file: {args.economy_file}")
    print(f"  Type: {eco['type']}")
    print(f"  Size: {m} labour classes x {n} goods")
    print(f"  Max iterations: {args.max_iter},  tolerance: {args.tol}")

    # Solve
    if eco["type"] == "linear":
        result = compute_equilibrium(
            eco["T"], eco["U"], eco["Y"], eco["p_init"],
            max_iter=args.max_iter, tol=args.tol
        )
        print()
        print_equilibrium(result, label="Equilibrium result")

        if not args.no_verify:
            print("\n  Equilibrium condition checks:")
            checks, all_pass = check_scm_equilibrium(
                result, eco["T"], eco["U"], eco["Y"],
                tol=args.verify_tol, verbose=True
            )
            print(f"\n  {'ALL CONDITIONS PASS' if all_pass else 'SOME CONDITIONS FAILED'}")

    elif eco["type"] == "plc":
        result = compute_equilibrium_plc(
            eco["T"], eco["U1"], eco["U2"], eco["L1"], eco["Y"], eco["p_init"],
            max_iter=args.max_iter, tol=args.tol
        )
        print()
        print_equilibrium_plc(result, label="PLC Equilibrium result")

        if not args.no_verify:
            print("\n  Equilibrium condition checks:")
            checks, all_pass = check_plc_equilibrium(
                result, eco["T"], eco["U1"], eco["U2"], eco["L1"], eco["Y"],
                tol=args.verify_tol, verbose=True
            )
            print(f"\n  {'ALL CONDITIONS PASS' if all_pass else 'SOME CONDITIONS FAILED'}")

    print("=" * 60)


if __name__ == "__main__":
    main()
