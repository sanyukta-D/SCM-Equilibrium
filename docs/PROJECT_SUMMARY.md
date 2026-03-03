# SCM Python Migration — Project Summary

---

## 1. What You Set Out to Do

**End goal (Citrini project):** Use the Simple Closed Model (SCM) from Deshpande & Sohoni (arXiv:2109.09248) to computationally test the economic claims in the Citrini Research report "The 2028 Global Intelligence Crisis" (Feb 2026) — a viral report that moved markets but lacked a formal macroeconomic model. The SCM provides that model.

**Immediate prerequisite (what we've been doing):** Before any Citrini analysis can happen, the SCM computational engine must exist in Python, be verified against the paper's own worked examples, and correctly implement both the linear-utility and PLC (Piecewise-Linear-Concave) utility variants. The MATLAB codes existed but were research-grade, undocumented, and not suitable for systematic analysis.

**Tasked work so far:**
- **Task 1 (done):** Migrate the core MATLAB SCM to Python with 100% accuracy — verified against all documented examples.
- **Task 1 extension (done):** Add PLC utilities, run exhaustive equilibrium verification against the paper's theory, create more synthetic examples.
- **Task 2 (pending):** Alternative equilibrium computation when tâtonnement doesn't converge.
- **Task 3 (pending):** Citrini claim testing.

---

## 2. What the Original MATLAB Codes Had

The `codes set/` folder contained ~30 MATLAB `.m` files, most of which were research scratch work. The relevant core files were:

| MATLAB file | Purpose |
|---|---|
| `fm.m` | One SCM round: production LP → wages → Fisher market → new prices |
| `fisherm.m` | Linear-utility Fisher market (Eisenberg-Gale EG program via quadprog) |
| `plcm.m` / `plcmarket.m` / `plc.m` | PLC Fisher market (2-segment piecewise-linear-concave utilities) |
| `equilibrium.m` | Tâtonnement loop: iterate `fm.m` until price convergence |
| `plcequilibrium.m` | Same loop but calling PLC Fisher market |
| `fm2.m` | Variant of `fm.m` |
| `FourPlayerMarket.m` | Multi-player example |
| `FeigningU.m` / `grad*.m` | Consumer Choice Game (CCG) payoff gradient — for Task 3 |
| `kkt.m`, `opti.m`, `LSQ.m` | Optimization helpers |
| `RicardoEx.m`, `ppf.m` etc. | Ricardo/production frontier experiments |

**Key issues with the MATLAB code:**
- No documentation or usage guide.
- Mixed utility conventions (linear vs. PLC) across files without clear separation.
- Hard-coded examples with no reusable structure.
- No systematic testing — examples run manually, no pass/fail framework.
- The paper's documentation (`market code documentation.pdf`) described the algorithms and gave worked numerical examples, but these were never systematically checked against the code.

---

## 3. What the Documentation Says

Two key documents inform the implementation:

**`market code documentation.pdf`** (the paper's appendix-style code guide):
- **§G.1:** Fisher market with PLC utilities. Defines the Eisenberg-Gale (EG) convex program for 2-segment PLC utilities, with a fully worked 3-class × 4-good example including exact prices, allocations, and bang-per-buck orderings.
- **§G.2.2:** Full SCM (tâtonnement) examples — three worked examples showing how production LP, wages, Fisher market, and price updates interact across iterations until convergence.
- Defines what constitutes an **SM equilibrium**: a price vector p such that applying one SCM round returns the same p. Characterized by 10 conditions: money conservation, price non-negativity, labour feasibility, production non-negativity, market clearing, budget exhaustion, wage consistency, bang-per-buck optimality, fixed-point, production optimality.

**`cowork_prompt_citrini_scm.md`** (the project brief):
- Details the five Citrini claims to test (Ghost GDP, wage compression, friction elimination, skill premium, tax revenue collapse).
- Specifies the economy parameterizations (T, Y, U matrices) for each test.
- Points to the "soap market" (paper §4.3/6.1) as the template for Claim 3 (friction via CCG).
- Notes that CCG payoff computation and zone boundary identification are also needed.

---

## 4. What Has Been Done

### Task 1: Core Python Migration

**Files created in `scm_python/`:**

| Python file | Role | MATLAB equivalent |
|---|---|---|
| `production.py` | Production LP + wage computation | `fm.m` (production side) |
| `fisher_market.py` | Linear-utility EG Fisher market | `fisherm.m` |
| `scm_round.py` | One full SCM round (linear) | `fm.m` |
| `scm_equilibrium.py` | Tâtonnement loop + cycle detection | `equilibrium.m` |
| `USAGE.md` | Usage guide with examples | (none in MATLAB) |

**Verification:** Tests 1–8 cover every documented numerical example from the paper — production LP (§G.2.2 ex1), Fisher markets (§G.1.3 linear, diagonal, symmetric), full SCM rounds, full equilibrium convergence, and wage dual consistency.

### Task 1 Extension: PLC Support + Exhaustive Verification

**Files created:**

| Python file | Role |
|---|---|
| `fisher_market_plc.py` | 2-segment PLC EG Fisher market via cvxpy/CLARABEL |
| `scm_round_plc.py` | One SCM round with PLC utilities |
| `scm_equilibrium_plc.py` | PLC tâtonnement loop |
| `verify_equilibrium.py` | Reusable equilibrium condition checker (10 conditions for linear, 11 for PLC) |

**`verify.py` expanded to 17 tests:**

| Tests | What they cover |
|---|---|
| 1–8 | Original Task 1 (linear SCM, all paper examples) |
| 9 | PLC Fisher market — §G.1.3 exact paper example (prices, seg-1 alloc, seg-2 totals, BPB ordering) |
| 10 | Exhaustive 10-condition check on 2×2 linear equilibrium |
| 11 | Exhaustive check on §G.2.2 example 3 |
| 12 | Synthetic diagonal SCM with known exact solution |
| 13 | Synthetic 3×3 asymmetric SCM — generated + exhaustive verify |
| 14 | PLC Fisher market analytical 2-class diagonal case |
| 15 | PLC `scm_round` one-round structural check |
| 16 | PLC equilibrium 2×2 with exhaustive 11-condition PLC check |
| 17 | Batch of 5 synthetic economies — all conditions checked |

**Current status: 17/17 tests PASS.**

### Key Technical Discoveries During Verification

**EG program degeneracy:** For the paper's §G.1.3 PLC example, the EG optimal face is degenerate — goods j=1 and j=3 have equal prices (0.368), making class 2 indifferent between their seg-2 allocations. CLARABEL finds a valid but different primal allocation than the paper's (actually with slightly higher EG objective). The dual prices (0.735, 0.368, 0.408, 0.368) are exact and match the paper. This is a known interior-point phenomenon: dual variables converge faster than primal variables in degenerate cases.

**Budget exhaustion and scale-sensitivity:** The Fisher market primal has ~1e-5 relative budget exhaustion error per solve. After de-normalising by total wages ΣW, this becomes |ΣW| × 1e-5 in absolute terms. For large economies (ΣW ~38), this gives ~4e-4 absolute error. The `verify_equilibrium.py` checker was updated to use **relative** budget exhaustion error (|spent − W[i]| / W[i]) to be scale-invariant.

**CLARABEL parameter names:** This version uses `tol_gap_abs`, `tol_gap_rel`, `tol_feas` (not `eps_abs`/`eps_rel`). Set to 1e-10 for the PLC solver.

**KKT price recovery bug (identified and reverted):** An attempt to use KKT-based price recovery (`p_j = U2[i,j] * b_i / v_i`) was correctly identified in concept but had a sign inversion bug (`/ lam_i` instead of `* lam_i`), and was reverted. The dual variable approach (Lagrange multipliers of market-clearing equality constraints) is correct and sufficient after money-conservation normalization.

---

## 5. What Remains (Task 2 and Task 3)

**Task 2:** Investigate alternative equilibrium computation when tâtonnement doesn't converge — e.g., direct fixed-point solvers, Newton methods on the SCM map, or convex reformulations.

**Task 3:** Citrini claim testing using the SCM engine:
- Set up the 4-class × 4-good base economy.
- Implement CCG payoff and gradient computation (port `FeigningU.m`, `grad*.m`).
- Run comparative statics for Claims 1–7.
- Produce the five key output plots (Ghost GDP feasibility, wage compression curves, friction zone map, skill premium trajectory, tax revenue divergence).
