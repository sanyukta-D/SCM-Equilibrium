# MATLAB Reference Code

These are the original MATLAB implementations from the research codebase.
They are included for provenance and cross-reference only. The Python
package in `scm/` is the maintained implementation.

## File mapping

| MATLAB file          | Python equivalent              | Purpose                                      |
|----------------------|-------------------------------|----------------------------------------------|
| `fm.m`               | `scm/scm_round.py`           | One SCM round: production LP + wages + Fisher market + price update |
| `fisherm.m`          | `scm/fisher_market.py`       | Linear-utility Fisher market via Eisenberg-Gale |
| `equilibrium.m`      | `scm/equilibrium.py`         | Tatonnement loop (iterates fm.m until convergence) |
| `plcm.m`             | `scm/fisher_market_plc.py`   | PLC Fisher market (2-segment utilities)       |
| `plcmarket.m`        | `scm/scm_round_plc.py`      | One PLC SCM round                            |
| `plcequilibrium.m`   | `scm/equilibrium_plc.py`     | PLC tatonnement loop                         |
| `forone.m`           | *(eliminated)*                | Degeneracy workaround for 2x2 (handled natively by CLARABEL) |
| `plc.m`              | `scm/fisher_market_plc.py`   | PLC market variant                           |
| `FourPlayerMarket.m` | *(examples/)*                 | Multi-player example                         |
| `FeigningU.m`        | *(Task 3, pending)*           | Consumer Choice Game payoff computation       |
| `kkt.m`              | *(not needed)*                | KKT condition helper                         |
| `opti.m`             | *(not needed)*                | Optimization helper                          |
