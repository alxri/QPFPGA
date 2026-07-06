import cvxpy as cp

from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS, SOLVER_MAP_QP

from .data import QPData, QPSolverOptions
from .solver import QPFPGA

# --- Global Solver Registration ---

if "QPFPGA" not in QP_SOLVERS:
    QP_SOLVERS.append("QPFPGA")

if "QPFPGA" not in INSTALLED_SOLVERS:
    INSTALLED_SOLVERS.append("QPFPGA")

# Register instance in the solver map for dispatch.
SOLVER_MAP_QP["QPFPGA"] = QPFPGA()

# Inject into cvxpy namespace
setattr(cp, "QPFPGA", QPFPGA)

__all__ = ["QPData", "QPSolverOptions", "QPFPGA"]