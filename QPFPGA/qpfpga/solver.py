from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cvxpy.interface as intf
    import cvxpy.settings as s
    import cvxpy.reductions.solution as solution_mod
    from cvxpy.reductions.solvers import utilities
    from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
except Exception:
    intf = None
    s = None
    solution_mod = None
    utilities = None

    class QpSolver:
        pass

from .backend import QPFPGABackend, as_osqp_problem, default_backend
from .data import QPSolverOptions, QPSolverResult

Solution = getattr(solution_mod, "Solution", None)
failure_solution = getattr(solution_mod, "failure_solution", None)


class QPFPGA(QpSolver):
    """CVXPY QP solver adapter for the FPGA backend."""

    STATUS_MAP = {} if s is None else {
        "optimal": s.OPTIMAL,
        "optimal_inaccurate": s.OPTIMAL_INACCURATE,
        "user_limit": s.USER_LIMIT,
        "infeasible": s.INFEASIBLE,
        "infeasible_inaccurate": s.INFEASIBLE_INACCURATE,
        "unbounded": s.UNBOUNDED,
        "unbounded_inaccurate": s.UNBOUNDED_INACCURATE,
        "solver_error": s.SOLVER_ERROR,
        "not_implemented": s.SOLVER_ERROR,
    }

    def __init__(self, backend: QPFPGABackend | None = None) -> None:
        self.backend = backend or default_backend()

    def name(self) -> str:
        return "QPFPGA"

    def import_solver(self) -> None:
        return None

    def cite(self, data):
        return ""

    def _solver_options(self, solver_opts: dict[str, Any] | None) -> QPSolverOptions:
        solver_opts = solver_opts or {}
        return QPSolverOptions(
            sigma=float(solver_opts.get("sigma", 1e-2)),
            alpha=float(solver_opts.get("alpha", 1.8)),
            eps_abs=float(solver_opts.get("eps_abs", 1e-3)),
            eps_rel=float(solver_opts.get("eps_rel", 1e-3)),
            pcg_tol_fraction=float(solver_opts.get("pcg_tol_fraction", 1.0)),
            admm_max_iter=int(solver_opts.get("admm_max_iter", 5000)),
            pcg_max_iter=int(solver_opts.get("pcg_max_iter", 5)),
            adaptive_rho=bool(solver_opts.get("adaptive_rho", True)),
            measure_energy=bool(solver_opts.get("measure_energy", False)),
            extra={k: v for k, v in solver_opts.items() if k not in {
                "sigma", "alpha", "eps_abs", "eps_rel", "pcg_tol_fraction",
                "admm_max_iter", "pcg_max_iter", "adaptive_rho", "measure_energy"
            }},
        )

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        if s is None:
            raise ImportError("CVXPY is required to use QPFPGA as a solver backend.")
        if verbose:
            print("=== QPFPGA received CVXPY data ===")
            print("keys:", sorted(data.keys()))
            print(f"P: shape={data[s.P].shape}, nnz={data[s.P].nnz}")
            print(f"q: shape={np.asarray(data[s.Q]).shape}")
            print(f"A: shape={data[s.A].shape}, nnz={data[s.A].nnz}")
            print(f"b: shape={np.asarray(data[s.B]).shape}")
            print(f"F: shape={data[s.F].shape}, nnz={data[s.F].nnz}")
            print(f"g: shape={np.asarray(data[s.G]).shape}")
        qp_data = as_osqp_problem(data)
        options = self._solver_options(solver_opts)

        if verbose:
            print("\n=== OSQP Canonical Form (Hardware Input) ===")
            print(f"P: shape={qp_data.P.shape}, nnz={qp_data.P.nnz}")
            print(f"q: shape={qp_data.q.shape}")
            print(f"A: shape={qp_data.A.shape}, nnz={qp_data.A.nnz}")
            print(f"l: shape={qp_data.l.shape}")
            print(f"u: shape={qp_data.u.shape}")

        results = self.backend.solve(qp_data, options)

        if verbose:
            pcg_iters = results.extra_stats.get("pcg_iters", 0)
            avg_pcg = (pcg_iters / results.num_iters) if results.num_iters else 0.0
            bs_time = results.extra_stats.get("bitstream_time_s", 0.0) * 1000.0
            cpp_time = results.extra_stats.get("total_cpp_time_s", 0.0) * 1000.0
            setup_time = results.extra_stats.get("setup_time_ms", 0.0)
            hw_time = results.solve_time_s * 1000.0
            
            print("\n" + "="*50)
            print("           QPFPGA Hardware Execution Stats")
            print("="*50)
            print(f"Status           : {results.status.upper()}")
            print(f"ADMM Iterations  : {results.num_iters}")
            print(f"PCG Iterations   : {pcg_iters} (Avg: {avg_pcg:.1f}/ADMM)")
            print(f"Primal Residual  : {results.primal_residual:.5e}")
            print(f"Dual Residual    : {results.dual_residual:.5e}")
            print("-" * 50)
            print("Timing Breakdown:")
            if bs_time > 0:
                print(f"  Bitstream Load : {bs_time:>8.3f} ms")
            print(f"  C++ Total Call : {cpp_time:>8.3f} ms")
            print(f"    ├─ Data Prep : {setup_time:>8.3f} ms")
            print(f"    └─ HW Exec   : {hw_time:>8.3f} ms")
            print("="*50 + "\n")

            if options.measure_energy:
                core_e = results.extra_stats.get("core_energy_j", 0.0)
                aux_e = results.extra_stats.get("aux_energy_j", 0.0)
                fpga_e = results.extra_stats.get("fpga_energy_j", 0.0)
                board_e = results.extra_stats.get("board_energy_j", 0.0)
                
                # Calculate average power
                hw_time_s = results.solve_time_s
                avg_power = (fpga_e / hw_time_s) if hw_time_s > 0 else 0.0
                
                print("-" * 50)
                print("Energy Measurements:")
                print(f"  FPGA Core Logic : {core_e:.6f} J")
                print(f"  FPGA AUX        : {aux_e:.6f} J")
                print(f"  FPGA Total      : {fpga_e:.6f} J")
                print(f"  Board Total     : {board_e:.6f} J")
                print(f"  Avg FPGA Power  : {avg_power:.3f} W")
                
            print("="*50 + "\n")    

        data["_qpfpga_result"] = results
        return results

    def invert(self, solution: QPSolverResult, inverse_data):
        if s is None:
            raise ImportError("CVXPY is required to use QPFPGA as a solver backend.")
        if Solution is None or failure_solution is None:
            raise ImportError(
                "This CVXPY version does not expose the expected solution helpers; "
                "use the QPFPGA backend directly or upgrade/downgrade CVXPY."
            )
        attr = {
            s.SOLVE_TIME: solution.solve_time_s,
            s.NUM_ITERS: solution.num_iters,
            s.EXTRA_STATS: solution,
        }
        status = self.STATUS_MAP.get(solution.status, s.SOLVER_ERROR)
        
        if status in s.SOLUTION_PRESENT and solution.x is not None:
            primal_vars = {
                inverse_data[self.VAR_ID]: intf.DEFAULT_INTF.const_to_matrix(np.asarray(solution.x))
            }
            
            dual_vars = None 
            
            if solution.y is not None:
                y_arr = np.asarray(solution.y)
                n_eq = inverse_data[self.DIMS].zero
                
                eq_constrs = inverse_data.get(self.EQ_CONSTR, [])
                ineq_constrs = inverse_data.get(self.NEQ_CONSTR, [])

                try:
                    mapped_duals = utilities.get_dual_values(
                        y_arr[:n_eq],
                        utilities.extract_dual_value,
                        eq_constrs,
                    ) | utilities.get_dual_values(
                        y_arr[n_eq:],
                        utilities.extract_dual_value,
                        ineq_constrs,
                    )
                    
                    # Only apply mapping if it actually extracted values
                    if mapped_duals:
                        dual_vars = mapped_duals

                except Exception:
                    pass 
                    
            opt_val = solution.obj_val
            return Solution(status, opt_val, primal_vars, dual_vars, attr)

        return failure_solution(status, attr)