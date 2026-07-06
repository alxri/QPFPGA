from __future__ import annotations

import ctypes
from dataclasses import asdict
import os
import time
import shlex
import subprocess
import shutil
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from .data import QPData, QPSolverOptions, QPSolverResult


class QPFPGABackend(Protocol):
    def solve(self, problem: QPData, options: QPSolverOptions) -> QPSolverResult:
        raise NotImplementedError


class MockBackend:
    """CPU development backend used when no FPGA runtime is available."""

    def solve(self, problem: QPData, options: QPSolverOptions) -> QPSolverResult:
        try:
            import osqp
        except Exception as exc:
            raise RuntimeError(
                "QPFPGA CPU backend requires the osqp package when no FPGA library is available."
            ) from exc

        print("Solving QP using MockBackend with OSQP...")
        solver = osqp.OSQP()
        solver.setup(
            P=problem.P,
            q=problem.q,
            A=problem.A,
            l=problem.l,
            u=problem.u,
            verbose=False,
            eps_abs=options.eps_abs,
            eps_rel=options.eps_rel,
            max_iter=options.admm_max_iter,
            alpha=options.alpha,
            adaptive_rho=options.adaptive_rho,
            polishing=True,
        )
        result = solver.solve(raise_error=False)

        status = _osqp_status_name(result.info.status_val)
        x = None if result.x is None else np.asarray(result.x, dtype=np.float32)
        y = None if result.y is None else np.asarray(result.y, dtype=np.float32)
        obj_val = float(result.info.obj_val) if status in _SOLUTION_PRESENT else None

        return QPSolverResult(
            status=status,
            x=x,
            y=y,
            obj_val=obj_val,
            primal_residual=float(getattr(result.info, "prim_res", np.nan)),
            dual_residual=float(getattr(result.info, "dual_res", np.nan)),
            num_iters=int(getattr(result.info, "iter", 0)),
            solve_time_s=float(getattr(result.info, "run_time", 0.0)),
            extra_stats={
                "backend": "osqp",
                "osqp_status": getattr(result.info, "status", None),
                "options": asdict(options),
            },
        )


_SOLUTION_PRESENT = {"optimal", "optimal_inaccurate"}


def _osqp_status_name(status_val: int) -> str:
    status_map = {
        1: "optimal",
        2: "optimal_inaccurate",
        3: "infeasible",
        4: "infeasible_inaccurate",
        5: "unbounded",
        6: "unbounded_inaccurate",
        7: "user_limit",
        8: "user_limit",
        10: "solver_error",
        11: "solver_error",
    }
    return status_map.get(int(status_val), "solver_error")


class CtypesBackend:
    """Loads a shared C++ backend and forwards OSQP-style data to it."""

    def __init__(self, library_path: str | Path, bitstream_path: str | Path | None = None) -> None:
        self.library_path = Path(library_path)
        self.bitstream_path = Path(bitstream_path) if bitstream_path is not None else None
        self._bitstream_loaded = False
        self._lib = ctypes.CDLL(str(self.library_path))
        self._configure_abi()

    def _configure_abi(self) -> None:
        self._lib.qpfpga_solve.argtypes = [
            ctypes.POINTER(QPFPGAProblemC),
            ctypes.POINTER(QPFPGAOptionsC),
            ctypes.POINTER(QPFPGAResultC),
        ]
        self._lib.qpfpga_solve.restype = ctypes.c_int32

    def _ensure_bitstream_loaded(self) -> None:
        if self.bitstream_path is None or self._bitstream_loaded:
            return
            
        if not self.bitstream_path.exists():
            raise FileNotFoundError(f"QPFPGA bitstream not found: {self.bitstream_path}")

        print(f"Loading bitstream via PYNQ environment: {self.bitstream_path.name}...")

        # Path to the isolated PYNQ python interpreter
        pynq_python = "/usr/local/share/pynq-venv/bin/python"

        # Python script to run inside that isolated environment
        pynq_script = f"""
from pynq import Overlay
import sys
try:
    ol = Overlay('{self.bitstream_path.resolve()}')
    print('Hardware initialized and clocks running!')
except Exception as e:
    print(f'PYNQ Overlay Error: {{e}}', file=sys.stderr)
    sys.exit(1)
"""

        # Execute the script with the required XRT environment variables
        import subprocess
        import os
        
        # Necessary with sudo (strips environment variables). PYNQ requires XILINX_XRT to find the FPGA.
        sub_env = os.environ.copy()
        if "XILINX_XRT" not in sub_env:
            sub_env["XILINX_XRT"] = "/usr"

        t0_bs = time.perf_counter()
        try:
            result = subprocess.run(
                [pynq_python, "-c", pynq_script],
                capture_output=True,
                text=True,
                check=True,
                env=sub_env
            )
            if result.stdout:
                print(result.stdout.strip())
                
            self._bitstream_loaded = True
            self._last_bitstream_time_s = time.perf_counter() - t0_bs
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to program FPGA via PYNQ!\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            raise RuntimeError("Bitstream programming failed. See errors above.")

    def solve(self, problem: QPData, options: QPSolverOptions) -> QPSolverResult:
        from pathlib import Path
        import os
        import time

        # Problem size check agains FPGA limits
        MAX_DIM = 32768
        n_vars = problem.q.shape[0]
        m_constrs = problem.A.shape[0]
        
        if n_vars > MAX_DIM or m_constrs > MAX_DIM:
            raise ValueError(
                f"QPFPGA Hardware Limit Exceeded: The accelerator supports a maximum of "
                f"{MAX_DIM} variables and {MAX_DIM} constraints.\n"
                f"Your problem has n={n_vars} variables and m={m_constrs} constraints.\n"
                f"Please reduce the problem size."
            )
        
        self._last_bitstream_time_s = 0.0

        ncols = problem.A.shape[1]
        nnz_per_col = problem.A.nnz / ncols if ncols > 0 else 0

        # Heuristic Table
        if nnz_per_col < 4:
            target_bs = Path("/home/xilinx/QPFPGA/bitstreams/admm_pcg_16384x16384_reshape1_pes12.bit")
            hw_tile_size = 16384
        elif nnz_per_col < 16:
            target_bs = Path("/home/xilinx/QPFPGA/bitstreams/admm_pcg_16384x16384_reshape8_pes20.bit")
            hw_tile_size = 16384
        else:
            target_bs = Path("/home/xilinx/QPFPGA/bitstreams/admm_pcg_8192x8192_reshape1_pes30.bit")
            hw_tile_size = 8192

        # Only reprogram if it's the first run OR if the optimal bitstream changed
        state_file = Path("/tmp/qpfpga_state.txt")
        loaded_bs_name = ""

        if state_file.exists():
            loaded_bs_name = state_file.read_text().strip()

        if loaded_bs_name != target_bs.name:
            print(f"\n[QPFPGA] Matrix A density: {nnz_per_col:.2f} nnz/col")
            print(f"[QPFPGA] Hardware changing: {loaded_bs_name or 'None'} -> {target_bs.name}")
            
            self.bitstream_path = target_bs
            self._bitstream_loaded = False 

            self._last_bitstream_time_s = 0.0
            self._ensure_bitstream_loaded()
            
            # Save the successful load to the state file for the NEXT python run
            state_file.write_text(target_bs.name)
        else:
            self.bitstream_path = target_bs
            self._bitstream_loaded = True

        p_indptr = np.ascontiguousarray(problem.P.indptr, dtype=np.int32)
        p_indices = np.ascontiguousarray(problem.P.indices, dtype=np.int32)
        p_data = np.ascontiguousarray(problem.P.data, dtype=np.float32)
        a_indptr = np.ascontiguousarray(problem.A.indptr, dtype=np.int32)
        a_indices = np.ascontiguousarray(problem.A.indices, dtype=np.int32)
        a_data = np.ascontiguousarray(problem.A.data, dtype=np.float32)
        q = np.ascontiguousarray(problem.q, dtype=np.float32)
        l = np.ascontiguousarray(problem.l, dtype=np.float32)
        u = np.ascontiguousarray(problem.u, dtype=np.float32)

        p_matrix = QPFPGACscMatrixC(
            nrows=problem.P.shape[0],
            ncols=problem.P.shape[1],
            nnz=problem.P.nnz,
            indptr=p_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            indices=p_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            data=p_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        a_matrix = QPFPGACscMatrixC(
            nrows=problem.A.shape[0],
            ncols=problem.A.shape[1],
            nnz=problem.A.nnz,
            indptr=a_indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            indices=a_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            data=a_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        problem_c = QPFPGAProblemC(
            P=p_matrix,
            A=a_matrix,
            q=q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            l=l.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            u=u.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n=problem.q.shape[0],
            m=problem.A.shape[0],
        )

        options_c = QPFPGAOptionsC(
            sigma=ctypes.c_float(options.sigma),
            alpha=ctypes.c_float(options.alpha),
            eps_abs=ctypes.c_float(options.eps_abs),
            eps_rel=ctypes.c_float(options.eps_rel),
            pcg_tol_fraction=ctypes.c_float(options.pcg_tol_fraction),
            admm_max_iter=ctypes.c_int32(options.admm_max_iter),
            pcg_max_iter=ctypes.c_int32(options.pcg_max_iter),
            adaptive_rho=ctypes.c_int32(int(options.adaptive_rho)),
            tile_size=ctypes.c_int32(hw_tile_size),
            measure_energy=ctypes.c_int32(int(options.measure_energy))
        )
        x_out = np.zeros(problem.q.shape[0], dtype=np.float32)
        y_out = np.zeros(problem.A.shape[0], dtype=np.float32)
        result_c = QPFPGAResultC()

        t0_cpp = time.perf_counter()
        status = self._lib.qpfpga_solve(
            ctypes.byref(problem_c),
            ctypes.byref(options_c),
            ctypes.byref(result_c),
        )

        total_cpp_time_s = time.perf_counter() - t0_cpp

        if result_c.x:
            x_out = np.ctypeslib.as_array(result_c.x, shape=(problem.q.shape[0],)).astype(np.float32, copy=True)
        if result_c.y:
            y_out = np.ctypeslib.as_array(result_c.y, shape=(problem.A.shape[0],)).astype(np.float32, copy=True)

        return QPSolverResult(
            status=_status_name_from_code(status),
            x=x_out if result_c.x else None,
            y=y_out if result_c.y else None,
            obj_val=float(result_c.objective_value) if result_c.status != 0 else None,
            primal_residual=float(result_c.primal_residual),
            dual_residual=float(result_c.dual_residual),
            num_iters=int(result_c.admm_iters),
            solve_time_s=float(result_c.solve_time_ms) / 1000.0,
            extra_stats={
                "pcg_iters": int(result_c.pcg_iters),
                "bitstream_time_s": self._last_bitstream_time_s,
                "total_cpp_time_s": total_cpp_time_s,
                "setup_time_ms": float(result_c.setup_time_ms),
                "core_energy_j": float(result_c.core_energy_j),
                "aux_energy_j": float(result_c.aux_energy_j),
                "fpga_energy_j": float(result_c.fpga_energy_j),
                "board_energy_j": float(result_c.board_energy_j),
                "library_path": str(self.library_path),
                "return_code": int(status),
                "options": asdict(options),
            },
        )


def default_backend(library_path: str | Path | None = None) -> QPFPGABackend:
    force_cpu = os.environ.get("QPFPGA_FORCE_CPU", "")
    if force_cpu and force_cpu not in {"0", "false", "False"}:
        return MockBackend()
        
    if library_path is None:
        library_path = os.environ.get("QPFPGA_LIBRARY")
        
    if library_path is None:
        print("QPFPGA_LIBRARY environment variable not set")
        raise RuntimeError("QPFPGA_LIBRARY environment variable must be set to load the FPGA backend (cpp/build).")
        
    try:
        print(f"Attempting to load QPFPGA library from {library_path}... (Hardware selected dynamically)")
        return CtypesBackend(library_path)
        
    except (OSError, AttributeError):
        print(f"Failed to load QPFPGA library from {library_path}; CPU fallback is not available.")
        raise RuntimeError(
             f"Failed to load QPFPGA library from {library_path}; CPU fallback is not available. Ensure that the library path is correct and points to a valid shared library."
        )


def as_osqp_problem(data: dict) -> QPData:
    try:
        import cvxpy.settings as s
    except ModuleNotFoundError as exc:
        raise ImportError("CVXPY settings are required to map solver data keys.") from exc

    P = sp.csc_matrix((data[s.P].data, data[s.P].indices, data[s.P].indptr), shape=data[s.P].shape)
    Aeq = sp.csc_matrix(data[s.A])
    Aineq = sp.csc_matrix(data[s.F])
    A = sp.vstack([Aeq, Aineq]).tocsc()
    q = np.asarray(data[s.Q], dtype=np.float32)
    b = np.asarray(data[s.B], dtype=np.float32)
    g = np.asarray(data[s.G], dtype=np.float32)
    l = np.concatenate([b, -1e17 * np.ones_like(g)])
    u = np.concatenate([b, g])
    return QPData(P=P, q=q, A=A, l=l, u=u)


class QPFPGACscMatrixC(ctypes.Structure):
    _fields_ = [
        ("nrows", ctypes.c_int32),
        ("ncols", ctypes.c_int32),
        ("nnz", ctypes.c_int32),
        ("indptr", ctypes.POINTER(ctypes.c_int32)),
        ("indices", ctypes.POINTER(ctypes.c_int32)),
        ("data", ctypes.POINTER(ctypes.c_float)),
    ]


class QPFPGAOptionsC(ctypes.Structure):
    _fields_ = [
        ("sigma", ctypes.c_float),
        ("alpha", ctypes.c_float),
        ("eps_abs", ctypes.c_float),
        ("eps_rel", ctypes.c_float),
        ("pcg_tol_fraction", ctypes.c_float),
        ("admm_max_iter", ctypes.c_int32),
        ("pcg_max_iter", ctypes.c_int32),
        ("adaptive_rho", ctypes.c_int32),
        ("tile_size", ctypes.c_int32),
        ("measure_energy", ctypes.c_int32),
    ]


class QPFPGAProblemC(ctypes.Structure):
    _fields_ = [
        ("P", QPFPGACscMatrixC),
        ("A", QPFPGACscMatrixC),
        ("q", ctypes.POINTER(ctypes.c_float)),
        ("l", ctypes.POINTER(ctypes.c_float)),
        ("u", ctypes.POINTER(ctypes.c_float)),
        ("n", ctypes.c_int32),
        ("m", ctypes.c_int32),
    ]


class QPFPGAResultC(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("admm_iters", ctypes.c_int32),
        ("pcg_iters", ctypes.c_int32),
        ("primal_residual", ctypes.c_float),
        ("dual_residual", ctypes.c_float),
        ("objective_value", ctypes.c_float),
        ("solve_time_ms", ctypes.c_double),
        ("setup_time_ms", ctypes.c_double),
        ("core_energy_j", ctypes.c_double),
        ("aux_energy_j", ctypes.c_double),
        ("fpga_energy_j", ctypes.c_double),
        ("board_energy_j", ctypes.c_double),
        ("x", ctypes.POINTER(ctypes.c_float)),
        ("y", ctypes.POINTER(ctypes.c_float)),
    ]


def _status_name_from_code(status_code: int) -> str:
    status_map = {
        1: "optimal",
        2: "optimal_inaccurate",
        3: "infeasible",
        4: "infeasible_inaccurate",
        5: "unbounded",
        6: "unbounded_inaccurate",
        7: "user_limit",
        10: "solver_error",
        -1: "not_implemented",
    }
    return status_map.get(int(status_code), "solver_error")