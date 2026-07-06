from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass
class QPData:
    P: sp.csc_matrix
    q: np.ndarray
    A: sp.csc_matrix
    l: np.ndarray
    u: np.ndarray

    def osqp_style(self) -> tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray]:
        return self.P, self.q, self.A, self.l, self.u


@dataclass
class QPSolverOptions:
    sigma: float = 1e-2
    alpha: float = 1.8
    eps_abs: float = 1e-3
    eps_rel: float = 1e-3
    pcg_tol_fraction: float = 1.0
    admm_max_iter: int = 2000
    pcg_max_iter: int = 5
    adaptive_rho: bool = False
    measure_energy: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class QPSolverResult:
    status: str
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    obj_val: float | None = None
    primal_residual: float | None = None
    dual_residual: float | None = None
    num_iters: int | None = None
    solve_time_s: float | None = None
    extra_stats: dict[str, Any] = field(default_factory=dict)