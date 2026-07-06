from __future__ import annotations

import os
import sys
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import time

if "QPFPGA_LIBRARY" not in os.environ:
    expected_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "libqpfpga.so"))
    if os.path.exists(expected_path):
        os.environ["QPFPGA_LIBRARY"] = expected_path
    else:
        print(f"ERROR: Could not find libqpfpga.so at {expected_path}")
        print("Please compile the backend or set QPFPGA_LIBRARY manually.")
        sys.exit(1)

import qpfpga  # registration of cp.QPFPGA

def main() -> None:
    np.random.seed(123)
    
    N = 1024
    
    P_diag = np.random.uniform(1.0, 5.0, N).astype(np.float32)
    P_data = np.diag(P_diag)
    
    q_data = np.random.uniform(-2.0, 2.0, N).astype(np.float32)

    print("=== Problem assembly output ===")
    
    x = cp.Variable(N)
    
    objective = cp.Minimize(0.5 * cp.quad_form(x, P_data) + q_data @ x)
    
    constraints = [x >= 0, x <= 1]
    problem = cp.Problem(objective, constraints)

    print(f"n_var: {N}")
    print("n_eq: 0")
    print(f"n_ineq: {N * 2}")
    
    print("\n=== Solver call (using CVXPY interface) ===")
    
    print("Calling problem.solve...")

    t0 = time.perf_counter()
    value = problem.solve(solver="QPFPGA", verbose=True, measure_energy=True)
    t1 = time.perf_counter()
    print(f"Total solve time (CVXPY call): {(t1 - t0) * 1000:.3f} ms")

    print(f"returned status: {problem.status}")

    if x.value is not None:
        print(f"returned x (first 4): {x.value[:4].tolist()}")
    else:
        print("returned x: None")

    print(f"returned objective: {value:.6f}")
            

if __name__ == "__main__":
    main()