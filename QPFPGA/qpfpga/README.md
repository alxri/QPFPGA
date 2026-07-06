# QPFPGA Adapter Scaffold

This package is the CVXPY-facing layer for the FPGA QP solver.

Planned flow:

1. CVXPY canonicalizes a QP into `P, q, A, b, F, g`.
2. `qpfpga.solver.QPFPGA` converts that into OSQP-style data.
3. A backend bridges Python to the C++/FPGA runtime.

Current status:

- CVXPY adapter class exists.
- Mock backend exists for PC-side development.
- Ctypes backend and C++ ABI are scaffolded, not yet implemented.