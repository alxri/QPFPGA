# QPFPGA: FPGA-Accelerated QP Solver

A Quadratic Programming (QP) solver library combining Vitis HLS FPGA acceleration with a CVXPY-integrated Python interface for solving convex optimization problems on FPGA hardware.

## Repository Structure

This repository contains two main components:

### `admm/` – Hardware Accelerator Development
Vitis HLS and Vivado sources for building the ADMM-based QP solver FPGA accelerator.

**Key contents:**
- `HLS/` – HLS kernel sources: ADMM and PCG modules, SpMV engine
- `DSE/` – Design Space Exploration (PE count, reshape factor, complete exploration)
- `scripts/` – Tcl automation for HLS synthesis and Vivado implementation

This is where you build, optimize, and validate the FPGA design. See [admm/README.md](admm/README.md) for build instructions and [admm/HLS/README.md](admm/HLS/README.md) for kernel details.

### `QPFPGA/` – Solver Library (Python Package)
Python package and C++ backend for FPGA-accelerated QP solving via CVXPY.

**Key contents:**
- `qpfpga/` – Python package with CVXPY solver interface and backend selection
- `cpp/` – C++ shared-library backend (`libqpfpga.so`) for FPGA runtime
- `examples/` – Demo using CVXPY
- `benchmarks/` – Benchmark suite for CPU/FPGA performance comparison
- `bitstreams/` – Pre-built bitstreams for heuristic table based on DSE performance (Pareto Optimal) for runtime deployment

**For solver users:** Install this package to solve QP problems using CVXPY with FPGA acceleration. See [QPFPGA/README.md](QPFPGA/README.md) for installation and usage.

## Quick Start

### For End Users (Solving QPs)

1. Install the QPFPGA package in a Python environment:
   ```bash
   cd QPFPGA
   pip install -e .
   ```

2. Build the C++ backend:
   ```bash
   cd QPFPGA/cpp
   mkdir -p build && cd build
   cmake ..
   make
   ```

3. Run the demo:
   ```bash
   cd QPFPGA
   python examples/cvxpy_qpfpga_demo.py
   ```

   See [QPFPGA/README.md](QPFPGA/README.md) for full installation and configuration details.

### For Hardware Developers (Building Bitstreams)

1. Review the design overview in [admm/README.md](admm/README.md)

2. Build the accelerator:
   ```bash
   cd admm
   make hls_project
   make hls_synth
   make hls_cosim
   make bitstream
   ```

3. Explore design trade-offs using DSE scripts in [admm/DSE/](admm/DSE/):
   ```bash
   cd admm/DSE
   python run_dse.py
   ```

   See [admm/DSE/README.md](admm/DSE/README.md) for exploration details.

## Benchmarking

Compare FPGA performance against CPU solvers (OSQP, Clarabel, SCS, ECOS):

```bash
cd QPFPGA/benchmarks/osqp_benchmarks
python run_benchmark_qpfpga.py      # FPGA runs
python run_benchmark_osqp_only.py   # CPU baseline
```

## Key Files & Documentation

| Component | File | Purpose |
|-----------|------|---------|
| HLS Design | [admm/HLS/README.md](admm/HLS/README.md) | Kernel architecture and implementation details |
| Design Space Exploration | [admm/DSE/README.md](admm/DSE/README.md) | PE count, reshape, and problem size sweeps |
| Solver Library | [QPFPGA/README.md](QPFPGA/README.md) | Python API, installation, configuration |
| Build Flow | [admm/Makefile](admm/Makefile) | End-to-end HLS/Vivado build targets |

## System Requirements

- **For FPGA Users:** Linux with Xilinx PYNQ-capable FPGA board
- **For Developers:** Xilinx Vitis HLS 2021.2+ and Vivado 2021.2+
- **Python:** 3.9 or newer with numpy, scipy, cvxpy, osqp, clarabel, scs, qdldl, ecos
