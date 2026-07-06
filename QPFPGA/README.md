# QPFPGA

QPFPGA is the CVXPY-facing Python package for the FPGA QP solver stack.
It registers a custom `QPFPGA` solver with CVXPY and bridges the solver call to a C++ backend that can drive the FPGA runtime.

## What Is Here

- `qpfpga/` - Python package with the CVXPY solver adapter and backend selection logic
- `cpp/` - C++ shared-library backend built as `libqpfpga.so`
- `examples/` - a minimal end-to-end demo using CVXPY
- `benchmarks/osqp_benchmarks/` - benchmark suite used for CPU and FPGA runs
- `bitstreams/` - FPGA bitstreams used by the backend
- `tests/` - package tests and local validation helpers

## Requirements

The package targets Linux systems with a Xilinx PYNQ-capable FPGA setup.

You need:

- Python 3.9 or newer
- `numpy`, `scipy`, `cvxpy`, `osqp`, `clarabel`, `scs`, `qdldl`, `ecos`
- `pynq==3.0.1` for FPGA board execution
- build tools for the C++ backend: `cmake`, `build-essential`, `python3-dev`
- BLAS/LAPACK development packages

On Ubuntu/Debian systems, the basic system packages are:

```bash
sudo apt update
sudo apt install -y python3.10-venv python3.10-dev build-essential cmake pkg-config \
	gfortran libblas-dev liblapack-dev
```

## Install

Create and activate a virtual environment:

```bash
python3 -m venv ~/qpfpga-venv
source ~/qpfpga-venv/bin/activate
pip install --upgrade pip setuptools wheel
```

Install the package in editable mode from the repository root:

```bash
cd ~/QPFPGA
pip install -e .
```

## Build The C++ Backend

The Python package expects a shared library at `cpp/build/libqpfpga.so` unless you override the path with `QPFPGA_LIBRARY`.

Rebuild the backend whenever the C++ sources change or the shared library is missing:

```bash
cd ~/QPFPGA/cpp
rm -rf build
mkdir build
cd build
cmake ..
make
```

## Run The Demo

The demo in `examples/cvxpy_qpfpga_demo.py` solves a simple box-constrained QP through CVXPY.

By default the backend loads the FPGA shared library from `QPFPGA_LIBRARY`.
If that variable is not set, the example script tries to use `cpp/build/libqpfpga.so` relative to the repository.

Run against the FPGA backend:

```bash
cd ~/QPFPGA
sudo QPFPGA_LIBRARY="/home/xilinx/QPFPGA/cpp/build/libqpfpga.so" LD_LIBRARY_PATH="/usr/lib" /home/xilinx/qpfpga-venv/bin/python examples/cvxpy_qpfpga_demo.py
```

Run with the CPU fallback backend:

```bash
cd ~/QPFPGA
sudo QPFPGA_FORCE_CPU="yes" \
QPFPGA_LIBRARY="/home/xilinx/QPFPGA/cpp/build/libqpfpga.so" \
LD_LIBRARY_PATH="/usr/lib" \
python examples/cvxpy_qpfpga_demo.py
```

## Benchmarks

The benchmark suite lives in `benchmarks/osqp_benchmarks/`.

Useful entry points:

- `run_benchmark_osqp_only.py` - CPU baseline runs
- `run_benchmark_qpfpga.py` - FPGA runs

Example:

```bash
cd ~/QPFPGA/benchmarks/osqp_benchmarks
python run_benchmark_osqp_only.py
python run_benchmark_qpfpga.py
```

## Runtime Behavior

The package registers a custom solver with CVXPY when `qpfpga` is imported:

```python
import qpfpga
problem.solve(solver="QPFPGA")
```

The backend selection works as follows:

- If `QPFPGA_FORCE_CPU` is set to a truthy value, the package uses the CPU mock backend.
- Otherwise, `QPFPGA_LIBRARY` must point to the shared library built from `cpp/`.
- The C++ backend selects a bitstream based on problem sparsity and uses the files in `bitstreams/`.

Supported solver options include `sigma`, `alpha`, `eps_abs`, `eps_rel`, `pcg_tol_fraction`, `admm_max_iter`, `pcg_max_iter`, `adaptive_rho`, and `measure_energy`.

## Python API

The package exports the following symbols:

- `QPFPGA` - CVXPY solver class
- `QPData` - OSQP-style problem container
- `QPSolverOptions` - solver configuration dataclass
- `QPSolverResult` - solver result dataclass

Typical use:

```python
from qpfpga import QPFPGA

problem.solve(solver="QPFPGA", verbose=True, measure_energy=True)
```

## Notes

- The package is designed for the FPGA workflow first, but it can fall back to a CPU backend for local development.
- The backend enforces a maximum problem size of 32768 variables and 32768 constraints.
- If you change the backend library path or bitstream location, update the corresponding environment variables before launching Python.