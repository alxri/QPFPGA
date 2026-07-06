# HLS

This directory contains the Vitis HLS implementation of the ADMM solver.

The accelerator is built from three main kernels:

- [src/admm.cpp](src/admm.cpp) - top-level ADMM controller
- [src/pcg.cpp](src/pcg.cpp) - preconditioned conjugate-gradient solver used inside ADMM
- [src/spmv_csc.cpp](src/spmv_csc.cpp) - sparse matrix-vector multiply in CSC format, including the tiled path

## Key Headers

- [include/admm.h](include/admm.h) - top-level accelerator interface
- [include/pcg.h](include/pcg.h) - PCG interface
- [include/spmv_csc.h](include/spmv_csc.h) - sparse matrix types and tiled CSC helpers
- [include/config.h](include/config.h) - configuration constants used by HLS and DSE

`include/config.h` is the compile-time configuration header used by the HLS code and controls:

- `NUM_PES`
- `MAX_ROWS` and `MAX_COLS`
- `RESHAPE_FACTOR`
- solver constants used by the HLS code

## Data Representation

The sparse matrices are stored in compressed sparse column format.

- Regular CSC data is used for preconditioner updates.
- Tiled CSC data is used for the sparse matrix-vector products inside the ADMM and PCG loops.
- Packed 512-bit words are represented with `hls::vector<float, PACK_SIZE>` and `hls::vector<int, PACK_SIZE>` where `PACK_SIZE = 16`.

## Testbenches

- [test/tb_admm.cpp](test/tb_admm.cpp) builds a randomized 15x15 problem and exercises the full accelerator path.
- [test/tb_admm_3x3.cpp](test/tb_admm_3x3.cpp) is a deterministic sanity check for a tiny 3x3 problem.
- [test/tb_admm_ill.cpp](test/tb_admm_ill.cpp) stresses the adaptive-rho logic with a more difficult problem.

## HLS Flow

These are the relevant build targets from the root [Makefile](../Makefile):

```bash
make hls_project
make hls_synth
make hls_sim
make hls_cosim
make ip
```

The HLS project itself is generated in `admm_HW_HLS/` and the reports are copied into `reports/`.

## Implementation Notes

- The top function in [src/admm.cpp](src/admm.cpp) performs the ADMM loop, preconditioner update, residual checks, and optional adaptive-rho updates.
- [src/pcg.cpp](src/pcg.cpp) uses tiled sparse products and reshaped arrays to reduce latency in the inner linear solve.
- [src/spmv_csc.cpp](src/spmv_csc.cpp) distributes nonzeros across `NUM_PES` processing elements using streams.