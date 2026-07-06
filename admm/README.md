
# ADMM FPGA Accelerator

This repository contains an FPGA implementation of an ADMM-based quadratic-program solver built with Vitis HLS and Vivado.

The design is organized around three main pieces:

- HLS kernel sources in [HLS/](HLS/)
- Design-space exploration scripts and results in [DSE/](DSE/)
- Tcl build scripts in [scripts/](scripts/)

## Repository Layout

- [HLS/](HLS/) - HLS top function, sparse linear algebra kernels, headers, and testbenches
- [DSE/](DSE/) - synthesis/implementation sweeps, generated bitstreams, and DSE result tables
- [scripts/](scripts/) - Vitis HLS and Vivado automation scripts used by the Makefile
- [reports/](reports/) - copied HLS synthesis and co-simulation reports
- [admm_HW_HLS/](admm_HW_HLS/) - generated HLS project directory

## Build Flow

The root [Makefile](Makefile) drives the full flow.

```bash
make hls_project
make hls_synth
make hls_sim
make hls_cosim
make ip
make vivado_project
make bitstream
```

## What Each Target Does

- `make hls_project` creates the Vitis HLS project under `admm_HW_HLS/`
- `make hls_synth` runs C synthesis and copies the synthesis report into `reports/`
- `make hls_sim` runs Vitis software simulation
- `make hls_cosim` runs RTL co-simulation and copies the co-sim report into `reports/`
- `make ip` exports the synthesized design as an IP core
- `make vivado_project` creates the Vivado project and block design
- `make bitstream` runs Vivado implementation and generates `admm.bit` and `admm.hwh`

## Generated Artifacts

After a successful full build, the most important outputs are:

- `admm.bit`
- `admm.hwh`
- `reports/admm_csynth.rpt`
- `reports/admm_cosim.rpt`

## Notes

- The HLS design uses [HLS/include/config.h](HLS/include/config.h) for compile-time parameters.
- The DSE scripts are meant to rewrite that header to sweep PE count, reshape factor, and problem size.
- Use `make clean` to remove Vivado/HLS build artifacts and the generated HLS project.

See [HLS/README.md](HLS/README.md) for kernel details and [DSE/README.md](DSE/README.md) for the exploration scripts.
