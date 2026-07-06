
# DSE

This directory contains the scripts and outputs used to explore the ADMM accelerator design space.

## Scripts

- [run_dse.py](run_dse.py) runs a C-synthesis sweep. It rewrites the HLS compile-time configuration header, invokes `make hls_synth`, parses the synthesis report, and writes summary results.
- [generate_bitstreams.py](generate_bitstreams.py) builds full bitstreams for a set of architecture configurations, then archives the generated artifacts per configuration.
- [generate_1024_bin_densities.py](generate_1024_bin_densities.py) generates density-oriented benchmark data for the 1024x1024 case.
- [generate_32768_bin_densities.py](generate_32768_bin_densities.py) generates density-oriented benchmark data for the 32768x32768 case.

## Synthesis Sweep

The default C-synthesis sweep in [run_dse.py](run_dse.py) currently runs a single configuration:

- `NUM_PES = 6`
- `Size = 32768`
- `RESHAPE_FACTOR = 8`

It writes:

- `dse_results_csynth_only_added.csv`
- `dse_results_csynth_only.md`

The repository also includes precomputed results in:

- `dse_results_csynth_only.csv`
- `dse_results_csynth_only.md`

## Bitstream Generation

The bitstream sweep in [generate_bitstreams.py](generate_bitstreams.py) iterates through three experiment families:

- Experiment 1: PE sweep at 1024x1024 with reshape factor 4
- Experiment 2: reshape-factor sweep at 1024x1024 with 20 PEs
- Experiment 3: multi-size sweep across 1024, 4096, 8192, 16384, and 32768 with multiple PE and reshape combinations

For each architecture, the script:

1. rewrites the HLS compile-time configuration header
2. runs `make clean`
3. runs `make hls_project`
4. runs `make ip`
5. runs `make vivado_project`
6. runs `make bitstream`
7. archives `admm.bit`, `admm.hwh`, and `vivado_build/`

The archived outputs are stored under `generated_bitstreams/<size>x<size>_reshape<rf>_pes<pes>/`.

## Data Folders

The subdirectories `1024x1024_data_bin_dse/` and `32768x32768_data_bin_dse/` contain the density-based benchmark inputs used for the exploration flow.

## Running The Scripts

Run these from the repository root so the relative paths resolve correctly:

```bash
python DSE/run_dse.py
python DSE/generate_bitstreams.py
```

The scripts expect `vitis_hls` and `vivado` to be available on `PATH` after sourcing the Xilinx settings environment.