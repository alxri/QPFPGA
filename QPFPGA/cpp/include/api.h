#include "fpga_utils.h"
#pragma once

#include <cstdint>
#include "utils.h"


extern "C" {

enum QPFPGAStatus : int32_t {
    QPFPGA_STATUS_OPTIMAL = 1,
    QPFPGA_STATUS_OPTIMAL_INACCURATE = 2,
    QPFPGA_STATUS_INFEASIBLE = 3,
    QPFPGA_STATUS_INFEASIBLE_INACCURATE = 4,
    QPFPGA_STATUS_UNBOUNDED = 5,
    QPFPGA_STATUS_UNBOUNDED_INACCURATE = 6,
    QPFPGA_STATUS_USER_LIMIT = 7,
    QPFPGA_STATUS_SOLVER_ERROR = 10,
    QPFPGA_STATUS_NOT_IMPLEMENTED = -1,
};

struct QPFPGACscMatrix {
    int32_t nrows;
    int32_t ncols;
    int32_t nnz;
    const int32_t* indptr;
    const int32_t* indices;
    const float* data;
};

struct QPFPGAOptions {
    float sigma;
    float alpha;
    float eps_abs;
    float eps_rel;
    float pcg_tol_fraction;
    int32_t admm_max_iter;
    int32_t pcg_max_iter;
    int32_t adaptive_rho;
    int tile_size;
    int measure_energy;
};

struct QPFPGAProblem {
    QPFPGACscMatrix P;
    QPFPGACscMatrix A;
    const float* q;
    const float* l;
    const float* u;
    int32_t n;
    int32_t m;
};

struct QPFPGAResult {
    QPFPGAStatus status;
    int32_t admm_iters;
    int32_t pcg_iters;
    float primal_residual;
    float dual_residual;
    float objective_value;
    double solve_time_ms;
    double setup_time_ms;
    double core_energy_j;
    double aux_energy_j;
    double fpga_energy_j;
    double board_energy_j;
    const float* x;
    const float* y;
};

QPFPGAStatus qpfpga_solve(
    const QPFPGAProblem* problem,
    const QPFPGAOptions* options,
    QPFPGAResult* result);

}