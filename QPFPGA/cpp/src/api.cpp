#include "api.h"
#include "utils.h"
#include "fpga_utils.h"
#include <utility>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <pmt.h>

extern "C" QPFPGAStatus qpfpga_solve(
    const QPFPGAProblem* problem,
    const QPFPGAOptions* options,
    QPFPGAResult* result)
{
    if (!problem || !options || !result) return QPFPGA_STATUS_SOLVER_ERROR;

    double prep_start = get_time_ms();

    int n = problem->n;
    int m = problem->m;

    auto cpu_fallback = [&](const std::vector<float>& q_vec,
                            const std::vector<float>& P_diag_vec,
                            const std::vector<float>& E_vec) -> QPFPGAStatus {
        float* out_x = new float[n];
        float* out_y = new float[m];
        for (int i = 0; i < n; ++i) {
            float scale = (i < (int)E_vec.size()) ? E_vec[i] : 1.0f;
            out_x[i] = (-q_vec[i] / std::max(P_diag_vec[i], options->sigma)) * scale;        
        }
        for (int i = 0; i < m; ++i) out_y[i] = 0.0f;

        result->status = QPFPGA_STATUS_OPTIMAL;
        result->admm_iters = 0;
        result->pcg_iters = 0;
        result->primal_residual = 0.0f;
        result->dual_residual = 0.0f;
        result->objective_value = 0.0f;
        result->solve_time_ms = 0.0;
        result->setup_time_ms = 0.0;
        result->x = out_x;
        result->y = out_y;
        return result->status;
    };

    // Copy P (CSC)
    std::vector<int> P_cptr; std::vector<int> P_ridx; std::vector<float> P_vals;
    copy_csc_from_raw(problem->P.ncols, problem->P.nnz, problem->P.indptr, problem->P.indices, problem->P.data,
                      P_cptr, P_ridx, P_vals);

    // Copy A (CSC)
    std::vector<int> A_cptr; std::vector<int> A_ridx; std::vector<float> A_vals;
    copy_csc_from_raw(problem->A.ncols, problem->A.nnz, problem->A.indptr, problem->A.indices, problem->A.data,
                      A_cptr, A_ridx, A_vals);

    // Copy vectors
    std::vector<float> q(problem->q, problem->q + n);
    std::vector<float> l(problem->l, problem->l + m);
    std::vector<float> u(problem->u, problem->u + m);

    // Build P diagonal vector
    std::vector<float> P_diag;
    build_diag_from_csc(n, P_cptr, P_ridx, P_vals, P_diag);

    // Scaling (in-place modifies A_vals, P_diag, q, l, u)
    std::vector<float> D(m, 1.0f);
    std::vector<float> E(n, 1.0f);
    float c_scale = 1.0f; // Track cost scaling factor to unscale 'y' later

    try {
        apply_scaling(m, n, A_cptr, A_ridx, A_vals, P_diag, q, l, u, D, E, c_scale, SCALING_ITER_DEFAULT);    
    } catch (...) {
    }

    for (int c = 0; c < n; ++c) {
        for (int idx = P_cptr[c]; idx < P_cptr[c+1]; ++idx) {
            int r = P_ridx[idx];
            if (r == c) {
                P_vals[idx] = P_diag[c]; 
            } else {
                P_vals[idx] = P_vals[idx] * E[r] * E[c] * c_scale; 
            }
        }
    }

    for (float& val : A_vals) if (std::abs(val) > 0.0f && std::abs(val) < 1e-15f) val = 0.0f;
    for (float& val : P_vals) if (std::abs(val) > 0.0f && std::abs(val) < 1e-15f) val = 0.0f;
    for (float& val : q)      if (std::abs(val) > 0.0f && std::abs(val) < 1e-15f) val = 0.0f;
    for (float& val : P_diag) if (std::abs(val) > 0.0f && std::abs(val) < 1e-15f) val = 0.0f;

    // Build transpose of A
    std::vector<int> AT_cptr; std::vector<int> AT_ridx; std::vector<float> AT_vals;
    transpose_csc(m, n, A_cptr, A_ridx, A_vals, AT_cptr, AT_ridx, AT_vals);

    // Tile matrices
    int t_size = options->tile_size;
    TiledMatrix tm_A = build_tiled_csc(m, n, A_cptr, A_ridx, A_vals, t_size);
    TiledMatrix tm_AT = build_tiled_csc(n, m, AT_cptr, AT_ridx, AT_vals, t_size);
    TiledMatrix tm_P = build_tiled_csc(n, n, P_cptr, P_ridx, P_vals, t_size);

    // Prepare CMA buffers and copy data for A (packed words)
    CmaTracker cma;
    int32_words* A_reg_ridx = nullptr;
    float32_words* A_reg_vals = nullptr;
    int* A_reg_cptr = nullptr;
    int a_words_cnt = 0;
    allocate_and_copy_csc_to_cma(cma, A_cptr, A_ridx, A_vals, &A_reg_ridx, &A_reg_vals, &A_reg_cptr, &a_words_cnt);

    // Allocate tiled CMA buffers for A, AT, P
#define ALLOC_TILE_CMA_LOCAL(mat, tm) \
    int* hw_tile_##mat##_cnt = cma.alloc<int>(tm.counts.size()); cma_copy(hw_tile_##mat##_cnt, tm.counts.data(), tm.counts.size()); \
    int* hw_tile_##mat##_noff = cma.alloc<int>(tm.noff.size()); cma_copy(hw_tile_##mat##_noff, tm.noff.data(), tm.noff.size()); \
    int* hw_tile_##mat##_coff = cma.alloc<int>(tm.coff.size()); cma_copy(hw_tile_##mat##_coff, tm.coff.data(), tm.coff.size()); \
    int* hw_tile_##mat##_cptr = cma.alloc<int>(tm.cptr.size()); cma_copy(hw_tile_##mat##_cptr, tm.cptr.data(), tm.cptr.size()); \
    int32_words* hw_tile_##mat##_ridx = cma.alloc<int32_words>(tm.ridx.size()/PACK_SIZE); cma_copy(hw_tile_##mat##_ridx, (const int32_words*)tm.ridx.data(), tm.ridx.size()/PACK_SIZE); \
    float32_words* hw_tile_##mat##_vals = cma.alloc<float32_words>(tm.vals.size()/PACK_SIZE); cma_copy(hw_tile_##mat##_vals, (const float32_words*)tm.vals.data(), tm.vals.size()/PACK_SIZE);

    ALLOC_TILE_CMA_LOCAL(A, tm_A);
    ALLOC_TILE_CMA_LOCAL(AT, tm_AT);
    ALLOC_TILE_CMA_LOCAL(P, tm_P);

    std::vector<float> rho = build_rho_vector(m, l, u);

    int n_pad = ceil_div(n, PACK_SIZE) * PACK_SIZE;
    int m_pad = ceil_div(m, PACK_SIZE) * PACK_SIZE;

    P_diag.resize(n_pad, 0.0f);
    q.resize(n_pad, 0.0f);
    l.resize(m_pad, -ADMM_INFTY);
    u.resize(m_pad, ADMM_INFTY);
    rho.resize(m_pad, 1.0f);

    // Allocate CMA buffers using the padded sizes
    float* hw_Pdiag = cma.alloc<float>(n_pad); cma_copy(hw_Pdiag, P_diag.data(), n_pad);
    float* hw_l = cma.alloc<float>(m_pad);     cma_copy(hw_l, l.data(), m_pad);
    float* hw_u = cma.alloc<float>(m_pad);     cma_copy(hw_u, u.data(), m_pad);
    float* hw_q = cma.alloc<float>(n_pad);     cma_copy(hw_q, q.data(), n_pad);
    float* hw_rho = cma.alloc<float>(m_pad);   cma_copy(hw_rho, rho.data(), m_pad);
    float* hw_x = cma.alloc<float>(n_pad);     // HW writes padded size, we read back 'n'
    float* hw_y = cma.alloc<float>(m_pad);     // HW writes padded size, we read back 'm'

    // Memory map control registers
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        QPFPGAStatus status = cpu_fallback(q, P_diag, E);
        cma.free_all();
        return status;
    }
    void *ip_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, ADMM_IP_CONTROL_BASE & ~MAP_MASK);
    void *ctrl = (void*)((uint8_t*)ip_base + (ADMM_IP_CONTROL_BASE & MAP_MASK));
    void *ctrl_r = (void*)((uint8_t*)ip_base + (ADMM_IP_CONTROL_R_BASE & MAP_MASK));

    // Program CMA addresses to control_r registers
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_ROW_IDX_DATA, cma_get_phy_addr(A_reg_ridx));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_COL_PTR_DATA, cma_get_phy_addr(A_reg_cptr));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_VALUES_DATA, cma_get_phy_addr(A_reg_vals));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_TILE_NNZ_COUNTS_DATA, cma_get_phy_addr(hw_tile_A_cnt)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_TILE_NNZ_OFFSETS_DATA, cma_get_phy_addr(hw_tile_A_noff));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_TILE_COL_OFFSETS_DATA, cma_get_phy_addr(hw_tile_A_coff)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_ROW_IDX_TILED_DATA, cma_get_phy_addr(hw_tile_A_ridx));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_COL_PTR_TILED_DATA, cma_get_phy_addr(hw_tile_A_cptr)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_A_VALUES_TILED_DATA, cma_get_phy_addr(hw_tile_A_vals));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_AT_TILE_NNZ_COUNTS_DATA, cma_get_phy_addr(hw_tile_AT_cnt)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_AT_TILE_NNZ_OFFSETS_DATA, cma_get_phy_addr(hw_tile_AT_noff));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_AT_TILE_COL_OFFSETS_DATA, cma_get_phy_addr(hw_tile_AT_coff)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_AT_ROW_IDX_TILED_DATA, cma_get_phy_addr(hw_tile_AT_ridx));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_AT_COL_PTR_TILED_DATA, cma_get_phy_addr(hw_tile_AT_cptr)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_AT_VALUES_TILED_DATA, cma_get_phy_addr(hw_tile_AT_vals));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_P_TILE_NNZ_COUNTS_DATA, cma_get_phy_addr(hw_tile_P_cnt)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_P_TILE_NNZ_OFFSETS_DATA, cma_get_phy_addr(hw_tile_P_noff));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_P_TILE_COL_OFFSETS_DATA, cma_get_phy_addr(hw_tile_P_coff)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_P_ROW_IDX_TILED_DATA, cma_get_phy_addr(hw_tile_P_ridx));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_P_COL_PTR_TILED_DATA, cma_get_phy_addr(hw_tile_P_cptr)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_P_VALUES_TILED_DATA, cma_get_phy_addr(hw_tile_P_vals));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_P_DIAG_DATA, cma_get_phy_addr(hw_Pdiag)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_L_IN_DATA, cma_get_phy_addr(hw_l));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_U_IN_DATA, cma_get_phy_addr(hw_u)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_Q_IN_DATA, cma_get_phy_addr(hw_q));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_RHO_IN_DATA, cma_get_phy_addr(hw_rho)); write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_X_OUT_DATA, cma_get_phy_addr(hw_x));
    write_64bit_address(ctrl_r, XADMM_CONTROL_R_ADDR_Y_OUT_DATA, cma_get_phy_addr(hw_y));

    // Write control scalars and start
    write_reg(ctrl, XADMM_CONTROL_ADDR_NUM_ROWS_DATA, m); write_reg(ctrl, XADMM_CONTROL_ADDR_NUM_COLS_DATA, n); write_reg(ctrl, XADMM_CONTROL_ADDR_A_NNZ_DATA, (int)A_vals.size());
    write_reg(ctrl, XADMM_CONTROL_ADDR_A_NUM_ROW_TILES_DATA, tm_A.rtiles); write_reg(ctrl, XADMM_CONTROL_ADDR_A_NUM_COL_TILES_DATA, tm_A.ctiles);
    write_reg(ctrl, XADMM_CONTROL_ADDR_AT_NUM_ROW_TILES_DATA, tm_AT.rtiles); write_reg(ctrl, XADMM_CONTROL_ADDR_AT_NUM_COL_TILES_DATA, tm_AT.ctiles);
    write_reg(ctrl, XADMM_CONTROL_ADDR_P_NUM_ROW_TILES_DATA, tm_P.rtiles); write_reg(ctrl, XADMM_CONTROL_ADDR_P_NUM_COL_TILES_DATA, tm_P.ctiles);
    write_reg(ctrl, XADMM_CONTROL_ADDR_SIGMA_DATA, float_to_uint(options->sigma)); write_reg(ctrl, XADMM_CONTROL_ADDR_ALPHA_DATA, float_to_uint(options->alpha));
    write_reg(ctrl, XADMM_CONTROL_ADDR_ADMM_MAX_ITERATIONS_DATA, options->admm_max_iter); write_reg(ctrl, XADMM_CONTROL_ADDR_PCG_MAX_ITERATIONS_DATA, options->pcg_max_iter);
    write_reg(ctrl, XADMM_CONTROL_ADDR_ADAPTIVE_RHO_DATA, options->adaptive_rho);
    write_reg(ctrl, XADMM_CONTROL_ADDR_EPS_ABS_DATA, float_to_uint(options->eps_abs)); write_reg(ctrl, XADMM_CONTROL_ADDR_EPS_REL_DATA, float_to_uint(options->eps_rel));
    write_reg(ctrl, XADMM_CONTROL_ADDR_PCG_TOL_FRACTION_DATA, float_to_uint(options->pcg_tol_fraction));

    cma.flush_all();

    double core_energy = 0.0;
    double aux_energy = 0.0;
    double fpga_energy = 0.0;
    double board_energy = 0.0;

    std::unique_ptr<pmt::PMT> sensor;
    pmt::State prev_state;

    if (options->measure_energy) {
        sensor = pmt::xilinx::Xilinx::Create();
        sensor->Read();
        prev_state = sensor->Read();
    }

    result->setup_time_ms = get_time_ms() - prep_start;

    double hw_start = get_time_ms();
    
    // Start accelerator
    write_reg(ctrl, XADMM_CONTROL_ADDR_AP_CTRL, 0x01);

    if (options->measure_energy) {
        while ((read_reg(ctrl, XADMM_CONTROL_ADDR_AP_CTRL) & 0x02) == 0) {
            usleep(10000);

            pmt::State curr_state = sensor->Read();
            std::chrono::duration<double> diff = curr_state.timestamp_ - prev_state.timestamp_;
            double dt = diff.count();

            core_energy  += 0.5 * (prev_state.watt_[1] + curr_state.watt_[1]) * dt;
            aux_energy   += 0.5 * (prev_state.watt_[2] + curr_state.watt_[2]) * dt;
            fpga_energy  += 0.5 * (prev_state.watt_[3] + curr_state.watt_[3]) * dt;
            board_energy += 0.5 * (prev_state.watt_[4] + curr_state.watt_[4]) * dt;

            prev_state = curr_state;
        }
        
        pmt::State curr_state = sensor->Read();
        std::chrono::duration<double> diff = curr_state.timestamp_ - prev_state.timestamp_;
        double dt = diff.count();
        if (dt > 0) {
            core_energy  += 0.5 * (prev_state.watt_[1] + curr_state.watt_[1]) * dt;
            aux_energy   += 0.5 * (prev_state.watt_[2] + curr_state.watt_[2]) * dt;
            fpga_energy  += 0.5 * (prev_state.watt_[3] + curr_state.watt_[3]) * dt;
            board_energy += 0.5 * (prev_state.watt_[4] + curr_state.watt_[4]) * dt;
        }

    } else {
        while ((read_reg(ctrl, XADMM_CONTROL_ADDR_AP_CTRL) & 0x02) == 0) {
            usleep(100);
        }
    }

    double hw_end = get_time_ms();

    // Assign everything back to Python
    result->core_energy_j = core_energy;
    result->aux_energy_j = aux_energy;
    result->fpga_energy_j = fpga_energy;
    result->board_energy_j = board_energy;

    printf("Measured Energy (J): Core=%.4f | Aux=%.4f | FPGA=%.4f | Board=%.4f\n",
           core_energy, aux_energy, fpga_energy, board_energy);

    cma.invalidate_all();

    int admm_iters = read_reg(ctrl, XADMM_CONTROL_ADDR_ADMM_NUM_ITERATIONS_OUT_DATA);
    int pcg_iters = read_reg(ctrl, XADMM_CONTROL_ADDR_PCG_NUM_ITERATIONS_OUT_DATA);
    float p_res = uint_to_float(read_reg(ctrl, XADMM_CONTROL_ADDR_R_PRIM_OUT_DATA));
    float d_res = uint_to_float(read_reg(ctrl, XADMM_CONTROL_ADDR_R_DUAL_OUT_DATA));
    int status = read_reg(ctrl, XADMM_CONTROL_ADDR_STATUS_OUT_DATA);

    // Copy out results into freshly allocated arrays returned to caller
    float* out_x = new float[n];
    float* out_y = new float[m];
    
    for (int i = 0; i < n; ++i) out_x[i] = hw_x[i] * E[i];
    for (int i = 0; i < m; ++i) out_y[i] = hw_y[i] * D[i] / c_scale;

    result->status = (status == 1) ? QPFPGA_STATUS_OPTIMAL : QPFPGA_STATUS_USER_LIMIT;
    result->admm_iters = admm_iters;
    result->pcg_iters = pcg_iters;
    result->primal_residual = p_res;
    result->dual_residual = d_res;
    result->objective_value = 0.0f;
    result->solve_time_ms = hw_end - hw_start;
    result->x = out_x;
    result->y = out_y;

    // Cleanup
    munmap(ip_base, MAP_SIZE); close(mem_fd);
    cma.free_all();

    return result->status;
}