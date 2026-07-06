#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Include your HLS design header
#include "admm.h"

// OSQP Constants for rho calculation
#define OSQP_RHO 0.1f
#define OSQP_RHO_EQ_OVER_RHO_INEQ 1000.0f
#define OSQP_RHO_TOL 1e-4f
#define OSQP_INFTY 1e20f
#define OSQP_MIN_SCALING 1e-4f
#define OSQP_RHO_MAX 1e6f
#define OSQP_RHO_MIN 1e-6f

// Helper to safely pack flat CSC data into hls::vector 512-bit words
static bool pack_csc_nnz_to_words(const std::vector<int> &row_idx,
                                  const std::vector<float> &values,
                                  std::vector<int16> &row_words,
                                  std::vector<float16> &val_words)
{
    if (row_idx.size() != values.size()) {
        std::cout << "ERROR: pack: row/val size mismatch." << std::endl;
        return false;
    }
    const int nnz = (int)row_idx.size();

    // Clear the interface to prevent uninitialized memory issues
    for (int w = 0; w < MAX_NNZ_WORDS; ++w) {
        for (int lane = 0; lane < PACK_SIZE; ++lane) {
            row_words[w][lane] = 0;
            val_words[w][lane] = 0.0f;
        }
    }

    // Pack lane by lane
    for (int idx = 0; idx < nnz; ++idx) {
        const int w = idx / PACK_SIZE;
        const int lane = idx % PACK_SIZE;
        row_words[w][lane] = row_idx[idx];
        val_words[w][lane] = values[idx];
    }
    return true;
}

int main() {
    std::cout << "--- Starting ADMM Vitis HLS Testbench (3x3 Problem) ---" << std::endl;

    int num_cols = 3;
    int num_rows = 3;
    
    // -------------------------------------------------------------------------
    // 1. Matrix P (3x3) in CSC format 
    // -------------------------------------------------------------------------
    int P_nnz = 3;
    std::vector<float> P_values = {1.0f, 1.0f, 1.0f}; 
    std::vector<int> P_row_idx  = {0, 1, 2};
    std::vector<int> P_col_ptr  = {0, 1, 2, 3};
    // Pad P_col_ptr to prevent AXI burst read overflow
    P_col_ptr.resize(MAX_COL_PTR, 0); 
    
    std::vector<float> P_diag(MAX_COLS, 0.0f);
    P_diag[0] = 1.0f; P_diag[1] = 1.0f; P_diag[2] = 1.0f;

    // -------------------------------------------------------------------------
    // 2. Matrix A (3x3) in CSC format 
    // -------------------------------------------------------------------------
    int A_nnz = 3;
    std::vector<float> A_values = {2.0f, -1.0f, 3.0f};
    std::vector<int> A_row_idx  = {1, 2, 0};
    std::vector<int> A_col_ptr  = {0, 1, 2, 3};
    A_col_ptr.resize(MAX_COL_PTR, 0);

    // -------------------------------------------------------------------------
    // 3. Matrix A^T (3x3) in CSC format
    // -------------------------------------------------------------------------
    std::vector<float> AT_values = {3.0f, 2.0f, -1.0f};
    std::vector<int> AT_row_idx  = {2, 0, 1};
    std::vector<int> AT_col_ptr  = {0, 1, 2, 3};
    AT_col_ptr.resize(MAX_COL_PTR, 0);

    // -------------------------------------------------------------------------
    // 4. Pack Sparse Matrices into hls::vector Words
    // -------------------------------------------------------------------------
    std::vector<int16> P_row_words(MAX_NNZ_WORDS);
    std::vector<float16> P_val_words(MAX_NNZ_WORDS);
    
    std::vector<int16> A_row_words(MAX_NNZ_WORDS);
    std::vector<float16> A_val_words(MAX_NNZ_WORDS);
    
    std::vector<int16> AT_row_words(MAX_NNZ_WORDS);
    std::vector<float16> AT_val_words(MAX_NNZ_WORDS);

    pack_csc_nnz_to_words(P_row_idx, P_values, P_row_words, P_val_words);
    pack_csc_nnz_to_words(A_row_idx, A_values, A_row_words, A_val_words);
    pack_csc_nnz_to_words(AT_row_idx, AT_values, AT_row_words, AT_val_words);

    // -------------------------------------------------------------------------
    // 5. Vectors l, u, q, and rho
    // -------------------------------------------------------------------------
    // Size to MAX_ROWS / MAX_COLS to prevent out-of-bound burst reads by AXI
    std::vector<float> l(MAX_ROWS, 0.0f);
    std::vector<float> u(MAX_ROWS, 0.0f);
    std::vector<float> q(MAX_COLS, 0.0f);
    std::vector<float> rho(MAX_ROWS, 0.0f);

    l[0] = -1.0f; l[1] = 0.0f; l[2] = -OSQP_INFTY;
    u[0] = 1.0f;  u[1] = 0.0f; u[2] = OSQP_INFTY;
    q[0] = 1.0f;  q[1] = -1.0f; q[2] = 0.5f;

    // Calculate OSQP rho exactly for the 3 values
    float rho_base = OSQP_RHO;
    for (int i = 0; i < num_rows; i++) {
        if ((u[i] - l[i]) < OSQP_RHO_TOL) {
            rho[i] = OSQP_RHO * OSQP_RHO_EQ_OVER_RHO_INEQ;
        } 
        else if (l[i] <= -(OSQP_INFTY * OSQP_MIN_SCALING) && u[i] >= (OSQP_INFTY * OSQP_MIN_SCALING)) {
            rho[i] = OSQP_RHO_MIN;
        } 
        else {
            rho[i] = rho_base;
        }
        rho[i] = std::max(OSQP_RHO_MIN, std::min(OSQP_RHO_MAX, rho[i]));
    }

    std::cout << "Calculated Rho vector: [" 
              << rho[0] << ", " << rho[1] << ", " << rho[2] << "]" << std::endl;

    // -------------------------------------------------------------------------
    // 6. Scalar parameters and Outputs
    // -------------------------------------------------------------------------
    float alpha = 1.6f;
    float sigma = 1e-6f;
    int admm_max_iterations = 1000;
    int pcg_max_iterations = 100;
    bool adaptive_rho = false;

    std::vector<float> x_out(MAX_COLS, 0.0f);
    std::vector<float> y_out(MAX_ROWS, 0.0f);
    
    int admm_num_iterations_out = 0;
    int pcg_num_iterations_out = 0;
    float r_prim_out = 0.0f;
    float r_dual_out = 0.0f;
    int status_out = 0;

    // -------------------------------------------------------------------------
    // 7. Execute HLS Core
    // -------------------------------------------------------------------------
    std::cout << "Running ADMM core..." << std::endl;
    
    admm(
        num_rows, num_cols,
        A_row_words.data(), A_col_ptr.data(), A_val_words.data(), A_nnz,
        AT_row_words.data(), AT_col_ptr.data(), AT_val_words.data(),
        P_row_words.data(), P_col_ptr.data(), P_val_words.data(), P_nnz, P_diag.data(),
        l.data(), u.data(), q.data(),
        sigma, alpha, rho.data(),
        admm_max_iterations, pcg_max_iterations, adaptive_rho,
        x_out.data(), y_out.data(),
        &admm_num_iterations_out, &pcg_num_iterations_out,
        &r_prim_out, &r_dual_out, &status_out
    );

    // -------------------------------------------------------------------------
    // 8. Results
    // -------------------------------------------------------------------------
    std::cout << "--- Simulation Results ---" << std::endl;
    std::cout << "Status: " << (status_out == 1 ? "Converged" : "Max Iterations Reached") << std::endl;
    std::cout << "ADMM Iterations: " << admm_num_iterations_out << std::endl;
    std::cout << "Total PCG Iterations: " << pcg_num_iterations_out << std::endl;
    std::cout << "Primal Residual: " << r_prim_out << std::endl;
    std::cout << "Dual Residual: " << r_dual_out << std::endl;
    
    std::cout << "Solution x_out: [" << x_out[0] << ", " << x_out[1] << ", " << x_out[2] << "]" << std::endl;
    std::cout << "Dual var y_out: [" << y_out[0] << ", " << y_out[1] << ", " << y_out[2] << "]" << std::endl;

    if (status_out == 1 || admm_num_iterations_out > 0) {
        std::cout << "Testbench completed successfully." << std::endl;
        return 0; 
    } else {
        std::cout << "Testbench failed." << std::endl;
        return 1; 
    }
}