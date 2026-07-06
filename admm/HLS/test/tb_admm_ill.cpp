#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>

// Include your HLS design header
#include "admm.h"

// OSQP Constants for problem generation
#define OSQP_RHO 0.1f
#define OSQP_RHO_EQ_OVER_RHO_INEQ 1000.0f
#define OSQP_INFTY 1e20f
#define OSQP_RHO_MAX 1e6f
#define OSQP_RHO_MIN 1e-6f

// Helper to transpose A into A^T in CSC format
static void transpose_csc(int num_rows, int num_cols,
                          const std::vector<int> &col_ptr, const std::vector<int> &row_idx, const std::vector<float> &values,
                          std::vector<int> &col_ptr_t, std::vector<int> &row_idx_t, std::vector<float> &values_t)
{
    col_ptr_t.assign(num_rows + 1, 0);
    std::vector<int> nnz_per_at_col(num_rows, 0);
    for (int idx = 0; idx < (int)row_idx.size(); ++idx) nnz_per_at_col[row_idx[idx]]++;
    for (int c = 0; c < num_rows; ++c) col_ptr_t[c + 1] = col_ptr_t[c] + nnz_per_at_col[c];
    
    const int nnz = (int)row_idx.size();
    row_idx_t.assign(nnz, 0);
    values_t.assign(nnz, 0.0f);
    std::vector<int> next(col_ptr_t.begin(), col_ptr_t.begin() + num_rows);

    for (int c = 0; c < num_cols; ++c) {
        for (int idx = col_ptr[c]; idx < col_ptr[c + 1]; ++idx) {
            const int r = row_idx[idx];
            const int dst = next[r]++;
            row_idx_t[dst] = c;
            values_t[dst] = values[idx];
        }
    }
}

// Helper to safely pack flat CSC data into hls::vector (int16/float16)
static void pack_csc_nnz_to_words(const std::vector<int> &row_idx, const std::vector<float> &values,
                                  std::vector<int16> &row_words, std::vector<float16> &val_words)
{
    const int nnz = (int)row_idx.size();
    for (int w = 0; w < MAX_NNZ_WORDS; ++w) {
        for (int lane = 0; lane < 16; ++lane) {
            row_words[w][lane] = 0;
            val_words[w][lane] = 0.0f;
        }
    }
    for (int idx = 0; idx < nnz; ++idx) {
        row_words[idx / 16][idx % 16] = row_idx[idx];
        val_words[idx / 16][idx % 16] = values[idx];
    }
}

int main() {
    // Increase size slightly to drive up iterations
    const int num_rows = 15;
    const int num_cols = 15;

    std::cout << "--- ADMM FPGA Testbench: Stress-Test for Adaptive Rho ---" << std::endl;

    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> x_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> p_dist(0.1f, 50.0f); // Ill-conditioned P
    std::uniform_real_distribution<float> a_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> q_noise(-20.0f, 20.0f); // Forces ADMM to struggle
    std::uniform_int_distribution<int> row_dist(0, num_rows - 1);
    std::uniform_int_distribution<int> type_dist(0, 1); // 0: Equality, 1: Inequality

    // 1. Generate P (Diagonal) and q
    std::vector<float> q(MAX_COLS, 0.0f);
    std::vector<float> P_diag(MAX_COLS, 0.0f);
    std::vector<float> P_values(num_cols);
    std::vector<int> P_row_idx(num_cols);
    std::vector<int> P_col_ptr(num_cols + 1);

    for (int i = 0; i < num_cols; ++i) {
        float val = p_dist(rng);
        P_diag[i] = val;
        P_values[i] = val;
        P_row_idx[i] = i;
        P_col_ptr[i] = i;
        q[i] = q_noise(rng); // Randomized q forces the solver to iterate more
    }
    P_col_ptr[num_cols] = num_cols;

    // 2. Generate Sparse A (Approx 2 NNZ per column)
    std::vector<int> A_col_ptr(num_cols + 1, 0);
    std::vector<int> A_row_idx;
    std::vector<float> A_values;
    for (int c = 0; c < num_cols; ++c) {
        A_col_ptr[c] = A_row_idx.size();
        for (int r_count = 0; r_count < 2; ++r_count) {
            A_row_idx.push_back(row_dist(rng));
            A_values.push_back(a_dist(rng));
        }
    }
    A_col_ptr[num_cols] = A_row_idx.size();
    int A_nnz = A_row_idx.size();

    // 3. Transpose A for AT
    std::vector<int> AT_col_ptr;
    std::vector<int> AT_row_idx;
    std::vector<float> AT_values;
    transpose_csc(num_rows, num_cols, A_col_ptr, A_row_idx, A_values, AT_col_ptr, AT_row_idx, AT_values);

    // 4. Pack into HLS Words
    std::vector<int16> P_row_words(MAX_NNZ_WORDS), A_row_words(MAX_NNZ_WORDS), AT_row_words(MAX_NNZ_WORDS);
    std::vector<float16> P_val_words(MAX_NNZ_WORDS), A_val_words(MAX_NNZ_WORDS), AT_val_words(MAX_NNZ_WORDS);
    pack_csc_nnz_to_words(P_row_idx, P_values, P_row_words, P_val_words);
    pack_csc_nnz_to_words(A_row_idx, A_values, A_row_words, A_val_words);
    pack_csc_nnz_to_words(AT_row_idx, AT_values, AT_row_words, AT_val_words);

    // 5. Constraints and Rho Initialization
    std::vector<float> l(MAX_ROWS), u(MAX_ROWS), rho(MAX_ROWS);
    for (int i = 0; i < num_rows; i++) {
        if (type_dist(rng) == 0) { // Equality
            l[i] = -1.0f; u[i] = -1.0f;
            rho[i] = OSQP_RHO * OSQP_RHO_EQ_OVER_RHO_INEQ;
        } else { // Inequality
            l[i] = -5.0f; u[i] = 10.0f;
            rho[i] = OSQP_RHO;
        }
    }

    // 6. ADMM Call
    std::vector<float> x_out(MAX_COLS, 0.0f), y_out(MAX_ROWS, 0.0f);
    int admm_iters = 0, pcg_iters = 0, status = 0;
    float r_prim = 0.0f, r_dual = 0.0f;

    std::cout << "Solving..." << std::endl;
    admm(
        num_rows, num_cols,
        A_row_words.data(), A_col_ptr.data(), A_val_words.data(), A_nnz,
        AT_row_words.data(), AT_col_ptr.data(), AT_val_words.data(),
        P_row_words.data(), P_col_ptr.data(), P_val_words.data(), num_cols, P_diag.data(),
        l.data(), u.data(), q.data(),
        1e-6f, 1.6f, rho.data(),
        1000, 300, true, // adaptive_rho = true
        x_out.data(), y_out.data(),
        &admm_iters, &pcg_iters, &r_prim, &r_dual, &status
    );

    std::cout << "--- Summary ---" << std::endl;
    std::cout << "Status: " << (status == 1 ? "Converged" : "Max Iters") << std::endl;
    std::cout << "ADMM Iterations: " << admm_iters << " (Target: > 50 to see rho update)" << std::endl;
    std::cout << "Final Primal Residual: " << r_prim << std::endl;
    std::cout << "Final Dual Residual: " << r_dual << std::endl;

    return 0;
}