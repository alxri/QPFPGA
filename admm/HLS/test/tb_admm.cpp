#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>

// Include your HLS design header
#include "admm.h"

// OSQP Constants
#define OSQP_RHO 1.0f
#define OSQP_RHO_EQ_OVER_RHO_INEQ 100
#define OSQP_RHO_TOL 0.01f
#define OSQP_INFTY 1e17f
#define OSQP_RHO_MAX 1e6f
#define OSQP_RHO_MIN 1e-6f

// Tiling bounds
#define TILE_ROWS MAX_ROWS
#define TILE_COLS MAX_COLS
#define MAX_TILE_ROWS (MAX_SIZE / TILE_ROWS)
#define MAX_TILE_COLS (MAX_SIZE / TILE_COLS)
#define MAX_TILES ((MAX_SIZE / MAX_ROWS) * (MAX_SIZE / MAX_COLS))
#define MAX_TILED_COLPTR_INTS (MAX_TILES * (TILE_COLS + 1))


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

// Helper to safely pack flat CSC data into hls::vector (Used for the Regular Matrix A)
static bool pack_csc_nnz_to_words(const std::vector<int> &row_idx, const std::vector<float> &values,
                                  std::vector<int16> &row_words, std::vector<float16> &val_words)
{
    if (row_idx.size() != values.size()) return false;
    const int nnz = (int)row_idx.size();

    for (int w = 0; w < MAX_NNZ_WORDS; ++w) {
        for (int lane = 0; lane < PACK_SIZE; ++lane) {
            row_words[w][lane] = 0;
            val_words[w][lane] = 0.0f;
        }
    }
    for (int idx = 0; idx < nnz; ++idx) {
        const int w = idx / PACK_SIZE;
        const int lane = idx % PACK_SIZE;
        row_words[w][lane] = row_idx[idx];
        val_words[w][lane] = values[idx];
    }
    return true;
}

// =========================================================================
// HELPER: Convert Global CSC to Tiled, Packed CSC
// =========================================================================
struct TileCscCols {
    std::vector<std::vector<int>> rows_by_col;
    std::vector<std::vector<float>> vals_by_col;
    explicit TileCscCols(int cols) : rows_by_col(cols), vals_by_col(cols) {}
};

static void build_tiled_csc(
    int global_rows, int global_cols,
    const std::vector<int> &g_col_ptr, const std::vector<int> &g_row_idx, const std::vector<float> &g_values,
    int &num_row_tiles, int &num_col_tiles,
    std::vector<int> &tile_nnz_counts,
    std::vector<int> &tile_nnz_offsets,
    std::vector<int> &tile_col_offsets,
    std::vector<int16> &row_idx_tiled,
    std::vector<int> &col_ptr_tiled,
    std::vector<float16> &values_tiled)
{
    num_row_tiles = (global_rows + TILE_ROWS - 1) / TILE_ROWS;
    num_col_tiles = (global_cols + TILE_COLS - 1) / TILE_COLS;
    const int num_tiles = num_row_tiles * num_col_tiles;

    // Allocate max sizes for HLS cosim
    tile_nnz_counts.assign(MAX_TILES, 0);
    tile_nnz_offsets.assign(MAX_TILES, 0);
    tile_col_offsets.assign(MAX_TILES, 0);
    col_ptr_tiled.assign(MAX_TILED_COLPTR_INTS, 0);
    row_idx_tiled.assign(MAX_NNZ_WORDS, int16(0));
    values_tiled.assign(MAX_NNZ_WORDS, float16(0.0f));

    std::vector<TileCscCols> tiles(num_tiles, TileCscCols(TILE_COLS));

    // 1) Slice global CSC into per-tile local CSC
    for (int c = 0; c < global_cols; ++c) {
        const int tc = c / TILE_COLS;
        const int local_c = c % TILE_COLS;
        for (int idx = g_col_ptr[c]; idx < g_col_ptr[c + 1]; ++idx) {
            const int r = g_row_idx[idx];
            const float v = g_values[idx];

            const int tr = r / TILE_ROWS;
            const int local_r = r % TILE_ROWS;
            const int tile_idx = tr * num_col_tiles + tc;
            tiles[tile_idx].rows_by_col[local_c].push_back(local_r);
            tiles[tile_idx].vals_by_col[local_c].push_back(v);
        }
    }

    // 2) Build 1D arrays
    int nnz_word_cursor = 0;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tile_col_offsets[tile_idx] = tile_idx * (TILE_COLS + 1);
    }

    for (int tr = 0; tr < num_row_tiles; ++tr) {
        for (int tc = 0; tc < num_col_tiles; ++tc) {
            const int tile_idx = tr * num_col_tiles + tc;

            // Build tile-local CSC col_ptr
            std::vector<int> local_col_ptr(TILE_COLS + 1, 0);
            int tile_nnz = 0;
            for (int c = 0; c < TILE_COLS; ++c) {
                tile_nnz += (int)tiles[tile_idx].rows_by_col[c].size();
                local_col_ptr[c + 1] = tile_nnz;
            }

            tile_nnz_counts[tile_idx] = tile_nnz;
            tile_nnz_offsets[tile_idx] = nnz_word_cursor;

            const int base = tile_col_offsets[tile_idx];
            for (int i = 0; i < (int)local_col_ptr.size(); ++i) {
                col_ptr_tiled[base + i] = local_col_ptr[i];
            }

            // Pack NNZ words for this tile
            std::vector<int> rows;
            std::vector<float> vals;
            rows.reserve(tile_nnz);
            vals.reserve(tile_nnz);
            for (int c = 0; c < TILE_COLS; ++c) {
                for (int k = 0; k < (int)tiles[tile_idx].rows_by_col[c].size(); ++k) {
                    rows.push_back(tiles[tile_idx].rows_by_col[c][k]);
                    vals.push_back(tiles[tile_idx].vals_by_col[c][k]);
                }
            }

            const int words = CEIL_DIV(tile_nnz, PACK_SIZE);
            for (int w = 0; w < words; ++w) {
                int16 row_word;
                float16 val_word;
                for (int lane = 0; lane < PACK_SIZE; ++lane) {
                    const int idx = w * PACK_SIZE + lane;
                    if (idx < tile_nnz) {
                        row_word[lane] = rows[idx];
                        val_word[lane] = vals[idx];
                    } else {
                        row_word[lane] = 0;
                        val_word[lane] = 0.0f;
                    }
                }
                row_idx_tiled[nnz_word_cursor + w] = row_word;
                values_tiled[nnz_word_cursor + w] = val_word;
            }
            nnz_word_cursor += words;
        }
    }
}


int main() {
    // Keep it small for Co-Simulation
    const int num_rows = 15;//15
    const int num_cols = 15;//15

    std::cout << "--- Starting ADMM Tiled Vitis HLS Testbench (" << num_rows << "x" << num_cols << ") ---" << std::endl;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> p_dist(1.0f, 3.0f);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> row_dist(0, num_rows - 1);
    std::uniform_int_distribution<int> constraint_type_dist(0, 2);
    std::uniform_real_distribution<float> slack_dist(0.5f, 5.0f);

    // -------------------------------------------------------------------------
    // 1. Generate Ground Truth x* and Matrix P
    // -------------------------------------------------------------------------
    std::vector<float> x_true(num_cols, 0.0f);
    std::vector<float> q(std::max(num_cols, (int)MAX_COLS), 0.0f);
    
    int P_nnz = num_cols;
    std::vector<float> P_values(P_nnz, 0.0f); 
    std::vector<int> P_row_idx(P_nnz, 0);
    std::vector<int> P_col_ptr(num_cols + 1, 0);
    std::vector<float> P_diag(std::max(num_cols, (int)MAX_COLS), 0.0f); 

    for (int c = 0; c < num_cols; ++c) {
        x_true[c] = x_dist(rng); 
        P_col_ptr[c] = c;
        P_row_idx[c] = c;
        float d = p_dist(rng);
        P_values[c] = d;
        P_diag[c] = d;
        q[c] = -d * x_true[c]; 
    }
    P_col_ptr[num_cols] = P_nnz;

    // -------------------------------------------------------------------------
    // 2. Generate Matrix A and AT
    // -------------------------------------------------------------------------
    int A_nnz = num_cols;
    std::vector<float> A_values(A_nnz, 0.0f);
    std::vector<int> A_row_idx(A_nnz, 0);
    std::vector<int> A_col_ptr(num_cols + 1, 0);
    std::vector<float> z_true(num_rows, 0.0f);
    
    for (int c = 0; c < num_cols; ++c) {
        A_col_ptr[c] = c;
        A_row_idx[c] = row_dist(rng);
        float v = a_dist(rng);
        if (std::fabs(v) < 1e-3f) v = 1e-3f;
        A_values[c] = v;
        z_true[A_row_idx[c]] += v * x_true[c];
    }
    A_col_ptr[num_cols] = A_nnz;

    std::vector<float> AT_values;
    std::vector<int> AT_row_idx;
    std::vector<int> AT_col_ptr;
    transpose_csc(num_rows, num_cols, A_col_ptr, A_row_idx, A_values, AT_col_ptr, AT_row_idx, AT_values);


    // -------------------------------------------------------------------------
    // 3. Pack Matrix Data (Regular and Tiled)
    // -------------------------------------------------------------------------
    
    // Regular A (Packed to words for update_preconditioner)
    std::vector<int16> A_row_words_reg(MAX_NNZ_WORDS);
    std::vector<float16> A_val_words_reg(MAX_NNZ_WORDS);
    std::vector<int> A_col_ptr_reg = A_col_ptr;
    A_col_ptr_reg.resize(MAX_COL_PTR, 0);
    pack_csc_nnz_to_words(A_row_idx, A_values, A_row_words_reg, A_val_words_reg);

    // Tiled A
    int A_num_row_tiles, A_num_col_tiles;
    std::vector<int> A_tile_nnz_counts, A_tile_nnz_offsets, A_tile_col_offsets, A_col_ptr_tiled;
    std::vector<int16> A_row_idx_tiled;
    std::vector<float16> A_values_tiled;
    build_tiled_csc(num_rows, num_cols, A_col_ptr, A_row_idx, A_values,
                    A_num_row_tiles, A_num_col_tiles, A_tile_nnz_counts, A_tile_nnz_offsets, A_tile_col_offsets,
                    A_row_idx_tiled, A_col_ptr_tiled, A_values_tiled);

    // Tiled AT
    int AT_num_row_tiles, AT_num_col_tiles;
    std::vector<int> AT_tile_nnz_counts, AT_tile_nnz_offsets, AT_tile_col_offsets, AT_col_ptr_tiled;
    std::vector<int16> AT_row_idx_tiled;
    std::vector<float16> AT_values_tiled;
    build_tiled_csc(num_cols, num_rows, AT_col_ptr, AT_row_idx, AT_values, // Dimensions swapped!
                    AT_num_row_tiles, AT_num_col_tiles, AT_tile_nnz_counts, AT_tile_nnz_offsets, AT_tile_col_offsets,
                    AT_row_idx_tiled, AT_col_ptr_tiled, AT_values_tiled);

    // Tiled P
    int P_num_row_tiles, P_num_col_tiles;
    std::vector<int> P_tile_nnz_counts, P_tile_nnz_offsets, P_tile_col_offsets, P_col_ptr_tiled;
    std::vector<int16> P_row_idx_tiled;
    std::vector<float16> P_values_tiled;
    build_tiled_csc(num_cols, num_cols, P_col_ptr, P_row_idx, P_values,
                    P_num_row_tiles, P_num_col_tiles, P_tile_nnz_counts, P_tile_nnz_offsets, P_tile_col_offsets,
                    P_row_idx_tiled, P_col_ptr_tiled, P_values_tiled);

    // -------------------------------------------------------------------------
    // 4. Assign Bounds based on z*
    // -------------------------------------------------------------------------
    std::vector<float> l(std::max(num_rows, (int)MAX_ROWS), 0.0f);
    std::vector<float> u(std::max(num_rows, (int)MAX_ROWS), 0.0f);
    std::vector<float> rho(std::max(num_rows, (int)MAX_ROWS), 0.0f);

    for (int i = 0; i < num_rows; i++) {
        int type = constraint_type_dist(rng);
        float slack1 = slack_dist(rng);
        float slack2 = slack_dist(rng);

        if (type == 0) { 
            l[i] = z_true[i];
            u[i] = z_true[i];
            rho[i] = OSQP_RHO * OSQP_RHO_EQ_OVER_RHO_INEQ;
        } 
        else if (type == 1) { 
            l[i] = z_true[i] - slack1;
            u[i] = z_true[i] + slack2;
            rho[i] = OSQP_RHO;
        } 
        else { 
            l[i] = -OSQP_INFTY;
            u[i] = OSQP_INFTY;
            rho[i] = OSQP_RHO_MIN;
        }
        rho[i] = std::max(OSQP_RHO_MIN, std::min(OSQP_RHO_MAX, rho[i]));
    }

    // -------------------------------------------------------------------------
    // 5. Execution parameters
    // -------------------------------------------------------------------------
    float alpha = 1.8f;
    float sigma = 1e-2f;
    float eps_abs = 1e-3f;
    float eps_rel = 1e-3f;
    float pcg_tol_frac = 1.0f;
    int admm_max_iterations = 2000;
    int pcg_max_iterations = 5;
    bool adaptive_rho = true;

    std::vector<float> x_out(std::max(num_cols, (int)MAX_COLS), 0.0f);
    std::vector<float> y_out(std::max(num_rows, (int)MAX_ROWS), 0.0f);
    int admm_num_iterations_out = 0;
    int pcg_num_iterations_out = 0;
    float r_prim_out = 0.0f, r_dual_out = 0.0f;
    int status_out = 0;

    std::cout << "Invoking top-level admm() function..." << std::endl;
    
    admm(
        num_rows, num_cols,
        
        // Regular Matrix A
        A_row_words_reg.data(), A_col_ptr_reg.data(), A_val_words_reg.data(), A_nnz,
        
        // Tiled Matrix A
        A_num_row_tiles, A_num_col_tiles,
        A_tile_nnz_counts.data(), A_tile_nnz_offsets.data(), A_tile_col_offsets.data(),
        A_row_idx_tiled.data(), A_col_ptr_tiled.data(), A_values_tiled.data(),
        
        // Tiled Matrix AT
        AT_num_row_tiles, AT_num_col_tiles,
        AT_tile_nnz_counts.data(), AT_tile_nnz_offsets.data(), AT_tile_col_offsets.data(),
        AT_row_idx_tiled.data(), AT_col_ptr_tiled.data(), AT_values_tiled.data(),
        
        // Tiled Matrix P
        P_num_row_tiles, P_num_col_tiles,
        P_tile_nnz_counts.data(), P_tile_nnz_offsets.data(), P_tile_col_offsets.data(),
        P_row_idx_tiled.data(), P_col_ptr_tiled.data(), P_values_tiled.data(),
        
        // Scalars and standard vectors
        P_diag.data(), l.data(), u.data(), q.data(),
        sigma, alpha, rho.data(),
        admm_max_iterations, pcg_max_iterations, adaptive_rho,
        eps_abs, eps_rel, pcg_tol_frac,
        
        // Outputs
        x_out.data(), y_out.data(),
        &admm_num_iterations_out, &pcg_num_iterations_out,
        &r_prim_out, &r_dual_out, &status_out
    );

    // -------------------------------------------------------------------------
    // 6. Verify Results against Ground Truth
    // -------------------------------------------------------------------------
    std::cout << "--- Simulation Results ---" << std::endl;
    std::cout << "Status: " << (status_out == 1 ? "Converged" : "Max Iterations Reached") << std::endl;
    std::cout << "ADMM Iterations: " << admm_num_iterations_out << std::endl;
    std::cout << "Total PCG Iterations: " << pcg_num_iterations_out << std::endl;

    std::cout << std::scientific << std::setprecision(5);
    std::cout << "Primal Residual: " << r_prim_out << std::endl;
    std::cout << "Dual Residual: " << r_dual_out << std::endl;
    
    float mae = 0.0f;
    for (int i = 0; i < num_cols; i++) {
        mae += std::fabs(x_out[i] - x_true[i]);
    }
    mae /= num_cols;
    
    std::cout << "\nMean Absolute Error from x_true: " << mae << std::endl;
    std::cout << "--- Full x_out vs Expected ---" << std::endl;
    for (int i = 0; i < num_cols; i++) {
        std::cout << "x[" << std::setw(2) << i << "]: " 
                  << std::setw(13) << x_out[i] 
                  << " | Expected: " << std::setw(13) << x_true[i] << std::endl;
    }

    if (status_out == 1 && mae < 1e-2f) {
        std::cout << "\n>>> SUCCESS: Problem converged perfectly to the ground truth! <<<" << std::endl;
        return 0; 
    } else {
        std::cout << "\n>>> FAILED: Did not converge to the expected target. <<<" << std::endl;
        return 1; 
    }
}