#include "utils.h"

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

uint32_t float_to_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
float uint_to_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }  
int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Matrix utilities (TiledMatrix declared in utils.h)
TiledMatrix build_tiled_csc(int global_rows, int global_cols, const vector<int>& cptr_in, const vector<int>& ridx_in, const vector<float>& vals_in, int tile_size) {
    TiledMatrix tm;
    tm.rtiles = ceil_div(global_rows, tile_size);
    tm.ctiles = ceil_div(global_cols, tile_size);
    int num_tiles = tm.rtiles * tm.ctiles;

    tm.counts.resize(num_tiles, 0);
    tm.noff.resize(num_tiles, 0);
    tm.coff.resize(num_tiles, 0);

    vector<vector<vector<int>>> t_rows(num_tiles, vector<vector<int>>(tile_size));
    vector<vector<vector<float>>> t_vals(num_tiles, vector<vector<float>>(tile_size));

    for (int c = 0; c < global_cols; ++c) {
        int tc = c / tile_size;
        int local_c = c % tile_size;
        for (int idx = cptr_in[c]; idx < cptr_in[c+1]; ++idx) {
            int r = ridx_in[idx];
            float v = vals_in[idx];
            int tr = r / tile_size;
            int tile_idx = tr * tm.ctiles + tc;
            t_rows[tile_idx][local_c].push_back(r % tile_size);
            t_vals[tile_idx][local_c].push_back(v);
        }
    }

    int nnz_word_cursor = 0;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tm.coff[tile_idx] = tm.cptr.size();
        
        vector<int> local_cptr(tile_size + 1, 0);
        int tile_nnz = 0;
        for (int c = 0; c < tile_size; ++c) {
            tile_nnz += t_rows[tile_idx][c].size();
            local_cptr[c+1] = tile_nnz;
        }

        tm.counts[tile_idx] = tile_nnz;
        tm.noff[tile_idx] = nnz_word_cursor;
        tm.cptr.insert(tm.cptr.end(), local_cptr.begin(), local_cptr.end());

        int words = ceil_div(tile_nnz, PACK_SIZE);
        int flat_idx = 0;
        vector<int> flat_r(words * PACK_SIZE, 0);
        vector<float> flat_v(words * PACK_SIZE, 0.0f);

        for (int c = 0; c < tile_size; ++c) {
            for (size_t i = 0; i < t_rows[tile_idx][c].size(); ++i) {
                flat_r[flat_idx] = t_rows[tile_idx][c][i];
                flat_v[flat_idx] = t_vals[tile_idx][c][i];
                flat_idx++;
            }
        }
        
        tm.ridx.insert(tm.ridx.end(), flat_r.begin(), flat_r.end());
        tm.vals.insert(tm.vals.end(), flat_v.begin(), flat_v.end());
        nnz_word_cursor += words;
    }
    
    if (tm.ridx.empty()) { tm.ridx.resize(PACK_SIZE, 0); tm.vals.resize(PACK_SIZE, 0.0f); }
    return tm;
}

void transpose_csc(int rows, int cols, const vector<int>& cptr, const vector<int>& ridx, const vector<float>& vals, vector<int>& cptr_t, vector<int>& ridx_t, vector<float>& vals_t) {
    vector<int> nnz_per_col(rows, 0);
    for (size_t i = 0; i < ridx.size(); ++i) nnz_per_col[ridx[i]]++;
    
    cptr_t.assign(rows + 1, 0);
    for (int c = 0; c < rows; ++c) cptr_t[c+1] = cptr_t[c] + nnz_per_col[c];
    
    vector<int> next = cptr_t;
    ridx_t.resize(ridx.size());
    vals_t.resize(vals.size());
    
    for (int c = 0; c < cols; ++c) {
        for (int idx = cptr[c]; idx < cptr[c+1]; ++idx) {
            int dst = next[ridx[idx]]++;
            ridx_t[dst] = c;
            vals_t[dst] = vals[idx];
        }
    }
}

void apply_scaling(
    int NUM_ROWS,
    int NUM_COLS,
    const std::vector<int>& A_cptr,
    const std::vector<int>& A_ridx,
    std::vector<float>& A_vals,
    std::vector<float>& P_diag,
    std::vector<float>& q,
    std::vector<float>& l,
    std::vector<float>& u,
    std::vector<float>& D,
    std::vector<float>& E,
    float& c_scale,
    int iterations)
{
    D.assign(NUM_ROWS, 1.0f);
    E.assign(NUM_COLS, 1.0f);
    // Scaling iterations
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<float> A_col_norm(NUM_COLS, 0.0f);
        std::vector<float> A_row_norm(NUM_ROWS, 0.0f);
        // Compute row/column infinity norms
        for (int c = 0; c < NUM_COLS; ++c) {
            for (int idx = A_cptr[c]; idx < A_cptr[c + 1]; ++idx) {
                int r = A_ridx[idx];
                float val = std::fabs(A_vals[idx]);
                A_col_norm[c] = std::max(A_col_norm[c], val);
                A_row_norm[r] = std::max(A_row_norm[r], val);
            }
        }

        std::vector<float> E_new(NUM_COLS);
        std::vector<float> D_new(NUM_ROWS);
        // Column scaling
        for (int c = 0; c < NUM_COLS; ++c) {
            float x_norm = std::max(
                std::max(std::fabs(P_diag[c]), A_col_norm[c]),
                1e-4f);
            E_new[c] = 1.0f / std::sqrt(x_norm);
            E[c] *= E_new[c];
            // P <- E P E
            P_diag[c] *= (E_new[c] * E_new[c]);
        }

        // Row scaling
        for (int r = 0; r < NUM_ROWS; ++r) {
            float z_norm = std::max(A_row_norm[r], 1e-4f);
            D_new[r] = 1.0f / std::sqrt(z_norm);
            D[r] *= D_new[r];
        }

        // A <- D A E
        for (int c = 0; c < NUM_COLS; ++c) {
            float ec = E_new[c];
            for (int idx = A_cptr[c]; idx < A_cptr[c + 1]; ++idx) {
                int r = A_ridx[idx];
                A_vals[idx] *= (D_new[r] * ec);
            }
        }
    }

    // Cost normalization
    float max_val = 1e-15f;
    for (int c = 0; c < NUM_COLS; ++c) {
        q[c] *= E[c];
        max_val = std::max(max_val, std::max(std::fabs(P_diag[c]), std::fabs(q[c])));
    }

    c_scale = std::max(1.0f / max_val, 1e-4f);

    // Apply final cost scaling
    for (int c = 0; c < NUM_COLS; ++c) {
        P_diag[c] *= c_scale;
        q[c] *= c_scale;
    }

    // Scale constraints
   for (int r = 0; r < NUM_ROWS; ++r) {
        if (l[r] > -0.9f * ADMM_INFTY)
            l[r] *= D[r];
        else
            l[r] = -ADMM_INFTY;

        if (u[r] < 0.9f * ADMM_INFTY)
            u[r] *= D[r];
        else
            u[r] = ADMM_INFTY;
    }
}

std::vector<float> build_rho_vector(
    int NUM_ROWS,
    const std::vector<float>& l,
    const std::vector<float>& u)
{
    std::vector<float> rho(NUM_ROWS, 1.0f);

    for (int r = 0; r < NUM_ROWS; ++r) {
        bool loose_l = (l[r] <= -0.9f * ADMM_INFTY);
        bool loose_u = (u[r] >= 0.9f * ADMM_INFTY);

        if (loose_l && loose_u) {
            rho[r] = 1e-6f;
        }
        else if (!loose_l && !loose_u && ((u[r] - l[r]) < 1e-3f)) {
            rho[r] = 100.0f; 
        }
    }
    return rho;
}

// High-level helpers
void copy_csc_from_raw(int ncols, int nnz, const int32_t* indptr, const int32_t* indices, const float* data,
                       std::vector<int>& cptr_out, std::vector<int>& ridx_out, std::vector<float>& vals_out)
{
    cptr_out.resize(ncols + 1);
    ridx_out.resize(nnz);
    vals_out.resize(nnz);
    for (int i = 0; i < ncols + 1; ++i) cptr_out[i] = indptr[i];
    for (int i = 0; i < nnz; ++i) ridx_out[i] = indices[i];
    for (int i = 0; i < nnz; ++i) vals_out[i] = data[i];
}

void build_diag_from_csc(int ncols, const std::vector<int>& cptr, const std::vector<int>& ridx, const std::vector<float>& vals,
                         std::vector<float>& diag_out)
{
    diag_out.assign(ncols, 0.0f);
    for (int c = 0; c < ncols; ++c) {
        for (int idx = cptr[c]; idx < cptr[c+1]; ++idx) {
            int r = ridx[idx];
            if (r == c) diag_out[c] = vals[idx];
        }
    }
}

void pack_indices_to_words(const std::vector<int>& ridx, std::vector<int32_words>& out_words)
{
    int words = ceil_div((int)ridx.size(), PACK_SIZE);
    out_words.assign(words, int32_words());
    for (size_t i = 0; i < out_words.size(); ++i) {
        for (int j = 0; j < PACK_SIZE; ++j) out_words[i].data[j] = 0;
    }
    for (size_t i = 0; i < ridx.size(); ++i) {
        out_words[i / PACK_SIZE].data[i % PACK_SIZE] = ridx[i];
    }
}

void pack_vals_to_words(const std::vector<float>& vals, std::vector<float32_words>& out_words)
{
    int words = ceil_div((int)vals.size(), PACK_SIZE);
    out_words.assign(words, float32_words());
    for (size_t i = 0; i < out_words.size(); ++i) {
        for (int j = 0; j < PACK_SIZE; ++j) out_words[i].data[j] = 0.0f;
    }
    for (size_t i = 0; i < vals.size(); ++i) {
        out_words[i / PACK_SIZE].data[i % PACK_SIZE] = vals[i];
    }
}