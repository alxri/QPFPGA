#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

// PYNQ libcma exports
extern "C" {
    void* cma_alloc(uint32_t len, uint32_t cacheable);
    unsigned long cma_get_phy_addr(void *buf);
    void cma_free(void *buf);
    void cma_flush_cache(void *buf, unsigned long phys_addr, int size);
    void cma_invalidate_cache(void *buf, unsigned long phys_addr, int size);
}

// ---------------------------------------------------------------------
// Configuration & Register Map
// ---------------------------------------------------------------------
#define PACK_SIZE 16
#define TILE_ROWS 8192
#define TILE_COLS 8192

#define ADMM_IP_CONTROL_BASE   0xA0000000 
#define ADMM_IP_CONTROL_R_BASE 0xA0010000
#define MAP_SIZE               0x20000UL
#define MAP_MASK               (MAP_SIZE - 1)

#define OSQP_RHO 1.0f
#define OSQP_RHO_EQ_OVER_RHO_INEQ 100
#define OSQP_RHO_TOL 0.01f
#define OSQP_INFTY 1e17f
#define OSQP_RHO_MAX 1e6f
#define OSQP_RHO_MIN 1e-6f

using namespace std;

struct int32_words { int32_t data[PACK_SIZE]; };
struct float32_words { float data[PACK_SIZE]; };

struct RunResult {
    int admm_iters;
    int pcg_iters;
    float r_prim;
    float r_dual;
    float viol;
    double obj;
    double hw_ms;
    float mae;
};

// ---------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

void write_reg(void *base, uint32_t offset, uint32_t val) { *((volatile uint32_t *)((uint8_t *)base + offset)) = val; }
uint32_t read_reg(void *base, uint32_t offset) { return *((volatile uint32_t *)((uint8_t *)base + offset)); }
void write_64bit_address(void *base, uint32_t offset, uintptr_t address) {
    write_reg(base, offset, (uint32_t)(address & 0xFFFFFFFF));
    write_reg(base, offset + 0x04, (uint32_t)((uint64_t)address >> 32));
}

uint32_t float_to_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
float uint_to_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }  
int ceil_div(int a, int b) { return (a + b - 1) / b; }

size_t get_file_elements(const string& filename, size_t element_size) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) { cout << "ERROR: Could not open " << filename << endl; exit(1); }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fclose(f);
    return sz / element_size;
}

template <typename T>
void load_bin(const string& filename, vector<T>& dest) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) { cout << "ERROR: Could not open " << filename << endl; exit(1); }
    size_t r = fread(dest.data(), sizeof(T), dest.size(), f);
    if (r != dest.size()) { cout << "WARNING: Read mismatch in " << filename << endl; }
    fclose(f);
}

// ---------------------------------------------------------------------
// CMA Memory Tracker
// ---------------------------------------------------------------------
struct CmaTracker {
    vector<void*> bufs;
    vector<size_t> sizes;
    
    template <typename T>
    T* alloc(size_t elements) {
        size_t bytes = elements * sizeof(T);
        T* ptr = (T*)cma_alloc(bytes, 1); 
        if (!ptr) { cout << "CMA allocation failed." << endl; exit(1); }
        memset(ptr, 0, bytes); 
        bufs.push_back(ptr);
        sizes.push_back(bytes);
        return ptr;
    }

    void flush_all() {
        for (size_t i = 0; i < bufs.size(); ++i) {
            cma_flush_cache(bufs[i], cma_get_phy_addr(bufs[i]), sizes[i]);
        }
    }

    void invalidate_all() {
        for (size_t i = 0; i < bufs.size(); ++i) {
            cma_invalidate_cache(bufs[i], cma_get_phy_addr(bufs[i]), sizes[i]);
        }
    }

    void free_all() {
        for (void* ptr : bufs) cma_free(ptr);
        bufs.clear(); sizes.clear();
    }
};

template <typename T>
void cma_copy(T* dest, const T* src, size_t elements) {
    memcpy(dest, src, elements * sizeof(T));
}

// ---------------------------------------------------------------------
// Matrix Tiling Engine
// ---------------------------------------------------------------------
struct TiledMatrix {
    int rtiles, ctiles;
    vector<int> counts, noff, coff, cptr, ridx;
    vector<float> vals;
};

TiledMatrix build_tiled_csc(int global_rows, int global_cols, const vector<int>& cptr_in, const vector<int>& ridx_in, const vector<float>& vals_in) {
    TiledMatrix tm;
    tm.rtiles = ceil_div(global_rows, TILE_ROWS);
    tm.ctiles = ceil_div(global_cols, TILE_COLS);
    int num_tiles = tm.rtiles * tm.ctiles;

    tm.counts.resize(num_tiles, 0);
    tm.noff.resize(num_tiles, 0);
    tm.coff.resize(num_tiles, 0);

    vector<vector<vector<int>>> t_rows(num_tiles, vector<vector<int>>(TILE_COLS));
    vector<vector<vector<float>>> t_vals(num_tiles, vector<vector<float>>(TILE_COLS));

    for (int c = 0; c < global_cols; ++c) {
        int tc = c / TILE_COLS;
        int local_c = c % TILE_COLS;
        for (int idx = cptr_in[c]; idx < cptr_in[c+1]; ++idx) {
            int r = ridx_in[idx];
            float v = vals_in[idx];
            int tr = r / TILE_ROWS;
            int tile_idx = tr * tm.ctiles + tc;
            t_rows[tile_idx][local_c].push_back(r % TILE_ROWS);
            t_vals[tile_idx][local_c].push_back(v);
        }
    }

    int nnz_word_cursor = 0;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tm.coff[tile_idx] = tm.cptr.size();
        
        vector<int> local_cptr(TILE_COLS + 1, 0);
        int tile_nnz = 0;
        for (int c = 0; c < TILE_COLS; ++c) {
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

        for (int c = 0; c < TILE_COLS; ++c) {
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

// ---------------------------------------------------------------------
// Main Program
// ---------------------------------------------------------------------
int main() {
    cout << "Loading overlay from /home/xilinx/admm/admm_fixed_tiles.bit...\n";
    cout << "Overlay loaded!\n\n";

    string ddir = "./data_bin/";
    const int num_rows = 1024;
    const int num_cols = 1024;

    cout << "Loading Data from " << ddir << "...\n";

    // 1. Load Ground Truth & Matrices
    vector<int> A_cptr(num_cols + 1); load_bin(ddir + "A_indptr.bin", A_cptr);
    int A_nnz = A_cptr.back();

    vector<int> P_cptr(num_cols + 1); load_bin(ddir + "P_indptr.bin", P_cptr);
    int P_nnz = P_cptr.back();

    std::vector<float> x_true(num_cols), q(num_cols), P_diag(num_cols), l(num_rows), u(num_rows);
    std::vector<int> A_ridx(A_nnz); std::vector<float> A_vals(A_nnz);
    std::vector<int> P_ridx(P_nnz); std::vector<float> P_vals(P_nnz);

    load_bin(ddir + "x_true.bin", x_true);
    load_bin(ddir + "q.bin", q);
    load_bin(ddir + "l.bin", l);
    load_bin(ddir + "u.bin", u);
    load_bin(ddir + "P_diag.bin", P_diag);
    load_bin(ddir + "A_indices.bin", A_ridx);
    load_bin(ddir + "A_data.bin", A_vals);
    load_bin(ddir + "P_indices.bin", P_ridx);
    load_bin(ddir + "P_data.bin", P_vals);

    // Create Transpose of A
    std::vector<float> AT_vals; std::vector<int> AT_ridx; std::vector<int> AT_cptr;
    transpose_csc(num_rows, num_cols, A_cptr, A_ridx, A_vals, AT_cptr, AT_ridx, AT_vals);

    // Save originals for objective calculation
    vector<float> P_diag_orig = P_diag;
    vector<float> q_orig = q;
    std::vector<float> rho(num_rows, 0.0f);
    
    // 2. Apply Scaling
    cout << "Applying Scaling...\n";
    vector<float> E(num_cols, 1.0f), D(num_rows, 1.0f);
    for (int iter = 0; iter < 10; ++iter) {
        vector<float> A_col_norm(num_cols, 0.0f), A_row_norm(num_rows, 0.0f);
        for (int c = 0; c < num_cols; ++c) {
            for (int idx = A_cptr[c]; idx < A_cptr[c+1]; ++idx) {
                float val = std::abs(A_vals[idx]);
                A_col_norm[c] = std::max(A_col_norm[c], val);
                A_row_norm[A_ridx[idx]] = std::max(A_row_norm[A_ridx[idx]], val);
            }
        }
        vector<float> E_new(num_cols), D_new(num_rows);
        for (int c = 0; c < num_cols; ++c) {
            float x_norm = std::max(std::max(std::abs(P_diag[c]), A_col_norm[c]), 1e-4f);
            E_new[c] = 1.0f / std::sqrt(x_norm);
            E[c] *= E_new[c];
            P_diag[c] *= (E_new[c] * E_new[c]);
        }
        for (int r = 0; r < num_rows; ++r) {
            float z_norm = std::max(A_row_norm[r], 1e-4f);
            D_new[r] = 1.0f / std::sqrt(z_norm);
            D[r] *= D_new[r];
        }
        for (int c = 0; c < num_cols; ++c) {
            for (int idx = A_cptr[c]; idx < A_cptr[c+1]; ++idx) {
                A_vals[idx] *= (D_new[A_ridx[idx]] * E_new[c]);
            }
        }
    }

    for (int c = 0; c < num_cols; ++c) q[c] *= E[c];

    for (int r = 0; r < num_rows; ++r) {
        if (l[r] <= -OSQP_INFTY * 0.9f) l[r] = -OSQP_INFTY;
        else l[r] *= D[r];

        if (u[r] >= OSQP_INFTY * 0.9f) u[r] = OSQP_INFTY;
        else u[r] *= D[r];
    }

    float max_val = 1e-15f;
    for (int c = 0; c < num_cols; ++c) {
        max_val = std::max(max_val, std::max(std::abs(P_diag[c]), std::abs(q[c])));
    }
    float c_scale = std::max(1.0f / max_val, 1e-4f);
    
    for (int c = 0; c < num_cols; ++c) { 
        P_diag[c] *= c_scale; 
        q[c] *= c_scale; 
    }

    // Process Rho values post-equilibration
    for (int i = 0; i < num_rows; i++) {
        rho[i] = OSQP_RHO;
        bool fin_l = (std::abs(l[i]) < OSQP_INFTY * 0.9f);
        bool fin_u = (std::abs(u[i]) < OSQP_INFTY * 0.9f);
        if (!fin_l && !fin_u) {
            rho[i] = OSQP_RHO_MIN;
        } else if (fin_l && fin_u && ((u[i] - l[i]) < 0.01f)) {
            rho[i] = OSQP_RHO * OSQP_RHO_EQ_OVER_RHO_INEQ;
        }
        rho[i] = std::max(OSQP_RHO_MIN, std::min(OSQP_RHO_MAX, rho[i]));
    }
    
    // --- CRITICAL FIX: Ensure P_vals is correctly updated for the hardware matrix ---
    for (int c = 0; c < num_cols; ++c) {
        for (int idx = P_cptr[c]; idx < P_cptr[c+1]; ++idx) {
            P_vals[idx] = P_diag[c];
        }
    }

    // 3. Matrix Tiling
    cout << "Slicing and Allocating Tiled Matrices...\n";
    TiledMatrix tm_A = build_tiled_csc(num_rows, num_cols, A_cptr, A_ridx, A_vals);
    TiledMatrix tm_AT = build_tiled_csc(num_cols, num_rows, AT_cptr, AT_ridx, AT_vals);
    TiledMatrix tm_P = build_tiled_csc(num_cols, num_cols, P_cptr, P_ridx, P_vals);

    // 4. CACHEABLE CMA Allocations
    CmaTracker cma;

    int a_words_cnt = ceil_div(A_nnz, PACK_SIZE);
    int32_words* A_reg_ridx = cma.alloc<int32_words>(a_words_cnt);
    float32_words* A_reg_vals = cma.alloc<float32_words>(a_words_cnt);
    int* A_reg_cptr = cma.alloc<int>(num_cols + 1);

    vector<int32_words> temp_A_ridx(a_words_cnt, {0});
    vector<float32_words> temp_A_vals(a_words_cnt, {0.0f});
    for (int i=0; i<A_nnz; i++) {
        temp_A_ridx[i / PACK_SIZE].data[i % PACK_SIZE] = A_ridx[i];
        temp_A_vals[i / PACK_SIZE].data[i % PACK_SIZE] = A_vals[i];
    }
    
    cma_copy(A_reg_cptr, A_cptr.data(), num_cols + 1);
    cma_copy(A_reg_ridx, temp_A_ridx.data(), a_words_cnt);
    cma_copy(A_reg_vals, temp_A_vals.data(), a_words_cnt);

    #define ALLOC_TILE_CMA(mat, tm) \
        int* hw_tile_##mat##_cnt = cma.alloc<int>(tm.counts.size()); cma_copy(hw_tile_##mat##_cnt, tm.counts.data(), tm.counts.size()); \
        int* hw_tile_##mat##_noff = cma.alloc<int>(tm.noff.size()); cma_copy(hw_tile_##mat##_noff, tm.noff.data(), tm.noff.size()); \
        int* hw_tile_##mat##_coff = cma.alloc<int>(tm.coff.size()); cma_copy(hw_tile_##mat##_coff, tm.coff.data(), tm.coff.size()); \
        int* hw_tile_##mat##_cptr = cma.alloc<int>(tm.cptr.size()); cma_copy(hw_tile_##mat##_cptr, tm.cptr.data(), tm.cptr.size()); \
        int32_words* hw_tile_##mat##_ridx = cma.alloc<int32_words>(tm.ridx.size()/PACK_SIZE); cma_copy(hw_tile_##mat##_ridx, (const int32_words*)tm.ridx.data(), tm.ridx.size()/PACK_SIZE); \
        float32_words* hw_tile_##mat##_vals = cma.alloc<float32_words>(tm.vals.size()/PACK_SIZE); cma_copy(hw_tile_##mat##_vals, (const float32_words*)tm.vals.data(), tm.vals.size()/PACK_SIZE);

    ALLOC_TILE_CMA(A, tm_A);
    ALLOC_TILE_CMA(AT, tm_AT);
    ALLOC_TILE_CMA(P, tm_P);

    float* hw_Pdiag = cma.alloc<float>(num_cols); cma_copy(hw_Pdiag, P_diag.data(), num_cols);
    float* hw_l = cma.alloc<float>(num_rows);     cma_copy(hw_l, l.data(), num_rows);
    float* hw_u = cma.alloc<float>(num_rows);     cma_copy(hw_u, u.data(), num_rows);
    float* hw_q = cma.alloc<float>(num_cols);     cma_copy(hw_q, q.data(), num_cols);
    float* hw_rho = cma.alloc<float>(num_rows);   cma_copy(hw_rho, rho.data(), num_rows);
    float* hw_x = cma.alloc<float>(num_cols);
    float* hw_y = cma.alloc<float>(num_rows);

    // 5. Memory Map
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    void *ip_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, ADMM_IP_CONTROL_BASE & ~MAP_MASK);
    void *ctrl = (void*)((uint8_t*)ip_base + (ADMM_IP_CONTROL_BASE & MAP_MASK));
    void *ctrl_r = (void*)((uint8_t*)ip_base + (ADMM_IP_CONTROL_R_BASE & MAP_MASK));

    // Pointer Setup
    write_64bit_address(ctrl_r, 0x010, cma_get_phy_addr(A_reg_ridx));
    write_64bit_address(ctrl_r, 0x01c, cma_get_phy_addr(A_reg_cptr));
    write_64bit_address(ctrl_r, 0x028, cma_get_phy_addr(A_reg_vals));
    write_64bit_address(ctrl_r, 0x034, cma_get_phy_addr(hw_tile_A_cnt)); write_64bit_address(ctrl_r, 0x040, cma_get_phy_addr(hw_tile_A_noff));
    write_64bit_address(ctrl_r, 0x04c, cma_get_phy_addr(hw_tile_A_coff)); write_64bit_address(ctrl_r, 0x058, cma_get_phy_addr(hw_tile_A_ridx));
    write_64bit_address(ctrl_r, 0x064, cma_get_phy_addr(hw_tile_A_cptr)); write_64bit_address(ctrl_r, 0x070, cma_get_phy_addr(hw_tile_A_vals));
    write_64bit_address(ctrl_r, 0x07c, cma_get_phy_addr(hw_tile_AT_cnt)); write_64bit_address(ctrl_r, 0x088, cma_get_phy_addr(hw_tile_AT_noff));
    write_64bit_address(ctrl_r, 0x094, cma_get_phy_addr(hw_tile_AT_coff)); write_64bit_address(ctrl_r, 0x0a0, cma_get_phy_addr(hw_tile_AT_ridx));
    write_64bit_address(ctrl_r, 0x0ac, cma_get_phy_addr(hw_tile_AT_cptr)); write_64bit_address(ctrl_r, 0x0b8, cma_get_phy_addr(hw_tile_AT_vals));
    write_64bit_address(ctrl_r, 0x0c4, cma_get_phy_addr(hw_tile_P_cnt)); write_64bit_address(ctrl_r, 0x0d0, cma_get_phy_addr(hw_tile_P_noff));
    write_64bit_address(ctrl_r, 0x0dc, cma_get_phy_addr(hw_tile_P_coff)); write_64bit_address(ctrl_r, 0x0e8, cma_get_phy_addr(hw_tile_P_ridx));
    write_64bit_address(ctrl_r, 0x0f4, cma_get_phy_addr(hw_tile_P_cptr)); write_64bit_address(ctrl_r, 0x100, cma_get_phy_addr(hw_tile_P_vals));
    write_64bit_address(ctrl_r, 0x10c, cma_get_phy_addr(hw_Pdiag)); write_64bit_address(ctrl_r, 0x118, cma_get_phy_addr(hw_l));
    write_64bit_address(ctrl_r, 0x124, cma_get_phy_addr(hw_u)); write_64bit_address(ctrl_r, 0x130, cma_get_phy_addr(hw_q));
    write_64bit_address(ctrl_r, 0x13c, cma_get_phy_addr(hw_rho)); write_64bit_address(ctrl_r, 0x148, cma_get_phy_addr(hw_x));
    write_64bit_address(ctrl_r, 0x154, cma_get_phy_addr(hw_y));

    RunResult results[2];

    float sigma = 1e-2f;
    float alpha = 1.8f;
    float eps_abs = 1e-3f, eps_rel = 1e-3f;
    float pcg_tol_fraction = 1.0f;
    int admm_max_iter = 2000, pcg_max_iter = 5;

    // 6. Execution Loop
    for (int ad_rho = 0; ad_rho <= 1; ++ad_rho) {
        printf("=== HW Run (adaptive_rho=%d) ===\n", ad_rho);

        volatile float* v_hw_x = static_cast<volatile float*>(hw_x);
        for(int i = 0; i < num_cols; i++) {
            v_hw_x[i] = 0.0f;
        }
        volatile float* v_hw_y = static_cast<volatile float*>(hw_y);
        for(int i = 0; i < num_rows; i++) {
            v_hw_y[i] = 0.0f;
        }

        write_reg(ctrl, 0x10, num_rows); write_reg(ctrl, 0x18, num_cols); write_reg(ctrl, 0x20, A_nnz);
        write_reg(ctrl, 0x28, tm_A.rtiles); write_reg(ctrl, 0x30, tm_A.ctiles);
        write_reg(ctrl, 0x38, tm_AT.rtiles); write_reg(ctrl, 0x40, tm_AT.ctiles);
        write_reg(ctrl, 0x48, tm_P.rtiles); write_reg(ctrl, 0x50, tm_P.ctiles);
        write_reg(ctrl, 0x58, float_to_uint(sigma)); write_reg(ctrl, 0x60, float_to_uint(alpha));
        write_reg(ctrl, 0x68, admm_max_iter); write_reg(ctrl, 0x70, pcg_max_iter);
        write_reg(ctrl, 0x78, ad_rho); 
        write_reg(ctrl, 0x80, float_to_uint(eps_abs)); write_reg(ctrl, 0x88, float_to_uint(eps_rel)); 
        write_reg(ctrl, 0x90, float_to_uint(pcg_tol_fraction));

        // Flush Cache to DDR
        cma.flush_all();

        double hw_start = get_time_ms();
        write_reg(ctrl, 0x00, 0x01);
        
        // High-res integer spin delay (approx 10us)
        auto poll_start = std::chrono::high_resolution_clock::now();
        while ((read_reg(ctrl, 0x00) & 0x02) == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            while (std::chrono::duration_cast<std::chrono::microseconds>(now - poll_start).count() < 10) {
                now = std::chrono::high_resolution_clock::now();
            }
            poll_start = std::chrono::high_resolution_clock::now();
        }
        
        double hw_end = get_time_ms();

        // Invalidate Cache to read DDR
        cma.invalidate_all();

        int admm_iters = read_reg(ctrl, 0x98);
        int pcg_iters = read_reg(ctrl, 0xa8);
        float p_res = uint_to_float(read_reg(ctrl, 0xb8));
        float d_res = uint_to_float(read_reg(ctrl, 0xc8));
        int status = read_reg(ctrl, 0xd8);

        // Unscale x and calculate Objective & MAE
        vector<float> x_unscaled(num_cols);
        float mae = 0.0f;
        double obj = 0.0;

        for (int i = 0; i < num_cols; i++) {
            x_unscaled[i] = hw_x[i] * E[i];
            mae += std::fabs(x_unscaled[i] - x_true[i]);
            obj += 0.5 * P_diag_orig[i] * x_unscaled[i] * x_unscaled[i] + q_orig[i] * x_unscaled[i];
        }
        mae /= num_cols;

        results[ad_rho] = {admm_iters, pcg_iters, p_res, d_res, 0.0f, obj, hw_end - hw_start, mae};

        printf("HW execution time: %.4f ms\n", hw_end - hw_start);
        printf("Status: %s\n", status == 1 ? "Converged" : "Max Iterations");
        printf("ADMM Iterations: %d\n", admm_iters);
        printf("PCG Iterations : %d\n", pcg_iters);
        printf("Primal Residual: %.5e\n", p_res);
        printf("Dual Residual  : %.5e\n", d_res);
        printf("Objective Value: %.6e\n", obj);
        printf("Mean Abs Error : %.5e\n", mae);

        printf("\n--- First 10 Elements of x_unscaled ---\n");
        for (int i = 0; i < std::min(10, num_cols); i++) {
            printf("  x[%2d] = %13.6f | Expected: %13.6f\n", i, x_unscaled[i], x_true[i]);
        }
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("ADMM iterations: off=%d | on=%d\n", results[0].admm_iters, results[1].admm_iters);
    printf("PCG iterations:  off=%d | on=%d\n", results[0].pcg_iters, results[1].pcg_iters);
    printf("Objective Value: off=%.6e | on=%.6e\n", results[0].obj, results[1].obj);
    printf("MAE from Truth:  off=%.3e | on=%.3e\n", results[0].mae, results[1].mae);
    printf("HW time (ms):    off=%.3f | on=%.3f\n\n", results[0].hw_ms, results[1].hw_ms);

    munmap(ip_base, MAP_SIZE); close(mem_fd);
    cma.free_all();
    printf("Buffers released.\n");
    return 0;
}