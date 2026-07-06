#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <pmt.h>

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

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------
// CMA Memory Tracker (Handles Cacheable Allocation + Flushing)
// ---------------------------------------------------------------------
struct CmaTracker {
    vector<void*> bufs;
    vector<size_t> sizes;
    
    template <typename T>
    T* alloc(size_t elements) {
        size_t bytes = elements * sizeof(T);
        // allocate as CACHEABLE (1) to allow AXI bursts and prevent Bus Faults
        T* ptr = (T*)cma_alloc(bytes, 1); 
        if (!ptr) { cout << "CMA allocation failed." << endl; exit(1); }
        memset(ptr, 0, bytes); // Standard memset works now!
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
    memcpy(dest, src, elements * sizeof(T)); // Standard memcpy works now!
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

int main() {
    cout << "Loading overlay from /home/xilinx/admm/admm_fixed_tiles.bit...\n";
    cout << "Overlay loaded!\n\n";

    // 1. Parameters
    float sigma = 1e-2f;
    float alpha = 1.8f;
    float eps_abs = 5e-3f, eps_rel = 5e-3f;
    float pcg_tol_fraction = 1.0f;
    int admm_max_iter = 2000, pcg_max_iter = 5;

    // 2. Data Loading
    string ddir = "./data_bin/";
    int NUM_COLS = get_file_elements(ddir + "q.bin", sizeof(float));
    int NUM_ROWS = get_file_elements(ddir + "l.bin", sizeof(float));

    vector<int> A_cptr(NUM_COLS + 1);
    load_bin(ddir + "A_indptr.bin", A_cptr);
    int A_NNZ = A_cptr.back();

    vector<int> P_cptr(NUM_COLS + 1);
    load_bin(ddir + "P_indptr.bin", P_cptr);
    int P_NNZ = P_cptr.back();

    cout << "Loading QP data from: " << ddir << "\n";
    cout << "Problem size: " << NUM_ROWS << " x " << NUM_COLS << "\n\n";

    vector<float> P_diag(NUM_COLS), q(NUM_COLS), l(NUM_ROWS), u(NUM_ROWS);
    vector<int> A_ridx(A_NNZ);
    vector<float> A_vals(A_NNZ);
    vector<int> P_ridx(P_NNZ);
    vector<float> P_vals(P_NNZ);

    load_bin(ddir + "P_diag.bin", P_diag);
    load_bin(ddir + "q.bin", q);
    load_bin(ddir + "l.bin", l);
    load_bin(ddir + "u.bin", u);
    load_bin(ddir + "A_indices.bin", A_ridx);
    load_bin(ddir + "A_data.bin", A_vals);
    load_bin(ddir + "P_indices.bin", P_ridx);
    load_bin(ddir + "P_data.bin", P_vals);

    vector<float> P_diag_orig = P_diag;
    vector<float> q_orig = q;
    vector<float> l_orig = l;
    vector<float> u_orig = u;
    vector<float> A_vals_orig = A_vals;

    // 3. Scaling
    cout << "Applying Scaling...\n\n";
    vector<float> E(NUM_COLS, 1.0f), D(NUM_ROWS, 1.0f);
    for (int iter = 0; iter < 10; ++iter) {
        vector<float> A_col_norm(NUM_COLS, 0.0f), A_row_norm(NUM_ROWS, 0.0f);
        for (int c = 0; c < NUM_COLS; ++c) {
            for (int idx = A_cptr[c]; idx < A_cptr[c+1]; ++idx) {
                float val = abs(A_vals[idx]);
                A_col_norm[c] = max(A_col_norm[c], val);
                A_row_norm[A_ridx[idx]] = max(A_row_norm[A_ridx[idx]], val);
            }
        }
        
        vector<float> E_new(NUM_COLS), D_new(NUM_ROWS);
        for (int c = 0; c < NUM_COLS; ++c) {
            float x_norm = max(max(abs(P_diag[c]), A_col_norm[c]), 1e-4f);
            E_new[c] = 1.0f / sqrt(x_norm);
            E[c] *= E_new[c];
            P_diag[c] *= (E_new[c] * E_new[c]);
        }
        for (int r = 0; r < NUM_ROWS; ++r) {
            float z_norm = max(A_row_norm[r], 1e-4f);
            D_new[r] = 1.0f / sqrt(z_norm);
            D[r] *= D_new[r];
        }
        for (int c = 0; c < NUM_COLS; ++c) {
            for (int idx = A_cptr[c]; idx < A_cptr[c+1]; ++idx) {
                A_vals[idx] *= (D_new[A_ridx[idx]] * E_new[c]);
            }
        }
    }

    float max_val = 1e-15f; 
    for (int c = 0; c < NUM_COLS; ++c) {
        q[c] *= E[c];
        max_val = max(max_val, max(abs(P_diag[c]), abs(q[c])));
    }
    float c_scale = max(1.0f / max_val, 1e-4f);
    
    for (int c = 0; c < NUM_COLS; ++c) { P_diag[c] *= c_scale; q[c] *= c_scale; }
    for (int r = 0; r < NUM_ROWS; ++r) {
        l[r] = isinf(l[r]) ? l[r] : l[r] * D[r];
        u[r] = isinf(u[r]) ? u[r] : u[r] * D[r];
    }

    vector<float> rho(NUM_ROWS, 1.0f);
    for(int r = 0; r < NUM_ROWS; r++) {
        bool fin_l = !isinf(l[r]);
        bool fin_u = !isinf(u[r]);
        if (!fin_l && !fin_u) rho[r] = 1e-6f;
        else if (fin_l && fin_u && (u[r] - l[r] < 0.01f)) rho[r] = 100.0f;
    }

    // 4. Matrix Transpositions & Tiling
    vector<int> AT_cptr, AT_ridx; vector<float> AT_vals;
    transpose_csc(NUM_ROWS, NUM_COLS, A_cptr, A_ridx, A_vals, AT_cptr, AT_ridx, AT_vals);

    cout << "Slicing and Allocating Tiled Matrices... (This may take a moment)\n";
    cout << "  Tiling Matrix A...\n";
    TiledMatrix tm_A = build_tiled_csc(NUM_ROWS, NUM_COLS, A_cptr, A_ridx, A_vals);
    cout << "  Tiling Matrix AT...\n";
    TiledMatrix tm_AT = build_tiled_csc(NUM_COLS, NUM_ROWS, AT_cptr, AT_ridx, AT_vals);

    for (int c = 0; c < NUM_COLS; ++c) {
        for (int idx = P_cptr[c]; idx < P_cptr[c+1]; ++idx) {
            P_vals[idx] = P_diag[c]; 
        }
    }
    cout << "  Tiling Matrix P...\n\n";
    TiledMatrix tm_P = build_tiled_csc(NUM_COLS, NUM_COLS, P_cptr, P_ridx, P_vals);

    // 5. Cacheable CMA Allocations
    CmaTracker cma;

    int a_words_cnt = ceil_div(A_NNZ, PACK_SIZE);
    int32_words* A_reg_ridx = cma.alloc<int32_words>(a_words_cnt);
    float32_words* A_reg_vals = cma.alloc<float32_words>(a_words_cnt);
    int* A_reg_cptr = cma.alloc<int>(NUM_COLS + 1);

    vector<int32_words> temp_A_ridx(a_words_cnt, {0});
    vector<float32_words> temp_A_vals(a_words_cnt, {0.0f});
    for (int i=0; i<A_NNZ; i++) {
        temp_A_ridx[i / PACK_SIZE].data[i % PACK_SIZE] = A_ridx[i];
        temp_A_vals[i / PACK_SIZE].data[i % PACK_SIZE] = A_vals[i];
    }
    
    cma_copy(A_reg_cptr, A_cptr.data(), NUM_COLS + 1);
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

    float* hw_Pdiag = cma.alloc<float>(NUM_COLS); cma_copy(hw_Pdiag, P_diag.data(), NUM_COLS);
    float* hw_l = cma.alloc<float>(NUM_ROWS);     cma_copy(hw_l, l.data(), NUM_ROWS);
    float* hw_u = cma.alloc<float>(NUM_ROWS);     cma_copy(hw_u, u.data(), NUM_ROWS);
    float* hw_q = cma.alloc<float>(NUM_COLS);     cma_copy(hw_q, q.data(), NUM_COLS);
    float* hw_rho = cma.alloc<float>(NUM_ROWS);   cma_copy(hw_rho, rho.data(), NUM_ROWS);
    float* hw_x = cma.alloc<float>(NUM_COLS);
    float* hw_y = cma.alloc<float>(NUM_ROWS);

    // 6. Memory Map
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

    // 7. Execution Loop
    for (int ad_rho = 0; ad_rho <= 1; ++ad_rho) {
        printf("=== HW Run (adaptive_rho=%d) ===\n", ad_rho);

        // Reset outputs
        memset(hw_x, 0, NUM_COLS * sizeof(float));
        memset(hw_y, 0, NUM_ROWS * sizeof(float));

        write_reg(ctrl, 0x10, NUM_ROWS); write_reg(ctrl, 0x18, NUM_COLS); write_reg(ctrl, 0x20, A_NNZ);
        write_reg(ctrl, 0x28, tm_A.rtiles); write_reg(ctrl, 0x30, tm_A.ctiles);
        write_reg(ctrl, 0x38, tm_AT.rtiles); write_reg(ctrl, 0x40, tm_AT.ctiles);
        write_reg(ctrl, 0x48, tm_P.rtiles); write_reg(ctrl, 0x50, tm_P.ctiles);
        write_reg(ctrl, 0x58, float_to_uint(sigma)); write_reg(ctrl, 0x60, float_to_uint(alpha));
        write_reg(ctrl, 0x68, admm_max_iter); write_reg(ctrl, 0x70, pcg_max_iter);
        write_reg(ctrl, 0x78, ad_rho); 
        write_reg(ctrl, 0x80, float_to_uint(eps_abs)); write_reg(ctrl, 0x88, float_to_uint(eps_rel)); 
        write_reg(ctrl, 0x90, float_to_uint(pcg_tol_fraction));

        // SYNC CACHE TO RAM BEFORE HW START
        cma.flush_all();

        std::unique_ptr<pmt::PMT> sensor(pmt::xilinx::Xilinx::Create());
        pmt::State start, end;
        start = sensor->Read();
        
        double hw_start = get_time_ms();
        write_reg(ctrl, 0x00, 0x01);
        
        // Wait loop + Throttle AXI-Lite
        while ((read_reg(ctrl, 0x00) & 0x02) == 0) {
            usleep(100); 
        }
        
        double hw_end = get_time_ms();
        end = sensor->Read();

        // SYNC RAM TO CACHE AFTER HW FINISHES
        cma.invalidate_all();

        std::cout << "Power Measurements (Watts):\n";
        std::cout << "  FPGA Core Logic: " << end.watt_[1] << " W\n";
        std::cout << "  FPGA AUX:        " << end.watt_[2] << " W\n";
        std::cout << "  FPGA Total:      " << end.watt_[3] << " W\n";
        std::cout << "  Board Total:     " << end.watt_[4] << " W\n";
        std::cout << std::defaultfloat << "\n";

        int admm_iters = read_reg(ctrl, 0x98);
        int pcg_iters = read_reg(ctrl, 0xa8);
        float p_res = uint_to_float(read_reg(ctrl, 0xb8));
        float d_res = uint_to_float(read_reg(ctrl, 0xc8));
        int status = read_reg(ctrl, 0xd8);
        float avg_pcg = admm_iters > 0 ? (float)pcg_iters / admm_iters : 0.0f;

        vector<float> x_unscaled(NUM_COLS);
        for (int i = 0; i < NUM_COLS; i++) {
            x_unscaled[i] = hw_x[i] * E[i];
        }

        double obj = 0.0;
        for (int i = 0; i < NUM_COLS; i++) {
            obj += 0.5 * P_diag_orig[i] * x_unscaled[i] * x_unscaled[i] + q_orig[i] * x_unscaled[i];
        }

        vector<float> Ax(NUM_ROWS, 0.0f);
        for (int c = 0; c < NUM_COLS; c++) {
            for (int idx = A_cptr[c]; idx < A_cptr[c+1]; idx++) {
                Ax[A_ridx[idx]] += A_vals_orig[idx] * x_unscaled[c];
            }
        }
        float max_viol = 0.0f;
        for (int r = 0; r < NUM_ROWS; r++) {
            float viol_l = l_orig[r] - Ax[r];
            float viol_u = Ax[r] - u_orig[r];
            float v = max(0.0f, max(viol_l, viol_u));
            max_viol = max(max_viol, v);
        }

        results[ad_rho] = {admm_iters, pcg_iters, p_res, d_res, max_viol, obj, hw_end - hw_start};

        printf("HW execution time: %.4f ms\n", hw_end - hw_start);
        printf("Status: %s\n", status == 1 ? "Converged" : "Max Iterations");
        printf("ADMM Iterations: %d\n", admm_iters);
        printf("PCG Iterations : %d (Average pcg/admm: %.1f)\n", pcg_iters, avg_pcg);
        printf("Primal Residual: %.5e\n", p_res);
        printf("Dual Residual  : %.5e\n", d_res);
        printf("Objective Value: %.6e\n", obj);
        printf("Max Violation  : %.3e\n\n", max_viol);
    }

    printf("=== Summary ===\n");
    printf("ADMM iterations: off=%d | on=%d\n", results[0].admm_iters, results[1].admm_iters);
    printf("PCG iterations:  off=%d | on=%d\n", results[0].pcg_iters, results[1].pcg_iters);
    printf("Primal residual: off=%.3e | on=%.3e\n", results[0].r_prim, results[1].r_prim);
    printf("Dual residual:   off=%.3e | on=%.3e\n", results[0].r_dual, results[1].r_dual);
    printf("Violation:       off=%.3e | on=%.3e\n", results[0].viol, results[1].viol);
    printf("Objective:       off=%.6e | on=%.6e\n", results[0].obj, results[1].obj);
    printf("HW time (ms):    off=%.3f | on=%.3f\n\n", results[0].hw_ms, results[1].hw_ms);

    munmap(ip_base, MAP_SIZE); close(mem_fd);
    cma.free_all();
    printf("Buffers released.\n");
    return 0;
}