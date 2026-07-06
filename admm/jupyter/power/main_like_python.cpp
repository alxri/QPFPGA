#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <vector>

extern "C" {
void *cma_alloc(uint32_t len, uint32_t cacheable);
unsigned long cma_get_phy_addr(void *buf);
void cma_free(void *buf);
}

// ---------------------------------------------------------------------
// Configuration & Register Map (must match the HLS kernel)
// ---------------------------------------------------------------------
#define PACK_SIZE 16
#define TILE_ROWS 8192
#define TILE_COLS 8192

#define ADMM_IP_CONTROL_BASE 0xA0000000
#define ADMM_IP_CONTROL_R_BASE 0xA0010000
#define MAP_SIZE 0x20000UL
#define MAP_MASK (MAP_SIZE - 1)

struct int32_words {
    int32_t data[PACK_SIZE];
};
struct float32_words {
    float data[PACK_SIZE];
};

struct RunResult {
    int admm_iters;
    int pcg_iters;
    float r_prim;
    float r_dual;
    float viol;
    double obj;
    double hw_ms;
};

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

static inline void write_reg(void *base, uint32_t offset, uint32_t val) {
    *((volatile uint32_t *)((uint8_t *)base + offset)) = val;
}
static inline uint32_t read_reg(void *base, uint32_t offset) {
    return *((volatile uint32_t *)((uint8_t *)base + offset));
}

static void write_64bit_address(void *base, uint32_t offset, uintptr_t address) {
    write_reg(base, offset, (uint32_t)(address & 0xFFFFFFFFu));
    write_reg(base, offset + 0x04, (uint32_t)((uint64_t)address >> 32));
}

static inline uint32_t float_to_uint(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
static inline float uint_to_float(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static size_t get_file_elements(const std::string &filename, size_t element_size) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) {
        std::cout << "ERROR: Could not open " << filename << std::endl;
        std::exit(1);
    }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fclose(f);
    return sz / element_size;
}

template <typename T>
static void load_bin(const std::string &filename, std::vector<T> &dest) {
    FILE *f = fopen(filename.c_str(), "rb");
    if (!f) {
        std::cout << "ERROR: Could not open " << filename << std::endl;
        std::exit(1);
    }
    size_t r = fread(dest.data(), sizeof(T), dest.size(), f);
    if (r != dest.size()) {
        std::cout << "WARNING: Read mismatch in " << filename << std::endl;
    }
    fclose(f);
}

static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------
// CMA Safe Handlers (avoid SIMD/NEON faults on some uncached mappings)
// ---------------------------------------------------------------------
template <typename T>
static void safe_cma_copy(T *dest, const T *src, size_t elements) {
    volatile uint32_t *d = (volatile uint32_t *)dest;
    const uint32_t *s = (const uint32_t *)src;
    size_t words = (elements * sizeof(T)) / 4;
    for (size_t i = 0; i < words; i++) {
        d[i] = s[i];
    }
}

template <typename T>
static void safe_cma_zero(T *dest, size_t elements) {
    volatile uint32_t *d = (volatile uint32_t *)dest;
    size_t words = (elements * sizeof(T)) / 4;
    for (size_t i = 0; i < words; i++) {
        d[i] = 0;
    }
}

template <typename T>
static T *alloc_cma(size_t elements) {
    // 0 = uncached mapping
    T *ptr = (T *)cma_alloc(elements * sizeof(T), 0);
    if (!ptr) {
        std::cout << "CMA allocation failed." << std::endl;
        std::exit(1);
    }
    safe_cma_zero(ptr, elements);
    return ptr;
}

// ---------------------------------------------------------------------
// Matrix helpers: transpose + tiling (matches notebook build_tiled_csc)
// ---------------------------------------------------------------------
struct TiledMatrix {
    int rtiles = 0;
    int ctiles = 0;
    std::vector<int> counts;
    std::vector<int> noff;
    std::vector<int> coff;
    std::vector<int> cptr;
    std::vector<int> ridx;   // flattened packed words (PACK_SIZE lanes)
    std::vector<float> vals; // flattened packed words
};

static void transpose_csc(int rows,
                          int cols,
                          const std::vector<int> &cptr,
                          const std::vector<int> &ridx,
                          const std::vector<float> &vals,
                          std::vector<int> &cptr_t,
                          std::vector<int> &ridx_t,
                          std::vector<float> &vals_t) {
    std::vector<int> nnz_per_col(rows, 0);
    for (size_t i = 0; i < ridx.size(); ++i) {
        nnz_per_col[ridx[i]]++;
    }

    cptr_t.assign(rows + 1, 0);
    for (int c = 0; c < rows; ++c) {
        cptr_t[c + 1] = cptr_t[c] + nnz_per_col[c];
    }

    std::vector<int> next = cptr_t;
    ridx_t.resize(ridx.size());
    vals_t.resize(vals.size());

    for (int c = 0; c < cols; ++c) {
        for (int idx = cptr[c]; idx < cptr[c + 1]; ++idx) {
            int dst = next[ridx[idx]]++;
            ridx_t[dst] = c;
            vals_t[dst] = vals[idx];
        }
    }
}

static TiledMatrix build_tiled_csc(int global_rows,
                                  int global_cols,
                                  const std::vector<int> &cptr_in,
                                  const std::vector<int> &ridx_in,
                                  const std::vector<float> &vals_in) {
    TiledMatrix tm;
    tm.rtiles = ceil_div(global_rows, TILE_ROWS);
    tm.ctiles = ceil_div(global_cols, TILE_COLS);
    int num_tiles = tm.rtiles * tm.ctiles;

    tm.counts.assign(num_tiles, 0);
    tm.noff.assign(num_tiles, 0);
    tm.coff.assign(num_tiles, 0);

    std::vector<std::vector<std::vector<int>>> t_rows(num_tiles, std::vector<std::vector<int>>(TILE_COLS));
    std::vector<std::vector<std::vector<float>>> t_vals(num_tiles, std::vector<std::vector<float>>(TILE_COLS));

    for (int c = 0; c < global_cols; ++c) {
        int tc = c / TILE_COLS;
        int local_c = c % TILE_COLS;
        for (int idx = cptr_in[c]; idx < cptr_in[c + 1]; ++idx) {
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
        tm.coff[tile_idx] = (int)tm.cptr.size();

        std::vector<int> local_cptr(TILE_COLS + 1, 0);
        int tile_nnz = 0;
        for (int c = 0; c < TILE_COLS; ++c) {
            tile_nnz += (int)t_rows[tile_idx][c].size();
            local_cptr[c + 1] = tile_nnz;
        }

        tm.counts[tile_idx] = tile_nnz;
        tm.noff[tile_idx] = nnz_word_cursor; // word offset
        tm.cptr.insert(tm.cptr.end(), local_cptr.begin(), local_cptr.end());

        int words = ceil_div(tile_nnz, PACK_SIZE);
        int flat_idx = 0;
        std::vector<int> flat_r(words * PACK_SIZE, 0);
        std::vector<float> flat_v(words * PACK_SIZE, 0.0f);

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

    if (tm.ridx.empty()) {
        tm.ridx.resize(PACK_SIZE, 0);
        tm.vals.resize(PACK_SIZE, 0.0f);
    }

    return tm;
}

int main() {
    // Parameters (match the notebook)
    const float sigma = 1e-2f;
    const float alpha = 1.8f;
    const float eps_abs = 5e-3f;
    const float eps_rel = 5e-3f;
    const float pcg_tol_fraction = 1.0f;
    const int admm_max_iter = 2000;
    const int pcg_max_iter = 5;

    const std::string ddir = "./data_bin/";

    int num_cols = (int)get_file_elements(ddir + "q.bin", sizeof(float));
    int num_rows = (int)get_file_elements(ddir + "l.bin", sizeof(float));

    std::vector<int> A_cptr(num_cols + 1);
    load_bin(ddir + "A_indptr.bin", A_cptr);
    int A_nnz = A_cptr.back();

    std::cout << "Loading QP data from: " << ddir << "\n";
    std::cout << "Problem size: " << num_rows << " x " << num_cols << "\n\n";

    std::vector<float> P_diag(num_cols), q(num_cols), l(num_rows), u(num_rows);
    std::vector<int> A_ridx(A_nnz);
    std::vector<float> A_vals(A_nnz);

    load_bin(ddir + "P_diag.bin", P_diag);
    load_bin(ddir + "q.bin", q);
    load_bin(ddir + "l.bin", l);
    load_bin(ddir + "u.bin", u);
    load_bin(ddir + "A_indices.bin", A_ridx);
    load_bin(ddir + "A_data.bin", A_vals);

    // Keep originals for objective/violation (matches notebook prints)
    const std::vector<float> P_diag_orig = P_diag;
    const std::vector<float> q_orig = q;
    const std::vector<float> l_orig = l;
    const std::vector<float> u_orig = u;
    const std::vector<float> A_vals_orig = A_vals;

    // -----------------------------------------------------------------
    // Scaling
    // -----------------------------------------------------------------
    std::cout << "Applying Scaling...\n\n";

    std::vector<float> E(num_cols, 1.0f);
    std::vector<float> D(num_rows, 1.0f);

    for (int iter = 0; iter < 10; ++iter) {
        std::vector<float> A_col_norm(num_cols, 0.0f);
        std::vector<float> A_row_norm(num_rows, 0.0f);

        for (int c = 0; c < num_cols; ++c) {
            for (int idx = A_cptr[c]; idx < A_cptr[c + 1]; ++idx) {
                float val = std::abs(A_vals[idx]);
                A_col_norm[c] = std::max(A_col_norm[c], val);
                A_row_norm[A_ridx[idx]] = std::max(A_row_norm[A_ridx[idx]], val);
            }
        }

        std::vector<float> E_new(num_cols);
        std::vector<float> D_new(num_rows);

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
            for (int idx = A_cptr[c]; idx < A_cptr[c + 1]; ++idx) {
                A_vals[idx] *= (D_new[A_ridx[idx]] * E_new[c]);
            }
        }
    }

    float max_val = 1e-15f;
    for (int c = 0; c < num_cols; ++c) {
        q[c] *= E[c];
        max_val = std::max(max_val, std::max(std::abs(P_diag[c]), std::abs(q[c])));
    }
    float c_scale = std::max(1.0f / max_val, 1e-4f);

    for (int c = 0; c < num_cols; ++c) {
        P_diag[c] *= c_scale;
        q[c] *= c_scale;
    }
    for (int r = 0; r < num_rows; ++r) {
        l[r] = std::isinf(l[r]) ? l[r] : l[r] * D[r];
        u[r] = std::isinf(u[r]) ? u[r] : u[r] * D[r];
    }

    // rho init (matches notebook)
    std::vector<float> rho(num_rows, 1.0f);
    for (int r = 0; r < num_rows; ++r) {
        bool fin_l = !std::isinf(l[r]);
        bool fin_u = !std::isinf(u[r]);
        if (!fin_l && !fin_u) {
            rho[r] = 1e-6f;
        } else if (fin_l && fin_u && (u[r] - l[r] < 0.01f)) {
            rho[r] = 100.0f;
        }
    }

    // -----------------------------------------------------------------
    // Build AT and diagonal-P CSC (matches notebook)
    // -----------------------------------------------------------------
    std::vector<int> AT_cptr, AT_ridx;
    std::vector<float> AT_vals;
    transpose_csc(num_rows, num_cols, A_cptr, A_ridx, A_vals, AT_cptr, AT_ridx, AT_vals);

    std::vector<int> P_cptr(num_cols + 1, 0);
    std::vector<int> P_ridx(num_cols);
    std::vector<float> P_vals(num_cols);
    for (int c = 0; c < num_cols; ++c) {
        P_cptr[c + 1] = c + 1;
        P_ridx[c] = c;
        P_vals[c] = P_diag[c];
    }

    // -----------------------------------------------------------------
    // Tiling (matches notebook build_tiled_csc)
    // -----------------------------------------------------------------
    std::cout << "Slicing and Allocating Tiled Matrices... (This may take a moment)\n";
    std::cout << "  Tiling Matrix A...\n";
    TiledMatrix tm_A = build_tiled_csc(num_rows, num_cols, A_cptr, A_ridx, A_vals);
    std::cout << "  Tiling Matrix AT...\n";
    TiledMatrix tm_AT = build_tiled_csc(num_cols, num_rows, AT_cptr, AT_ridx, AT_vals);
    std::cout << "  Tiling Matrix P...\n\n";
    TiledMatrix tm_P = build_tiled_csc(num_cols, num_cols, P_cptr, P_ridx, P_vals);

    // -----------------------------------------------------------------
    // CMA allocations + copies (matches notebook allocate(cacheable=False))
    // -----------------------------------------------------------------
    int A_words = ceil_div(A_nnz, PACK_SIZE);

    int32_words *A_row_idx_reg_hw = alloc_cma<int32_words>(A_words);
    float32_words *A_val_reg_hw = alloc_cma<float32_words>(A_words);
    int *A_col_ptr_reg_hw = alloc_cma<int>(num_cols + 1);

    std::vector<int32_words> A_row_words(A_words);
    std::vector<float32_words> A_val_words(A_words);
    for (int i = 0; i < A_words; ++i) {
        for (int lane = 0; lane < PACK_SIZE; ++lane) {
            A_row_words[i].data[lane] = 0;
            A_val_words[i].data[lane] = 0.0f;
        }
    }
    for (int i = 0; i < A_nnz; ++i) {
        A_row_words[i / PACK_SIZE].data[i % PACK_SIZE] = A_ridx[i];
        A_val_words[i / PACK_SIZE].data[i % PACK_SIZE] = A_vals[i];
    }

    safe_cma_copy(A_row_idx_reg_hw, A_row_words.data(), (size_t)A_words);
    safe_cma_copy(A_val_reg_hw, A_val_words.data(), (size_t)A_words);
    safe_cma_copy(A_col_ptr_reg_hw, A_cptr.data(), (size_t)(num_cols + 1));

#define ALLOC_TILE_CMA(mat, tm)                                                                                           \
    int *mat##_tile_nnz_counts_hw = alloc_cma<int>((tm).counts.size());                                                   \
    int *mat##_tile_nnz_offsets_hw = alloc_cma<int>((tm).noff.size());                                                    \
    int *mat##_tile_col_offsets_hw = alloc_cma<int>((tm).coff.size());                                                    \
    int *mat##_col_ptr_tiled_hw = alloc_cma<int>((tm).cptr.size());                                                       \
    int32_words *mat##_row_idx_tiled_hw = alloc_cma<int32_words>(((tm).ridx.size() / PACK_SIZE));                         \
    float32_words *mat##_values_tiled_hw = alloc_cma<float32_words>(((tm).vals.size() / PACK_SIZE));                      \
    safe_cma_copy(mat##_tile_nnz_counts_hw, (tm).counts.data(), (tm).counts.size());                                      \
    safe_cma_copy(mat##_tile_nnz_offsets_hw, (tm).noff.data(), (tm).noff.size());                                         \
    safe_cma_copy(mat##_tile_col_offsets_hw, (tm).coff.data(), (tm).coff.size());                                         \
    safe_cma_copy(mat##_col_ptr_tiled_hw, (tm).cptr.data(), (tm).cptr.size());                                            \
    safe_cma_copy(mat##_row_idx_tiled_hw, (const int32_words *)(tm).ridx.data(), ((tm).ridx.size() / PACK_SIZE));         \
    safe_cma_copy(mat##_values_tiled_hw, (const float32_words *)(tm).vals.data(), ((tm).vals.size() / PACK_SIZE));

    ALLOC_TILE_CMA(A, tm_A);
    ALLOC_TILE_CMA(AT, tm_AT);
    ALLOC_TILE_CMA(P, tm_P);

    float *P_diag_hw = alloc_cma<float>(num_cols);
    float *l_in_hw = alloc_cma<float>(num_rows);
    float *u_in_hw = alloc_cma<float>(num_rows);
    float *q_in_hw = alloc_cma<float>(num_cols);
    float *rho_in_hw = alloc_cma<float>(num_rows);

    float *x_out_hw = alloc_cma<float>(num_cols);
    float *y_out_hw = alloc_cma<float>(num_rows);

    safe_cma_copy(P_diag_hw, P_diag.data(), (size_t)num_cols);
    safe_cma_copy(l_in_hw, l.data(), (size_t)num_rows);
    safe_cma_copy(u_in_hw, u.data(), (size_t)num_rows);
    safe_cma_copy(q_in_hw, q.data(), (size_t)num_cols);
    safe_cma_copy(rho_in_hw, rho.data(), (size_t)num_rows);

    // -----------------------------------------------------------------
    // MMIO + run loop (matches notebook run_hw)
    // -----------------------------------------------------------------
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        std::perror("open(/dev/mem)");
        return 1;
    }

    void *ip_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, ADMM_IP_CONTROL_BASE & ~MAP_MASK);
    if (ip_base == MAP_FAILED) {
        std::perror("mmap");
        close(mem_fd);
        return 1;
    }

    void *ctrl = (void *)((uint8_t *)ip_base + (ADMM_IP_CONTROL_BASE & MAP_MASK));
    void *ctrl_r = (void *)((uint8_t *)ip_base + (ADMM_IP_CONTROL_R_BASE & MAP_MASK));

    RunResult results[2];

    for (int adaptive_rho = 0; adaptive_rho <= 1; ++adaptive_rho) {
        std::printf("=== HW Run (adaptive_rho=%d) ===\n", adaptive_rho);

        // Reset outputs (matches notebook)
        safe_cma_zero(x_out_hw, (size_t)num_cols);
        safe_cma_zero(y_out_hw, (size_t)num_rows);

        // Scalars
        write_reg(ctrl, 0x10, (uint32_t)num_rows);
        write_reg(ctrl, 0x18, (uint32_t)num_cols);
        write_reg(ctrl, 0x20, (uint32_t)A_nnz);
        write_reg(ctrl, 0x28, (uint32_t)tm_A.rtiles);
        write_reg(ctrl, 0x30, (uint32_t)tm_A.ctiles);
        write_reg(ctrl, 0x38, (uint32_t)tm_AT.rtiles);
        write_reg(ctrl, 0x40, (uint32_t)tm_AT.ctiles);
        write_reg(ctrl, 0x48, (uint32_t)tm_P.rtiles);
        write_reg(ctrl, 0x50, (uint32_t)tm_P.ctiles);

        write_reg(ctrl, 0x58, float_to_uint(sigma));
        write_reg(ctrl, 0x60, float_to_uint(alpha));
        write_reg(ctrl, 0x68, (uint32_t)admm_max_iter);
        write_reg(ctrl, 0x70, (uint32_t)pcg_max_iter);
        write_reg(ctrl, 0x78, (uint32_t)adaptive_rho);
        write_reg(ctrl, 0x80, float_to_uint(eps_abs));
        write_reg(ctrl, 0x88, float_to_uint(eps_rel));
        write_reg(ctrl, 0x90, float_to_uint(pcg_tol_fraction));

        // Addresses (matches notebook)
        write_64bit_address(ctrl_r, 0x010, cma_get_phy_addr(A_row_idx_reg_hw));
        write_64bit_address(ctrl_r, 0x01c, cma_get_phy_addr(A_col_ptr_reg_hw));
        write_64bit_address(ctrl_r, 0x028, cma_get_phy_addr(A_val_reg_hw));

        write_64bit_address(ctrl_r, 0x034, cma_get_phy_addr(A_tile_nnz_counts_hw));
        write_64bit_address(ctrl_r, 0x040, cma_get_phy_addr(A_tile_nnz_offsets_hw));
        write_64bit_address(ctrl_r, 0x04c, cma_get_phy_addr(A_tile_col_offsets_hw));
        write_64bit_address(ctrl_r, 0x058, cma_get_phy_addr(A_row_idx_tiled_hw));
        write_64bit_address(ctrl_r, 0x064, cma_get_phy_addr(A_col_ptr_tiled_hw));
        write_64bit_address(ctrl_r, 0x070, cma_get_phy_addr(A_values_tiled_hw));

        write_64bit_address(ctrl_r, 0x07c, cma_get_phy_addr(AT_tile_nnz_counts_hw));
        write_64bit_address(ctrl_r, 0x088, cma_get_phy_addr(AT_tile_nnz_offsets_hw));
        write_64bit_address(ctrl_r, 0x094, cma_get_phy_addr(AT_tile_col_offsets_hw));
        write_64bit_address(ctrl_r, 0x0a0, cma_get_phy_addr(AT_row_idx_tiled_hw));
        write_64bit_address(ctrl_r, 0x0ac, cma_get_phy_addr(AT_col_ptr_tiled_hw));
        write_64bit_address(ctrl_r, 0x0b8, cma_get_phy_addr(AT_values_tiled_hw));

        write_64bit_address(ctrl_r, 0x0c4, cma_get_phy_addr(P_tile_nnz_counts_hw));
        write_64bit_address(ctrl_r, 0x0d0, cma_get_phy_addr(P_tile_nnz_offsets_hw));
        write_64bit_address(ctrl_r, 0x0dc, cma_get_phy_addr(P_tile_col_offsets_hw));
        write_64bit_address(ctrl_r, 0x0e8, cma_get_phy_addr(P_row_idx_tiled_hw));
        write_64bit_address(ctrl_r, 0x0f4, cma_get_phy_addr(P_col_ptr_tiled_hw));
        write_64bit_address(ctrl_r, 0x100, cma_get_phy_addr(P_values_tiled_hw));

        write_64bit_address(ctrl_r, 0x10c, cma_get_phy_addr(P_diag_hw));
        write_64bit_address(ctrl_r, 0x118, cma_get_phy_addr(l_in_hw));
        write_64bit_address(ctrl_r, 0x124, cma_get_phy_addr(u_in_hw));
        write_64bit_address(ctrl_r, 0x130, cma_get_phy_addr(q_in_hw));
        write_64bit_address(ctrl_r, 0x13c, cma_get_phy_addr(rho_in_hw));
        write_64bit_address(ctrl_r, 0x148, cma_get_phy_addr(x_out_hw));
        write_64bit_address(ctrl_r, 0x154, cma_get_phy_addr(y_out_hw));

        // Run
        double hw_start = get_time_ms();
        write_reg(ctrl, 0x00, 0x01);
        while ((read_reg(ctrl, 0x00) & 0x02u) == 0) {
        }
        double hw_end = get_time_ms();

        int admm_iters = (int)read_reg(ctrl, 0x98);
        int pcg_iters = (int)read_reg(ctrl, 0xa8);
        float r_prim = uint_to_float(read_reg(ctrl, 0xb8));
        float r_dual = uint_to_float(read_reg(ctrl, 0xc8));
        int status = (int)read_reg(ctrl, 0xd8);

        // Unscale x + objective/violation prints (matches notebook)
        std::vector<float> x_unscaled(num_cols);
        for (int i = 0; i < num_cols; ++i) {
            x_unscaled[i] = x_out_hw[i] * E[i];
        }

        double obj = 0.0;
        for (int i = 0; i < num_cols; ++i) {
            obj += 0.5 * (double)P_diag_orig[i] * (double)x_unscaled[i] * (double)x_unscaled[i] + (double)q_orig[i] * (double)x_unscaled[i];
        }

        std::vector<float> Ax(num_rows, 0.0f);
        for (int c = 0; c < num_cols; ++c) {
            for (int idx = A_cptr[c]; idx < A_cptr[c + 1]; ++idx) {
                Ax[A_ridx[idx]] += A_vals_orig[idx] * x_unscaled[c];
            }
        }

        float max_viol = 0.0f;
        for (int r = 0; r < num_rows; ++r) {
            float viol_l = l_orig[r] - Ax[r];
            float viol_u = Ax[r] - u_orig[r];
            float v = std::max(0.0f, std::max(viol_l, viol_u));
            max_viol = std::max(max_viol, v);
        }

        float avg_pcg = (admm_iters > 0) ? ((float)pcg_iters / (float)admm_iters) : 0.0f;

        results[adaptive_rho] = {admm_iters, pcg_iters, r_prim, r_dual, max_viol, obj, (hw_end - hw_start)};

        std::printf("HW execution time: %.4f ms\n", hw_end - hw_start);
        std::printf("Status: %s\n", status == 1 ? "Converged" : "Max Iterations");
        std::printf("ADMM Iterations: %d\n", admm_iters);
        std::printf("PCG Iterations : %d (Average pcg/admm: %.1f)\n", pcg_iters, avg_pcg);
        std::printf("Primal Residual: %.5e\n", r_prim);
        std::printf("Dual Residual  : %.5e\n", r_dual);
        std::printf("Objective Value: %.6e\n", obj);
        std::printf("Max Violation  : %.3e\n\n", max_viol);
    }

    std::printf("=== Summary ===\n");
    std::printf("ADMM iterations: off=%d | on=%d\n", results[0].admm_iters, results[1].admm_iters);
    std::printf("PCG iterations:  off=%d | on=%d\n", results[0].pcg_iters, results[1].pcg_iters);
    std::printf("Primal residual: off=%.3e | on=%.3e\n", results[0].r_prim, results[1].r_prim);
    std::printf("Dual residual:   off=%.3e | on=%.3e\n", results[0].r_dual, results[1].r_dual);
    std::printf("Violation:       off=%.3e | on=%.3e\n", results[0].viol, results[1].viol);
    std::printf("Objective:       off=%.6e | on=%.6e\n", results[0].obj, results[1].obj);
    std::printf("HW time (ms):    off=%.3f | on=%.3f\n\n", results[0].hw_ms, results[1].hw_ms);

    munmap(ip_base, MAP_SIZE);
    close(mem_fd);

    // -----------------------------------------------------------------
    // Cleanup CMA
    // -----------------------------------------------------------------
    cma_free(A_row_idx_reg_hw);
    cma_free(A_val_reg_hw);
    cma_free(A_col_ptr_reg_hw);

    cma_free(A_tile_nnz_counts_hw);
    cma_free(A_tile_nnz_offsets_hw);
    cma_free(A_tile_col_offsets_hw);
    cma_free(A_col_ptr_tiled_hw);
    cma_free(A_row_idx_tiled_hw);
    cma_free(A_values_tiled_hw);

    cma_free(AT_tile_nnz_counts_hw);
    cma_free(AT_tile_nnz_offsets_hw);
    cma_free(AT_tile_col_offsets_hw);
    cma_free(AT_col_ptr_tiled_hw);
    cma_free(AT_row_idx_tiled_hw);
    cma_free(AT_values_tiled_hw);

    cma_free(P_tile_nnz_counts_hw);
    cma_free(P_tile_nnz_offsets_hw);
    cma_free(P_tile_col_offsets_hw);
    cma_free(P_col_ptr_tiled_hw);
    cma_free(P_row_idx_tiled_hw);
    cma_free(P_values_tiled_hw);

    cma_free(P_diag_hw);
    cma_free(l_in_hw);
    cma_free(u_in_hw);
    cma_free(q_in_hw);
    cma_free(rho_in_hw);
    cma_free(x_out_hw);
    cma_free(y_out_hw);

    std::printf("Buffers released.\n");
    return 0;
}
