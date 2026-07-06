#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

// PYNQ libcma prototypes
void *cma_alloc(uint32_t len, uint32_t cacheable);
unsigned long cma_get_phy_addr(void *buf);
void cma_free(void *buf);

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

typedef struct {
    int32_t data[PACK_SIZE];
} int32_words;

typedef struct {
    float data[PACK_SIZE];
} float32_words;

typedef struct {
    int admm_iters;
    int pcg_iters;
    float r_prim;
    float r_dual;
    float viol;
    double obj;
    double hw_ms;
} RunResult;

static double get_time_ms(void) {
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
    memcpy(&u, &f, 4);
    return u;
}

static inline float uint_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

static size_t get_file_elements(const char *filename, size_t element_size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Could not open %s\n", filename);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fclose(f);
    if (sz < 0) {
        fprintf(stderr, "ERROR: ftell failed for %s\n", filename);
        exit(1);
    }
    return (size_t)sz / element_size;
}

static void load_f32(const char *filename, float *dest, size_t count) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Could not open %s\n", filename);
        exit(1);
    }
    size_t r = fread(dest, sizeof(float), count, f);
    if (r != count) {
        fprintf(stderr, "WARNING: Read mismatch in %s (expected %zu, got %zu)\n", filename, count, r);
    }
    fclose(f);
}

static void load_i32(const char *filename, int32_t *dest, size_t count) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Could not open %s\n", filename);
        exit(1);
    }
    size_t r = fread(dest, sizeof(int32_t), count, f);
    if (r != count) {
        fprintf(stderr, "WARNING: Read mismatch in %s (expected %zu, got %zu)\n", filename, count, r);
    }
    fclose(f);
}

// ---------------------------------------------------------------------
// CMA Safe Handlers (avoid SIMD/NEON faults on some uncached mappings)
// ---------------------------------------------------------------------
static void safe_cma_copy_bytes(void *dest, const void *src, size_t bytes) {
    volatile uint32_t *d = (volatile uint32_t *)dest;
    const uint32_t *s = (const uint32_t *)src;
    const size_t words = bytes / 4;
    for (size_t i = 0; i < words; ++i) {
        d[i] = s[i];
    }
}

static void safe_cma_zero_bytes(void *dest, size_t bytes) {
    volatile uint32_t *d = (volatile uint32_t *)dest;
    const size_t words = bytes / 4;
    for (size_t i = 0; i < words; ++i) {
        d[i] = 0;
    }
}

static void *alloc_cma_bytes(size_t bytes) {
    void *ptr = cma_alloc((uint32_t)bytes, 0);
    if (!ptr) {
        fprintf(stderr, "CMA allocation failed for %zu bytes\n", bytes);
        exit(1);
    }
    safe_cma_zero_bytes(ptr, bytes);
    return ptr;
}

// ---------------------------------------------------------------------
// Matrix helpers: transpose + tiling (matches notebook build_tiled_csc)
// ---------------------------------------------------------------------
typedef struct {
    int rtiles;
    int ctiles;
    int num_tiles;
    int total_words;

    int32_t *counts;
    int32_t *noff;
    int32_t *coff;
    int32_t *cptr; // concatenated col_ptr arrays for all tiles

    int32_words *ridx_words;
    float32_words *vals_words;
} TiledMatrix;

static void free_tiled_matrix(TiledMatrix *tm) {
    if (!tm) {
        return;
    }
    free(tm->counts);
    free(tm->noff);
    free(tm->coff);
    free(tm->cptr);
    free(tm->ridx_words);
    free(tm->vals_words);
    memset(tm, 0, sizeof(*tm));
}

static void transpose_csc(int rows,
                          int cols,
                          const int32_t *cptr,
                          const int32_t *ridx,
                          const float *vals,
                          int32_t *cptr_t,
                          int32_t *ridx_t,
                          float *vals_t) {
    // AT will have 'rows' columns.
    int32_t *nnz_per_col = (int32_t *)calloc((size_t)rows, sizeof(int32_t));
    if (!nnz_per_col) {
        fprintf(stderr, "ERROR: calloc(nnz_per_col) failed\n");
        exit(1);
    }

    const int nnz = (int)cptr[cols];
    for (int i = 0; i < nnz; ++i) {
        int r = (int)ridx[i];
        nnz_per_col[r]++;
    }

    cptr_t[0] = 0;
    for (int c = 0; c < rows; ++c) {
        cptr_t[c + 1] = cptr_t[c] + nnz_per_col[c];
    }

    int32_t *next = (int32_t *)malloc((size_t)rows * sizeof(int32_t));
    if (!next) {
        fprintf(stderr, "ERROR: malloc(next) failed\n");
        exit(1);
    }
    memcpy(next, cptr_t, (size_t)rows * sizeof(int32_t));

    for (int c = 0; c < cols; ++c) {
        for (int idx = (int)cptr[c]; idx < (int)cptr[c + 1]; ++idx) {
            int r = (int)ridx[idx];
            int dst = (int)next[r]++;
            ridx_t[dst] = (int32_t)c;
            vals_t[dst] = vals[idx];
        }
    }

    free(nnz_per_col);
    free(next);
}

static TiledMatrix build_tiled_csc(int global_rows,
                                  int global_cols,
                                  const int32_t *cptr_in,
                                  const int32_t *ridx_in,
                                  const float *vals_in) {
    TiledMatrix tm;
    memset(&tm, 0, sizeof(tm));

    tm.rtiles = ceil_div_int(global_rows, TILE_ROWS);
    tm.ctiles = ceil_div_int(global_cols, TILE_COLS);
    tm.num_tiles = tm.rtiles * tm.ctiles;

    const size_t num_tiles = (size_t)tm.num_tiles;
    const size_t cptr_len = num_tiles * (size_t)(TILE_COLS + 1);
    const size_t col_counts_len = num_tiles * (size_t)TILE_COLS;

    tm.counts = (int32_t *)calloc(num_tiles, sizeof(int32_t));
    tm.noff = (int32_t *)calloc(num_tiles, sizeof(int32_t));
    tm.coff = (int32_t *)calloc(num_tiles, sizeof(int32_t));
    tm.cptr = (int32_t *)calloc(cptr_len, sizeof(int32_t));

    if (!tm.counts || !tm.noff || !tm.coff || !tm.cptr) {
        fprintf(stderr, "ERROR: Out of memory allocating tile metadata\n");
        exit(1);
    }

    // Pass 1: count nnz per (tile, local_col)
    int32_t *col_counts = (int32_t *)calloc(col_counts_len, sizeof(int32_t));
    if (!col_counts) {
        fprintf(stderr, "ERROR: Out of memory allocating col_counts\n");
        exit(1);
    }

    for (int c = 0; c < global_cols; ++c) {
        const int tc = c / TILE_COLS;
        const int local_c = c % TILE_COLS;
        for (int idx = (int)cptr_in[c]; idx < (int)cptr_in[c + 1]; ++idx) {
            const int r = (int)ridx_in[idx];
            const int tr = r / TILE_ROWS;
            const int tile_idx = tr * tm.ctiles + tc;
            col_counts[(size_t)tile_idx * (size_t)TILE_COLS + (size_t)local_c]++;
        }
    }

    // Build col_ptr arrays + compute per-tile offsets
    int word_cursor = 0;
    for (int tile_idx = 0; tile_idx < tm.num_tiles; ++tile_idx) {
        tm.coff[tile_idx] = (int32_t)(tile_idx * (TILE_COLS + 1));
        const size_t base = (size_t)tm.coff[tile_idx];

        tm.cptr[base + 0] = 0;
        for (int lc = 0; lc < TILE_COLS; ++lc) {
            const int32_t prev = tm.cptr[base + (size_t)lc];
            const int32_t add = col_counts[(size_t)tile_idx * (size_t)TILE_COLS + (size_t)lc];
            tm.cptr[base + (size_t)lc + 1] = prev + add;
        }

        const int tile_nnz = (int)tm.cptr[base + (size_t)TILE_COLS];
        tm.counts[tile_idx] = (int32_t)tile_nnz;
        tm.noff[tile_idx] = (int32_t)word_cursor;

        word_cursor += ceil_div_int(tile_nnz, PACK_SIZE);
    }

    tm.total_words = word_cursor;
    if (tm.total_words <= 0) {
        // Keep a non-null buffer for the (rare) all-zero matrix case.
        tm.total_words = 1;
    }

    tm.ridx_words = (int32_words *)calloc((size_t)tm.total_words, sizeof(int32_words));
    tm.vals_words = (float32_words *)calloc((size_t)tm.total_words, sizeof(float32_words));
    if (!tm.ridx_words || !tm.vals_words) {
        fprintf(stderr, "ERROR: Out of memory allocating packed tile data\n");
        exit(1);
    }

    // Initialize next positions for each (tile, local_col) from col_ptr
    int32_t *next_pos = (int32_t *)malloc(col_counts_len * sizeof(int32_t));
    if (!next_pos) {
        fprintf(stderr, "ERROR: Out of memory allocating next_pos\n");
        exit(1);
    }

    for (int tile_idx = 0; tile_idx < tm.num_tiles; ++tile_idx) {
        const size_t base = (size_t)tm.coff[tile_idx];
        const size_t row_base = (size_t)tile_idx * (size_t)TILE_COLS;
        for (int lc = 0; lc < TILE_COLS; ++lc) {
            next_pos[row_base + (size_t)lc] = tm.cptr[base + (size_t)lc];
        }
    }

    // Pass 2: fill packed ridx/vals arrays directly
    for (int c = 0; c < global_cols; ++c) {
        const int tc = c / TILE_COLS;
        const int local_c = c % TILE_COLS;

        for (int idx = (int)cptr_in[c]; idx < (int)cptr_in[c + 1]; ++idx) {
            const int r = (int)ridx_in[idx];
            const float v = vals_in[idx];

            const int tr = r / TILE_ROWS;
            const int tile_idx = tr * tm.ctiles + tc;
            const int local_r = r % TILE_ROWS;

            const size_t np_index = (size_t)tile_idx * (size_t)TILE_COLS + (size_t)local_c;
            const int32_t pos = next_pos[np_index]++;

            const int word = (int)tm.noff[tile_idx] + ((int)pos / PACK_SIZE);
            const int lane = (int)pos % PACK_SIZE;

            tm.ridx_words[word].data[lane] = (int32_t)local_r;
            tm.vals_words[word].data[lane] = v;
        }
    }

    free(col_counts);
    free(next_pos);

    return tm;
}

typedef struct {
    int32_t *tile_nnz_counts_hw;
    int32_t *tile_nnz_offsets_hw;
    int32_t *tile_col_offsets_hw;
    int32_t *col_ptr_tiled_hw;
    int32_words *row_idx_tiled_hw;
    float32_words *values_tiled_hw;
} HwTiled;

static HwTiled alloc_and_copy_tiled(const TiledMatrix *tm) {
    HwTiled hw;
    memset(&hw, 0, sizeof(hw));

    const size_t num_tiles = (size_t)tm->num_tiles;
    const size_t counts_bytes = num_tiles * sizeof(int32_t);
    const size_t cptr_bytes = (size_t)tm->num_tiles * (size_t)(TILE_COLS + 1) * sizeof(int32_t);
    const size_t words_bytes = (size_t)tm->total_words * sizeof(int32_words);
    const size_t vals_bytes = (size_t)tm->total_words * sizeof(float32_words);

    hw.tile_nnz_counts_hw = (int32_t *)alloc_cma_bytes(counts_bytes);
    hw.tile_nnz_offsets_hw = (int32_t *)alloc_cma_bytes(counts_bytes);
    hw.tile_col_offsets_hw = (int32_t *)alloc_cma_bytes(counts_bytes);
    hw.col_ptr_tiled_hw = (int32_t *)alloc_cma_bytes(cptr_bytes);
    hw.row_idx_tiled_hw = (int32_words *)alloc_cma_bytes(words_bytes);
    hw.values_tiled_hw = (float32_words *)alloc_cma_bytes(vals_bytes);

    safe_cma_copy_bytes(hw.tile_nnz_counts_hw, tm->counts, counts_bytes);
    safe_cma_copy_bytes(hw.tile_nnz_offsets_hw, tm->noff, counts_bytes);
    safe_cma_copy_bytes(hw.tile_col_offsets_hw, tm->coff, counts_bytes);
    safe_cma_copy_bytes(hw.col_ptr_tiled_hw, tm->cptr, cptr_bytes);
    safe_cma_copy_bytes(hw.row_idx_tiled_hw, tm->ridx_words, words_bytes);
    safe_cma_copy_bytes(hw.values_tiled_hw, tm->vals_words, vals_bytes);

    return hw;
}

static void free_hw_tiled(HwTiled *hw) {
    if (!hw) {
        return;
    }
    if (hw->tile_nnz_counts_hw)
        cma_free(hw->tile_nnz_counts_hw);
    if (hw->tile_nnz_offsets_hw)
        cma_free(hw->tile_nnz_offsets_hw);
    if (hw->tile_col_offsets_hw)
        cma_free(hw->tile_col_offsets_hw);
    if (hw->col_ptr_tiled_hw)
        cma_free(hw->col_ptr_tiled_hw);
    if (hw->row_idx_tiled_hw)
        cma_free(hw->row_idx_tiled_hw);
    if (hw->values_tiled_hw)
        cma_free(hw->values_tiled_hw);
    memset(hw, 0, sizeof(*hw));
}

int main(void) {
    // Parameters (match the notebook)
    const float sigma = 1e-2f;
    const float alpha = 1.8f;
    const float eps_abs = 5e-3f;
    const float eps_rel = 5e-3f;
    const float pcg_tol_fraction = 1.0f;
    const int admm_max_iter = 2000;
    const int pcg_max_iter = 5;

    const char *ddir = "./data_bin/";

    char path[512];

    snprintf(path, sizeof(path), "%sq.bin", ddir);
    const int num_cols = (int)get_file_elements(path, sizeof(float));
    snprintf(path, sizeof(path), "%sl.bin", ddir);
    const int num_rows = (int)get_file_elements(path, sizeof(float));

    // A_indptr
    int32_t *A_cptr = (int32_t *)malloc((size_t)(num_cols + 1) * sizeof(int32_t));
    if (!A_cptr) {
        fprintf(stderr, "ERROR: malloc(A_cptr) failed\n");
        return 1;
    }

    snprintf(path, sizeof(path), "%sA_indptr.bin", ddir);
    load_i32(path, A_cptr, (size_t)(num_cols + 1));
    const int A_nnz = (int)A_cptr[num_cols];

    printf("Loading QP data from: %s\n", ddir);
    printf("Problem size: %d x %d\n\n", num_rows, num_cols);

    float *P_diag = (float *)malloc((size_t)num_cols * sizeof(float));
    float *q = (float *)malloc((size_t)num_cols * sizeof(float));
    float *l = (float *)malloc((size_t)num_rows * sizeof(float));
    float *u = (float *)malloc((size_t)num_rows * sizeof(float));
    int32_t *A_ridx = (int32_t *)malloc((size_t)A_nnz * sizeof(int32_t));
    float *A_vals = (float *)malloc((size_t)A_nnz * sizeof(float));

    if (!P_diag || !q || !l || !u || !A_ridx || !A_vals) {
        fprintf(stderr, "ERROR: malloc(QP arrays) failed\n");
        return 1;
    }

    snprintf(path, sizeof(path), "%sP_diag.bin", ddir);
    load_f32(path, P_diag, (size_t)num_cols);
    snprintf(path, sizeof(path), "%sq.bin", ddir);
    load_f32(path, q, (size_t)num_cols);
    snprintf(path, sizeof(path), "%sl.bin", ddir);
    load_f32(path, l, (size_t)num_rows);
    snprintf(path, sizeof(path), "%su.bin", ddir);
    load_f32(path, u, (size_t)num_rows);
    snprintf(path, sizeof(path), "%sA_indices.bin", ddir);
    load_i32(path, A_ridx, (size_t)A_nnz);
    snprintf(path, sizeof(path), "%sA_data.bin", ddir);
    load_f32(path, A_vals, (size_t)A_nnz);

    // Keep originals for objective/violation (matches notebook prints)
    float *P_diag_orig = (float *)malloc((size_t)num_cols * sizeof(float));
    float *q_orig = (float *)malloc((size_t)num_cols * sizeof(float));
    float *l_orig = (float *)malloc((size_t)num_rows * sizeof(float));
    float *u_orig = (float *)malloc((size_t)num_rows * sizeof(float));
    float *A_vals_orig = (float *)malloc((size_t)A_nnz * sizeof(float));

    if (!P_diag_orig || !q_orig || !l_orig || !u_orig || !A_vals_orig) {
        fprintf(stderr, "ERROR: malloc(original copies) failed\n");
        return 1;
    }

    memcpy(P_diag_orig, P_diag, (size_t)num_cols * sizeof(float));
    memcpy(q_orig, q, (size_t)num_cols * sizeof(float));
    memcpy(l_orig, l, (size_t)num_rows * sizeof(float));
    memcpy(u_orig, u, (size_t)num_rows * sizeof(float));
    memcpy(A_vals_orig, A_vals, (size_t)A_nnz * sizeof(float));

    // -----------------------------------------------------------------
    // Ruiz scaling (matches notebook apply_scaling)
    // -----------------------------------------------------------------
    printf("Applying Ruiz Equilibration...\n\n");

    float *E = (float *)malloc((size_t)num_cols * sizeof(float));
    float *D = (float *)malloc((size_t)num_rows * sizeof(float));
    float *A_col_norm = (float *)malloc((size_t)num_cols * sizeof(float));
    float *A_row_norm = (float *)malloc((size_t)num_rows * sizeof(float));
    float *E_new = (float *)malloc((size_t)num_cols * sizeof(float));
    float *D_new = (float *)malloc((size_t)num_rows * sizeof(float));

    if (!E || !D || !A_col_norm || !A_row_norm || !E_new || !D_new) {
        fprintf(stderr, "ERROR: malloc(Ruiz arrays) failed\n");
        return 1;
    }

    for (int c = 0; c < num_cols; ++c)
        E[c] = 1.0f;
    for (int r = 0; r < num_rows; ++r)
        D[r] = 1.0f;

    for (int iter = 0; iter < 10; ++iter) {
        for (int c = 0; c < num_cols; ++c)
            A_col_norm[c] = 0.0f;
        for (int r = 0; r < num_rows; ++r)
            A_row_norm[r] = 0.0f;

        for (int c = 0; c < num_cols; ++c) {
            for (int idx = (int)A_cptr[c]; idx < (int)A_cptr[c + 1]; ++idx) {
                const float val = fabsf(A_vals[idx]);
                if (val > A_col_norm[c])
                    A_col_norm[c] = val;
                const int rr = (int)A_ridx[idx];
                if (val > A_row_norm[rr])
                    A_row_norm[rr] = val;
            }
        }

        for (int c = 0; c < num_cols; ++c) {
            float x_norm = fabsf(P_diag[c]);
            if (A_col_norm[c] > x_norm)
                x_norm = A_col_norm[c];
            if (x_norm < 1e-4f)
                x_norm = 1e-4f;

            E_new[c] = 1.0f / sqrtf(x_norm);
            E[c] *= E_new[c];
            P_diag[c] *= (E_new[c] * E_new[c]);
        }

        for (int r = 0; r < num_rows; ++r) {
            float z_norm = A_row_norm[r];
            if (z_norm < 1e-4f)
                z_norm = 1e-4f;

            D_new[r] = 1.0f / sqrtf(z_norm);
            D[r] *= D_new[r];
        }

        for (int c = 0; c < num_cols; ++c) {
            for (int idx = (int)A_cptr[c]; idx < (int)A_cptr[c + 1]; ++idx) {
                const int rr = (int)A_ridx[idx];
                A_vals[idx] *= (D_new[rr] * E_new[c]);
            }
        }
    }

    float max_val = 1e-15f;
    for (int c = 0; c < num_cols; ++c) {
        q[c] *= E[c];
        const float a = fabsf(P_diag[c]);
        const float b = fabsf(q[c]);
        const float m = (a > b) ? a : b;
        if (m > max_val)
            max_val = m;
    }

    float c_scale = 1.0f / max_val;
    if (c_scale < 1e-4f)
        c_scale = 1e-4f;

    for (int c = 0; c < num_cols; ++c) {
        P_diag[c] *= c_scale;
        q[c] *= c_scale;
    }

    for (int r = 0; r < num_rows; ++r) {
        l[r] = isinf(l[r]) ? l[r] : (l[r] * D[r]);
        u[r] = isinf(u[r]) ? u[r] : (u[r] * D[r]);
    }

    // rho init (matches notebook)
    float *rho = (float *)malloc((size_t)num_rows * sizeof(float));
    if (!rho) {
        fprintf(stderr, "ERROR: malloc(rho) failed\n");
        return 1;
    }
    for (int r = 0; r < num_rows; ++r)
        rho[r] = 1.0f;

    for (int r = 0; r < num_rows; ++r) {
        const int fin_l = !isinf(l[r]);
        const int fin_u = !isinf(u[r]);
        if (!fin_l && !fin_u) {
            rho[r] = 1e-6f;
        } else if (fin_l && fin_u && ((u[r] - l[r]) < 0.01f)) {
            rho[r] = 100.0f;
        }
    }

    // -----------------------------------------------------------------
    // Build AT and diagonal-P CSC (matches notebook)
    // -----------------------------------------------------------------
    int32_t *AT_cptr = (int32_t *)malloc((size_t)(num_rows + 1) * sizeof(int32_t));
    int32_t *AT_ridx = (int32_t *)malloc((size_t)A_nnz * sizeof(int32_t));
    float *AT_vals = (float *)malloc((size_t)A_nnz * sizeof(float));

    if (!AT_cptr || !AT_ridx || !AT_vals) {
        fprintf(stderr, "ERROR: malloc(AT) failed\n");
        return 1;
    }

    transpose_csc(num_rows, num_cols, A_cptr, A_ridx, A_vals, AT_cptr, AT_ridx, AT_vals);

    int32_t *P_cptr = (int32_t *)malloc((size_t)(num_cols + 1) * sizeof(int32_t));
    int32_t *P_ridx = (int32_t *)malloc((size_t)num_cols * sizeof(int32_t));
    float *P_vals = (float *)malloc((size_t)num_cols * sizeof(float));

    if (!P_cptr || !P_ridx || !P_vals) {
        fprintf(stderr, "ERROR: malloc(P CSC) failed\n");
        return 1;
    }

    P_cptr[0] = 0;
    for (int c = 0; c < num_cols; ++c) {
        P_cptr[c + 1] = (int32_t)(c + 1);
        P_ridx[c] = (int32_t)c;
        P_vals[c] = P_diag[c];
    }

    // -----------------------------------------------------------------
    // Tiling (matches notebook build_tiled_csc)
    // -----------------------------------------------------------------
    printf("Slicing and Allocating Tiled Matrices... (This may take a moment)\n");
    printf("  Tiling Matrix A...\n");
    TiledMatrix tm_A = build_tiled_csc(num_rows, num_cols, A_cptr, A_ridx, A_vals);
    printf("  Tiling Matrix AT...\n");
    TiledMatrix tm_AT = build_tiled_csc(num_cols, num_rows, AT_cptr, AT_ridx, AT_vals);
    printf("  Tiling Matrix P...\n\n");
    TiledMatrix tm_P = build_tiled_csc(num_cols, num_cols, P_cptr, P_ridx, P_vals);

    // -----------------------------------------------------------------
    // CMA allocations + copies (matches notebook allocate(cacheable=False))
    // -----------------------------------------------------------------
    const int A_words = ceil_div_int(A_nnz, PACK_SIZE);

    int32_words *A_row_idx_reg_hw = (int32_words *)alloc_cma_bytes((size_t)A_words * sizeof(int32_words));
    float32_words *A_val_reg_hw = (float32_words *)alloc_cma_bytes((size_t)A_words * sizeof(float32_words));
    int32_t *A_col_ptr_reg_hw = (int32_t *)alloc_cma_bytes((size_t)(num_cols + 1) * sizeof(int32_t));

    int32_words *A_row_words = (int32_words *)calloc((size_t)A_words, sizeof(int32_words));
    float32_words *A_val_words = (float32_words *)calloc((size_t)A_words, sizeof(float32_words));
    if (!A_row_words || !A_val_words) {
        fprintf(stderr, "ERROR: calloc(A_words) failed\n");
        return 1;
    }

    for (int i = 0; i < A_nnz; ++i) {
        const int w = i / PACK_SIZE;
        const int lane = i % PACK_SIZE;
        A_row_words[w].data[lane] = A_ridx[i];
        A_val_words[w].data[lane] = A_vals[i];
    }

    safe_cma_copy_bytes(A_row_idx_reg_hw, A_row_words, (size_t)A_words * sizeof(int32_words));
    safe_cma_copy_bytes(A_val_reg_hw, A_val_words, (size_t)A_words * sizeof(float32_words));
    safe_cma_copy_bytes(A_col_ptr_reg_hw, A_cptr, (size_t)(num_cols + 1) * sizeof(int32_t));

    free(A_row_words);
    free(A_val_words);

    HwTiled hw_A = alloc_and_copy_tiled(&tm_A);
    HwTiled hw_AT = alloc_and_copy_tiled(&tm_AT);
    HwTiled hw_P = alloc_and_copy_tiled(&tm_P);

    float *P_diag_hw = (float *)alloc_cma_bytes((size_t)num_cols * sizeof(float));
    float *l_in_hw = (float *)alloc_cma_bytes((size_t)num_rows * sizeof(float));
    float *u_in_hw = (float *)alloc_cma_bytes((size_t)num_rows * sizeof(float));
    float *q_in_hw = (float *)alloc_cma_bytes((size_t)num_cols * sizeof(float));
    float *rho_in_hw = (float *)alloc_cma_bytes((size_t)num_rows * sizeof(float));

    float *x_out_hw = (float *)alloc_cma_bytes((size_t)num_cols * sizeof(float));
    float *y_out_hw = (float *)alloc_cma_bytes((size_t)num_rows * sizeof(float));

    safe_cma_copy_bytes(P_diag_hw, P_diag, (size_t)num_cols * sizeof(float));
    safe_cma_copy_bytes(l_in_hw, l, (size_t)num_rows * sizeof(float));
    safe_cma_copy_bytes(u_in_hw, u, (size_t)num_rows * sizeof(float));
    safe_cma_copy_bytes(q_in_hw, q, (size_t)num_cols * sizeof(float));
    safe_cma_copy_bytes(rho_in_hw, rho, (size_t)num_rows * sizeof(float));

    // -----------------------------------------------------------------
    // MMIO + run loop (matches notebook run_hw)
    // -----------------------------------------------------------------
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        fprintf(stderr, "open(/dev/mem) failed: %s\n", strerror(errno));
        return 1;
    }

    void *ip_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, ADMM_IP_CONTROL_BASE & ~MAP_MASK);
    if (ip_base == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        close(mem_fd);
        return 1;
    }

    void *ctrl = (void *)((uint8_t *)ip_base + (ADMM_IP_CONTROL_BASE & MAP_MASK));
    void *ctrl_r = (void *)((uint8_t *)ip_base + (ADMM_IP_CONTROL_R_BASE & MAP_MASK));

    RunResult results[2];

    for (int adaptive_rho = 0; adaptive_rho <= 1; ++adaptive_rho) {
        printf("=== HW Run (adaptive_rho=%d) ===\n", adaptive_rho);

        // Reset outputs (matches notebook)
        safe_cma_zero_bytes(x_out_hw, (size_t)num_cols * sizeof(float));
        safe_cma_zero_bytes(y_out_hw, (size_t)num_rows * sizeof(float));

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
        write_64bit_address(ctrl_r, 0x010, (uintptr_t)cma_get_phy_addr(A_row_idx_reg_hw));
        write_64bit_address(ctrl_r, 0x01c, (uintptr_t)cma_get_phy_addr(A_col_ptr_reg_hw));
        write_64bit_address(ctrl_r, 0x028, (uintptr_t)cma_get_phy_addr(A_val_reg_hw));

        write_64bit_address(ctrl_r, 0x034, (uintptr_t)cma_get_phy_addr(hw_A.tile_nnz_counts_hw));
        write_64bit_address(ctrl_r, 0x040, (uintptr_t)cma_get_phy_addr(hw_A.tile_nnz_offsets_hw));
        write_64bit_address(ctrl_r, 0x04c, (uintptr_t)cma_get_phy_addr(hw_A.tile_col_offsets_hw));
        write_64bit_address(ctrl_r, 0x058, (uintptr_t)cma_get_phy_addr(hw_A.row_idx_tiled_hw));
        write_64bit_address(ctrl_r, 0x064, (uintptr_t)cma_get_phy_addr(hw_A.col_ptr_tiled_hw));
        write_64bit_address(ctrl_r, 0x070, (uintptr_t)cma_get_phy_addr(hw_A.values_tiled_hw));

        write_64bit_address(ctrl_r, 0x07c, (uintptr_t)cma_get_phy_addr(hw_AT.tile_nnz_counts_hw));
        write_64bit_address(ctrl_r, 0x088, (uintptr_t)cma_get_phy_addr(hw_AT.tile_nnz_offsets_hw));
        write_64bit_address(ctrl_r, 0x094, (uintptr_t)cma_get_phy_addr(hw_AT.tile_col_offsets_hw));
        write_64bit_address(ctrl_r, 0x0a0, (uintptr_t)cma_get_phy_addr(hw_AT.row_idx_tiled_hw));
        write_64bit_address(ctrl_r, 0x0ac, (uintptr_t)cma_get_phy_addr(hw_AT.col_ptr_tiled_hw));
        write_64bit_address(ctrl_r, 0x0b8, (uintptr_t)cma_get_phy_addr(hw_AT.values_tiled_hw));

        write_64bit_address(ctrl_r, 0x0c4, (uintptr_t)cma_get_phy_addr(hw_P.tile_nnz_counts_hw));
        write_64bit_address(ctrl_r, 0x0d0, (uintptr_t)cma_get_phy_addr(hw_P.tile_nnz_offsets_hw));
        write_64bit_address(ctrl_r, 0x0dc, (uintptr_t)cma_get_phy_addr(hw_P.tile_col_offsets_hw));
        write_64bit_address(ctrl_r, 0x0e8, (uintptr_t)cma_get_phy_addr(hw_P.row_idx_tiled_hw));
        write_64bit_address(ctrl_r, 0x0f4, (uintptr_t)cma_get_phy_addr(hw_P.col_ptr_tiled_hw));
        write_64bit_address(ctrl_r, 0x100, (uintptr_t)cma_get_phy_addr(hw_P.values_tiled_hw));

        write_64bit_address(ctrl_r, 0x10c, (uintptr_t)cma_get_phy_addr(P_diag_hw));
        write_64bit_address(ctrl_r, 0x118, (uintptr_t)cma_get_phy_addr(l_in_hw));
        write_64bit_address(ctrl_r, 0x124, (uintptr_t)cma_get_phy_addr(u_in_hw));
        write_64bit_address(ctrl_r, 0x130, (uintptr_t)cma_get_phy_addr(q_in_hw));
        write_64bit_address(ctrl_r, 0x13c, (uintptr_t)cma_get_phy_addr(rho_in_hw));
        write_64bit_address(ctrl_r, 0x148, (uintptr_t)cma_get_phy_addr(x_out_hw));
        write_64bit_address(ctrl_r, 0x154, (uintptr_t)cma_get_phy_addr(y_out_hw));

        // Run
        const double hw_start = get_time_ms();
        write_reg(ctrl, 0x00, 0x01);
        while ((read_reg(ctrl, 0x00) & 0x02u) == 0) {
        }
        const double hw_end = get_time_ms();

        const int admm_iters = (int)read_reg(ctrl, 0x98);
        const int pcg_iters = (int)read_reg(ctrl, 0xa8);
        const float r_prim = uint_to_float(read_reg(ctrl, 0xb8));
        const float r_dual = uint_to_float(read_reg(ctrl, 0xc8));
        const int status = (int)read_reg(ctrl, 0xd8);

        // Unscale x + objective/violation prints (matches notebook)
        float *x_unscaled = (float *)malloc((size_t)num_cols * sizeof(float));
        if (!x_unscaled) {
            fprintf(stderr, "ERROR: malloc(x_unscaled) failed\n");
            return 1;
        }
        for (int i = 0; i < num_cols; ++i) {
            x_unscaled[i] = x_out_hw[i] * E[i];
        }

        double obj = 0.0;
        for (int i = 0; i < num_cols; ++i) {
            obj += 0.5 * (double)P_diag_orig[i] * (double)x_unscaled[i] * (double)x_unscaled[i] + (double)q_orig[i] * (double)x_unscaled[i];
        }

        float *Ax = (float *)calloc((size_t)num_rows, sizeof(float));
        if (!Ax) {
            fprintf(stderr, "ERROR: calloc(Ax) failed\n");
            return 1;
        }

        for (int c = 0; c < num_cols; ++c) {
            for (int idx = (int)A_cptr[c]; idx < (int)A_cptr[c + 1]; ++idx) {
                const int rr = (int)A_ridx[idx];
                Ax[rr] += A_vals_orig[idx] * x_unscaled[c];
            }
        }

        float max_viol = 0.0f;
        for (int r = 0; r < num_rows; ++r) {
            const float viol_l = l_orig[r] - Ax[r];
            const float viol_u = Ax[r] - u_orig[r];
            float v = viol_l;
            if (viol_u > v)
                v = viol_u;
            if (v < 0.0f)
                v = 0.0f;
            if (v > max_viol)
                max_viol = v;
        }

        free(Ax);
        free(x_unscaled);

        const float avg_pcg = (admm_iters > 0) ? ((float)pcg_iters / (float)admm_iters) : 0.0f;

        results[adaptive_rho].admm_iters = admm_iters;
        results[adaptive_rho].pcg_iters = pcg_iters;
        results[adaptive_rho].r_prim = r_prim;
        results[adaptive_rho].r_dual = r_dual;
        results[adaptive_rho].viol = max_viol;
        results[adaptive_rho].obj = obj;
        results[adaptive_rho].hw_ms = (hw_end - hw_start);

        printf("HW execution time: %.4f ms\n", hw_end - hw_start);
        printf("Status: %s\n", status == 1 ? "Converged" : "Max Iterations");
        printf("ADMM Iterations: %d\n", admm_iters);
        printf("PCG Iterations : %d (Average pcg/admm: %.1f)\n", pcg_iters, avg_pcg);
        printf("Primal Residual: %.5e\n", r_prim);
        printf("Dual Residual  : %.5e\n", r_dual);
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

    munmap(ip_base, MAP_SIZE);
    close(mem_fd);

    // Cleanup CMA
    cma_free(A_row_idx_reg_hw);
    cma_free(A_val_reg_hw);
    cma_free(A_col_ptr_reg_hw);

    free_hw_tiled(&hw_A);
    free_hw_tiled(&hw_AT);
    free_hw_tiled(&hw_P);

    cma_free(P_diag_hw);
    cma_free(l_in_hw);
    cma_free(u_in_hw);
    cma_free(q_in_hw);
    cma_free(rho_in_hw);
    cma_free(x_out_hw);
    cma_free(y_out_hw);

    // Cleanup CPU
    free_tiled_matrix(&tm_A);
    free_tiled_matrix(&tm_AT);
    free_tiled_matrix(&tm_P);

    free(A_cptr);
    free(P_diag);
    free(q);
    free(l);
    free(u);
    free(A_ridx);
    free(A_vals);

    free(P_diag_orig);
    free(q_orig);
    free(l_orig);
    free(u_orig);
    free(A_vals_orig);

    free(E);
    free(D);
    free(A_col_norm);
    free(A_row_norm);
    free(E_new);
    free(D_new);
    free(rho);

    free(AT_cptr);
    free(AT_ridx);
    free(AT_vals);

    free(P_cptr);
    free(P_ridx);
    free(P_vals);

    printf("Buffers released.\n");
    return 0;
}
