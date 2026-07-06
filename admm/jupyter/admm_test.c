#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>

// PYNQ libcma prototypes
#ifdef __cplusplus
extern "C" {
#endif
void *cma_alloc(uint32_t len, uint32_t cacheable);
unsigned long cma_get_phy_addr(void *buf);
void cma_free(void *buf);
void cma_flush_cache(void *buf, unsigned int phys_addr, int size);
void cma_invalidate_cache(void *buf, unsigned int phys_addr, int size);
#ifdef __cplusplus
}
#endif

// =====================================================================
// Configuration & Register Map
// =====================================================================
#define NUM_ROWS 15
#define NUM_COLS 15
#define PACK_SIZE 16
#define MAX_NNZ_WORDS 16 // Sufficient for 15x15
#define PAD 16           // Safety padding for AXI bursts

// Directory where Python dumped the binary files
#define DATA_DIR "/home/xilinx/admm/baremetal/data/"

// Hardware Map
#define ADMM_IP_CONTROL_BASE_ADDR   0xA0000000 
#define ADMM_IP_CONTROL_R_BASE_ADDR 0xA0010000
#define MAP_SIZE 0x20000UL // 128KB (covers both addresses)
#define MAP_MASK (MAP_SIZE - 1)

// Control Bundle (0xA0000000)
#define ADDR_AP_CTRL             0x00
#define ADDR_NUM_ROWS            0x10
#define ADDR_NUM_COLS            0x18
#define ADDR_A_NNZ               0x20
#define ADDR_P_NNZ               0x28
#define ADDR_SIGMA               0x30
#define ADDR_ALPHA               0x38
#define ADDR_ADMM_MAX_ITER       0x40
#define ADDR_PCG_MAX_ITER        0x48
#define ADDR_ADMM_ITERS_OUT      0x50
#define ADDR_PCG_ITERS_OUT       0x60
#define ADDR_R_PRIM_OUT          0x70
#define ADDR_R_DUAL_OUT          0x80
#define ADDR_STATUS_OUT          0x90

// Control_R Bundle (0xA0010000)
#define ADDR_R_A_ROW         0x10
#define ADDR_R_A_COL         0x1c
#define ADDR_R_A_VAL         0x28
#define ADDR_R_AT_ROW        0x34
#define ADDR_R_AT_COL        0x40
#define ADDR_R_AT_VAL        0x4c
#define ADDR_R_P_ROW         0x58
#define ADDR_R_P_COL         0x64
#define ADDR_R_P_VAL         0x70
#define ADDR_R_P_DIAG        0x7c
#define ADDR_R_L_IN          0x88
#define ADDR_R_U_IN          0x94
#define ADDR_R_Q_IN          0xa0
#define ADDR_R_RHO_IN        0xac
#define ADDR_R_X_OUT         0xb8
#define ADDR_R_Y_OUT         0xc4

// Hardware Structs (32-bit types, 16 elements = 512 bits)
typedef struct { int32_t data[PACK_SIZE]; } int32_words;
typedef struct { float   data[PACK_SIZE]; } float32_words;

// =====================================================================
// Helper Functions
// =====================================================================
double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

void write_reg(void *base, uint32_t offset, uint32_t val) {
    *((volatile uint32_t *)((uint8_t *)base + offset)) = val;
}

uint32_t read_reg(void *base, uint32_t offset) {
    return *((volatile uint32_t *)((uint8_t *)base + offset));
}

void write_64bit_address(void *base, uint32_t offset, uintptr_t address) {
    write_reg(base, offset, (uint32_t)(address & 0xFFFFFFFF));
    write_reg(base, offset + 0x04, (uint32_t)((uint64_t)address >> 32));
}

uint32_t float_to_uint(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

float uint_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Helpers for loading binary files
void load_float_bin(const char* filename, float* dest, size_t count) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("ERROR: Could not open %s\n", filename);
        exit(1);
    }
    size_t read_count = fread(dest, sizeof(float), count, f);
    if (read_count != count) {
        printf("WARNING: Expected %zu floats in %s, but read %zu\n", count, filename, read_count);
    }
    fclose(f);
}

void load_int_bin(const char* filename, int* dest, size_t count) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("ERROR: Could not open %s\n", filename);
        exit(1);
    }
    size_t read_count = fread(dest, sizeof(int), count, f);
    if (read_count != count) {
        printf("WARNING: Expected %zu ints in %s, but read %zu\n", count, filename, read_count);
    }
    fclose(f);
}

void pack_csc_to_words(int nnz, const int *row_idx, const float *values, 
                       int32_words *row_words, float32_words *val_words) 
{
    for(int w = 0; w < MAX_NNZ_WORDS; ++w) {
        for(int lane = 0; lane < PACK_SIZE; ++lane) {
            row_words[w].data[lane] = 0;
            val_words[w].data[lane] = 0.0f;
        }
    }
    for(int idx = 0; idx < nnz; ++idx) {
        int w = idx / PACK_SIZE;
        int lane = idx % PACK_SIZE;
        row_words[w].data[lane] = row_idx[idx];
        val_words[w].data[lane] = values[idx];
    }
}

void transpose_csc(int rows, int cols, const int *col_ptr, const int *row_idx, const float *values,
                   int *col_ptr_t, int *row_idx_t, float *values_t) 
{
    for(int i = 0; i <= rows; i++) col_ptr_t[i] = 0;
    int *nnz_per_col = (int*)calloc(rows, sizeof(int));
    int nnz = col_ptr[cols];

    for (int idx = 0; idx < nnz; ++idx) nnz_per_col[row_idx[idx]]++;
    for (int c = 0; c < rows; ++c) col_ptr_t[c + 1] = col_ptr_t[c] + nnz_per_col[c];

    int *next = (int*)malloc(rows * sizeof(int));
    for (int i = 0; i < rows; i++) next[i] = col_ptr_t[i];

    for (int c = 0; c < cols; ++c) {
        for (int idx = col_ptr[c]; idx < col_ptr[c + 1]; ++idx) {
            int r = row_idx[idx];
            int dst = next[r]++;
            row_idx_t[dst] = c;
            values_t[dst] = values[idx];
        }
    }
    free(nnz_per_col); free(next);
}

// =====================================================================
// Main Logic
// =====================================================================
int main() {
    printf("\n--- Starting ADMM Hardware Testbench (15x15 File Loaded) ---\n");

    float sigma = 1e-6f;
    float alpha = 1.6f;
    int admm_max_iter = 2000;
    int pcg_max_iter = 500;

    // =====================================================================
    // 1. Allocate CPU Matrices & Load Data
    // =====================================================================
    printf("Loading Target Problem Data from %s...\n", DATA_DIR);
    
    int A_nnz = NUM_COLS; 
    int P_nnz = NUM_COLS; 

    float *x_true = (float *)malloc(NUM_COLS * sizeof(float));
    float *q      = (float *)calloc(NUM_COLS, sizeof(float));

    int *P_col_ptr_cpu   = (int *)malloc((NUM_COLS + 1) * sizeof(int));
    int *P_row_idx_cpu   = (int *)malloc(P_nnz * sizeof(int));
    float *P_values_cpu  = (float *)malloc(P_nnz * sizeof(float));
    float *P_diag_cpu    = (float *)malloc(NUM_COLS * sizeof(float));

    int *A_col_ptr_cpu   = (int *)malloc((NUM_COLS + 1) * sizeof(int));
    int *A_row_idx_cpu   = (int *)malloc(A_nnz * sizeof(int));
    float *A_values_cpu  = (float *)malloc(A_nnz * sizeof(float));

    float *l_cpu   = (float *)malloc(NUM_ROWS * sizeof(float));
    float *u_cpu   = (float *)malloc(NUM_ROWS * sizeof(float));
    float *rho_cpu = (float *)malloc(NUM_ROWS * sizeof(float));

    // File path buffer
    char filepath[256];

    sprintf(filepath, "%s%s", DATA_DIR, "x_true.bin");
    load_float_bin(filepath, x_true, NUM_COLS);

    sprintf(filepath, "%s%s", DATA_DIR, "q.bin");
    load_float_bin(filepath, q, NUM_COLS);

    sprintf(filepath, "%s%s", DATA_DIR, "P_values.bin");
    load_float_bin(filepath, P_values_cpu, NUM_COLS);

    sprintf(filepath, "%s%s", DATA_DIR, "P_col_ptr.bin");
    load_int_bin(filepath, P_col_ptr_cpu, NUM_COLS + 1);

    // Reconstruct P_diag and P_row_idx locally
    for (int c = 0; c < NUM_COLS; ++c) {
        P_diag_cpu[c] = P_values_cpu[c];
        P_row_idx_cpu[c] = c;
    }

    sprintf(filepath, "%s%s", DATA_DIR, "A_values.bin");
    load_float_bin(filepath, A_values_cpu, A_nnz);

    sprintf(filepath, "%s%s", DATA_DIR, "A_col_ptr.bin");
    load_int_bin(filepath, A_col_ptr_cpu, NUM_COLS + 1);

    sprintf(filepath, "%s%s", DATA_DIR, "A_row_idx.bin");
    load_int_bin(filepath, A_row_idx_cpu, A_nnz);

    sprintf(filepath, "%s%s", DATA_DIR, "l_in.bin");
    load_float_bin(filepath, l_cpu, NUM_ROWS);

    sprintf(filepath, "%s%s", DATA_DIR, "u_in.bin");
    load_float_bin(filepath, u_cpu, NUM_ROWS);

    sprintf(filepath, "%s%s", DATA_DIR, "rho_in.bin");
    load_float_bin(filepath, rho_cpu, NUM_ROWS);

    // Compute Transpose of A dynamically in software
    int *AT_col_ptr_cpu  = (int *)malloc((NUM_ROWS + 1) * sizeof(int));
    int *AT_row_idx_cpu  = (int *)malloc(A_nnz * sizeof(int));
    float *AT_values_cpu = (float *)malloc(A_nnz * sizeof(float));
    transpose_csc(NUM_ROWS, NUM_COLS, A_col_ptr_cpu, A_row_idx_cpu, A_values_cpu, 
                  AT_col_ptr_cpu, AT_row_idx_cpu, AT_values_cpu);

    // =====================================================================
    // 2. Allocate CMA (Contiguous Physical Memory, CACHED = 1)
    // =====================================================================
    printf("Allocating Hardware CMA Buffers...\n");

    int32_words *A_row_hw   = (int32_words *)cma_alloc(MAX_NNZ_WORDS * sizeof(int32_words), 1);
    float32_words *A_val_hw = (float32_words *)cma_alloc(MAX_NNZ_WORDS * sizeof(float32_words), 1);
    int *A_col_ptr_hw       = (int *)cma_alloc((NUM_COLS + 1 + PAD) * sizeof(int), 1);

    int32_words *AT_row_hw   = (int32_words *)cma_alloc(MAX_NNZ_WORDS * sizeof(int32_words), 1);
    float32_words *AT_val_hw = (float32_words *)cma_alloc(MAX_NNZ_WORDS * sizeof(float32_words), 1);
    int *AT_col_ptr_hw       = (int *)cma_alloc((NUM_ROWS + 1 + PAD) * sizeof(int), 1);

    int32_words *P_row_hw   = (int32_words *)cma_alloc(MAX_NNZ_WORDS * sizeof(int32_words), 1);
    float32_words *P_val_hw = (float32_words *)cma_alloc(MAX_NNZ_WORDS * sizeof(float32_words), 1);
    int *P_col_ptr_hw       = (int *)cma_alloc((NUM_COLS + 1 + PAD) * sizeof(int), 1);

    float *P_diag_hw  = (float *)cma_alloc((NUM_COLS + PAD) * sizeof(float), 1);
    float *l_hw       = (float *)cma_alloc((NUM_ROWS + PAD) * sizeof(float), 1);
    float *u_hw       = (float *)cma_alloc((NUM_ROWS + PAD) * sizeof(float), 1);
    float *q_hw       = (float *)cma_alloc((NUM_COLS + PAD) * sizeof(float), 1);
    float *rho_hw     = (float *)cma_alloc((NUM_ROWS + PAD) * sizeof(float), 1);

    float *x_out_hw   = (float *)cma_alloc((NUM_COLS + PAD) * sizeof(float), 1);
    float *y_out_hw   = (float *)cma_alloc((NUM_ROWS + PAD) * sizeof(float), 1);

    // Zero-initialize
    memset(A_col_ptr_hw, 0, (NUM_COLS + 1 + PAD) * sizeof(int));
    memset(AT_col_ptr_hw, 0, (NUM_ROWS + 1 + PAD) * sizeof(int));
    memset(P_col_ptr_hw, 0, (NUM_COLS + 1 + PAD) * sizeof(int));
    memset(P_diag_hw, 0, (NUM_COLS + PAD) * sizeof(float));
    memset(l_hw, 0, (NUM_ROWS + PAD) * sizeof(float));
    memset(u_hw, 0, (NUM_ROWS + PAD) * sizeof(float));
    memset(q_hw, 0, (NUM_COLS + PAD) * sizeof(float));
    memset(rho_hw, 0, (NUM_ROWS + PAD) * sizeof(float));
    memset(x_out_hw, 0, (NUM_COLS + PAD) * sizeof(float));
    memset(y_out_hw, 0, (NUM_ROWS + PAD) * sizeof(float));

    // Populate buffers
    memcpy(A_col_ptr_hw, A_col_ptr_cpu, (NUM_COLS + 1) * sizeof(int));
    memcpy(P_col_ptr_hw, P_col_ptr_cpu, (NUM_COLS + 1) * sizeof(int));
    memcpy(AT_col_ptr_hw, AT_col_ptr_cpu, (NUM_ROWS + 1) * sizeof(int));
    
    memcpy(P_diag_hw, P_diag_cpu, NUM_COLS * sizeof(float));
    memcpy(l_hw, l_cpu, NUM_ROWS * sizeof(float));
    memcpy(u_hw, u_cpu, NUM_ROWS * sizeof(float));
    memcpy(q_hw, q, NUM_COLS * sizeof(float));
    memcpy(rho_hw, rho_cpu, NUM_ROWS * sizeof(float));

    pack_csc_to_words(A_nnz, A_row_idx_cpu, A_values_cpu, A_row_hw, A_val_hw);
    pack_csc_to_words(A_nnz, AT_row_idx_cpu, AT_values_cpu, AT_row_hw, AT_val_hw);
    pack_csc_to_words(P_nnz, P_row_idx_cpu, P_values_cpu, P_row_hw, P_val_hw);

    // =====================================================================
    // 3. FLUSH CACHES to DDR
    // =====================================================================
    cma_flush_cache(A_row_hw, cma_get_phy_addr(A_row_hw), MAX_NNZ_WORDS * sizeof(int32_words));
    cma_flush_cache(A_val_hw, cma_get_phy_addr(A_val_hw), MAX_NNZ_WORDS * sizeof(float32_words));
    cma_flush_cache(A_col_ptr_hw, cma_get_phy_addr(A_col_ptr_hw), (NUM_COLS + 1 + PAD) * sizeof(int));
    
    cma_flush_cache(AT_row_hw, cma_get_phy_addr(AT_row_hw), MAX_NNZ_WORDS * sizeof(int32_words));
    cma_flush_cache(AT_val_hw, cma_get_phy_addr(AT_val_hw), MAX_NNZ_WORDS * sizeof(float32_words));
    cma_flush_cache(AT_col_ptr_hw, cma_get_phy_addr(AT_col_ptr_hw), (NUM_ROWS + 1 + PAD) * sizeof(int));
    
    cma_flush_cache(P_row_hw, cma_get_phy_addr(P_row_hw), MAX_NNZ_WORDS * sizeof(int32_words));
    cma_flush_cache(P_val_hw, cma_get_phy_addr(P_val_hw), MAX_NNZ_WORDS * sizeof(float32_words));
    cma_flush_cache(P_col_ptr_hw, cma_get_phy_addr(P_col_ptr_hw), (NUM_COLS + 1 + PAD) * sizeof(int));
    
    cma_flush_cache(P_diag_hw, cma_get_phy_addr(P_diag_hw), (NUM_COLS + PAD) * sizeof(float));
    cma_flush_cache(l_hw, cma_get_phy_addr(l_hw), (NUM_ROWS + PAD) * sizeof(float));
    cma_flush_cache(u_hw, cma_get_phy_addr(u_hw), (NUM_ROWS + PAD) * sizeof(float));
    cma_flush_cache(q_hw, cma_get_phy_addr(q_hw), (NUM_COLS + PAD) * sizeof(float));
    cma_flush_cache(rho_hw, cma_get_phy_addr(rho_hw), (NUM_ROWS + PAD) * sizeof(float));
    cma_flush_cache(x_out_hw, cma_get_phy_addr(x_out_hw), (NUM_COLS + PAD) * sizeof(float));
    cma_flush_cache(y_out_hw, cma_get_phy_addr(y_out_hw), (NUM_ROWS + PAD) * sizeof(float));

    // =====================================================================
    // 4. Map IP / Write Registers
    // =====================================================================
    printf("Configuring Hardware Registers...\n");
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    void *ip_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, ADMM_IP_CONTROL_BASE_ADDR & ~MAP_MASK);
    
    void *control_ip   = ip_base + (ADMM_IP_CONTROL_BASE_ADDR & MAP_MASK);
    void *control_r_ip = ip_base + (ADMM_IP_CONTROL_R_BASE_ADDR & MAP_MASK);

    // Control Bundle Writes
    write_reg(control_ip, ADDR_NUM_ROWS, NUM_ROWS);
    write_reg(control_ip, ADDR_NUM_COLS, NUM_COLS);
    write_reg(control_ip, ADDR_A_NNZ, A_nnz);
    write_reg(control_ip, ADDR_P_NNZ, P_nnz);
    write_reg(control_ip, ADDR_SIGMA, float_to_uint(sigma));
    write_reg(control_ip, ADDR_ALPHA, float_to_uint(alpha));
    write_reg(control_ip, ADDR_ADMM_MAX_ITER, admm_max_iter);
    write_reg(control_ip, ADDR_PCG_MAX_ITER, pcg_max_iter);

    // Control_R Bundle Writes
    write_64bit_address(control_r_ip, ADDR_R_A_ROW, cma_get_phy_addr(A_row_hw));
    write_64bit_address(control_r_ip, ADDR_R_A_COL, cma_get_phy_addr(A_col_ptr_hw));
    write_64bit_address(control_r_ip, ADDR_R_A_VAL, cma_get_phy_addr(A_val_hw));
    
    write_64bit_address(control_r_ip, ADDR_R_AT_ROW, cma_get_phy_addr(AT_row_hw));
    write_64bit_address(control_r_ip, ADDR_R_AT_COL, cma_get_phy_addr(AT_col_ptr_hw));
    write_64bit_address(control_r_ip, ADDR_R_AT_VAL, cma_get_phy_addr(AT_val_hw));
    
    write_64bit_address(control_r_ip, ADDR_R_P_ROW, cma_get_phy_addr(P_row_hw));
    write_64bit_address(control_r_ip, ADDR_R_P_COL, cma_get_phy_addr(P_col_ptr_hw));
    write_64bit_address(control_r_ip, ADDR_R_P_VAL, cma_get_phy_addr(P_val_hw));
    write_64bit_address(control_r_ip, ADDR_R_P_DIAG, cma_get_phy_addr(P_diag_hw));
    
    write_64bit_address(control_r_ip, ADDR_R_L_IN, cma_get_phy_addr(l_hw));
    write_64bit_address(control_r_ip, ADDR_R_U_IN, cma_get_phy_addr(u_hw));
    write_64bit_address(control_r_ip, ADDR_R_Q_IN, cma_get_phy_addr(q_hw));
    write_64bit_address(control_r_ip, ADDR_R_RHO_IN, cma_get_phy_addr(rho_hw));
    
    write_64bit_address(control_r_ip, ADDR_R_X_OUT, cma_get_phy_addr(x_out_hw));
    write_64bit_address(control_r_ip, ADDR_R_Y_OUT, cma_get_phy_addr(y_out_hw));

    // =====================================================================
    // 5. Execute
    // =====================================================================
    printf("Starting Hardware Accelerator...\n");
    double hw_start = get_time_ms();

    write_reg(control_ip, ADDR_AP_CTRL, 0x01);
    while ((read_reg(control_ip, ADDR_AP_CTRL) & 0x02) == 0); // Polling ap_done

    double hw_end = get_time_ms();
    printf("Hardware execution time: %.4f ms\n", hw_end - hw_start);

    // Read outputs
    uint32_t admm_iters_out = read_reg(control_ip, ADDR_ADMM_ITERS_OUT);
    uint32_t pcg_iters_out  = read_reg(control_ip, ADDR_PCG_ITERS_OUT);
    float r_prim_out        = uint_to_float(read_reg(control_ip, ADDR_R_PRIM_OUT));
    float r_dual_out        = uint_to_float(read_reg(control_ip, ADDR_R_DUAL_OUT));
    uint32_t status_out     = read_reg(control_ip, ADDR_STATUS_OUT);

    // =====================================================================
    // 6. INVALIDATE CACHE (Fetch results from DDR)
    // =====================================================================
    cma_invalidate_cache(x_out_hw, cma_get_phy_addr(x_out_hw), (NUM_COLS + PAD) * sizeof(float));
    cma_invalidate_cache(y_out_hw, cma_get_phy_addr(y_out_hw), (NUM_ROWS + PAD) * sizeof(float));

    // =====================================================================
    // 7. Verify Results
    // =====================================================================
    printf("\n--- Simulation Results ---\n");
    printf("Status: %s\n", status_out == 1 ? "Converged" : "Max Iterations Reached");
    printf("ADMM Iterations: %u\n", admm_iters_out);
    printf("Total PCG Iterations: %u\n", pcg_iters_out);
    printf("Primal Residual: %e\n", r_prim_out);
    printf("Dual Residual: %e\n", r_dual_out);

    float mae = 0.0f;
    for (int i = 0; i < NUM_COLS; ++i) {
        mae += fabsf(x_out_hw[i] - x_true[i]);
    }
    mae /= NUM_COLS;

    printf("\nMean Absolute Error from x_true: %e\n", mae);
    printf("--- Full x_out vs Expected ---\n");
    for (int i = 0; i < NUM_COLS; i++) {
        printf("x[%2d]: %13.5f | Expected: %13.5f\n", i, x_out_hw[i], x_true[i]);
    }

    if (status_out == 1 && mae < 1e-2f) {
        printf("\n>>> SUCCESS: Problem converged perfectly to the ground truth! <<<\n");
    } else {
        printf("\n>>> FAILED: Did not converge to the expected target. <<<\n");
    }

    // =====================================================================
    // 8. Cleanup
    // =====================================================================
    munmap(ip_base, MAP_SIZE); 
    close(mem_fd);

    cma_free(A_row_hw); cma_free(A_val_hw); cma_free(A_col_ptr_hw);
    cma_free(AT_row_hw); cma_free(AT_val_hw); cma_free(AT_col_ptr_hw);
    cma_free(P_row_hw); cma_free(P_val_hw); cma_free(P_col_ptr_hw);
    cma_free(P_diag_hw); cma_free(l_hw); cma_free(u_hw); cma_free(q_hw); 
    cma_free(rho_hw); cma_free(x_out_hw); cma_free(y_out_hw);

    free(x_true); free(q);
    free(A_col_ptr_cpu); free(A_row_idx_cpu); free(A_values_cpu);
    free(AT_col_ptr_cpu); free(AT_row_idx_cpu); free(AT_values_cpu);
    free(P_col_ptr_cpu); free(P_row_idx_cpu); free(P_values_cpu); free(P_diag_cpu);
    free(l_cpu); free(u_cpu); free(rho_cpu);

    return 0;
}