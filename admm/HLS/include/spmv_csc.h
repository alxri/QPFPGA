#ifndef SPMV_CSC_H
#define SPMV_CSC_H

#include "hls_stream.h"
#include "hls_vector.h"
#include "config.h"

#ifndef MAX_NNZ
#define MAX_NNZ 200000 // For Vitis simulation purposes, not determining for hw
#endif

#ifndef PACK_SIZE
#define PACK_SIZE 16
#endif

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define MAX_COL_PTR (MAX_COLS + 1)
#define MAX_COL_WORDS CEIL_DIV(MAX_COLS, PACK_SIZE)
#define MAX_ROW_WORDS CEIL_DIV(MAX_ROWS, PACK_SIZE)
#define MAX_NNZ_WORDS CEIL_DIV(MAX_NNZ, PACK_SIZE)

// 512 bit packed types (16 floats per packet)
typedef hls::vector<float, PACK_SIZE> float16;
typedef hls::vector<int, PACK_SIZE> int16;

struct TiledMatrix {
    int num_row_tiles;
    int num_col_tiles;
    const int *nnz_counts; 
    const int *nnz_offsets;
    const int *col_offsets;
    const int16 *row_idx;  
    const int *col_ptr;    
    const float16 *values; 
};

void spmv_csc(int num_rows,
              int num_cols,
              int nnz,
              const int16 *A_row_idx,
              const int *A_col_ptr,
              const float16 *A_values,
              const float *x,
              float *y);

void spmv_csc_tiled(int global_num_rows,
                    int global_num_cols,
                    const TiledMatrix &A,
                    float *x,
                    float *y);

#endif // SPMV_CSC_H