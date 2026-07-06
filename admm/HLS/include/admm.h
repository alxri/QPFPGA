#ifndef ADMM_H
#define ADMM_H

#include "spmv_csc.h"
#include "pcg.h"
#include "hls_math.h"
#include "config.h"

void admm(int num_rows,
          int num_cols,
          // Regular format Matrix A (for preconditioner update)
          const int16 *A_row_idx, const int *A_col_ptr, const float16 *A_values, int A_nnz,
          // Tiled Matrix A
          int A_num_row_tiles, int A_num_col_tiles,
          const int *A_tile_nnz_counts, const int *A_tile_nnz_offsets, const int *A_tile_col_offsets,
          const int16 *A_row_idx_tiled, const int *A_col_ptr_tiled, const float16 *A_values_tiled,
          // Tiled Matrix AT
          int AT_num_row_tiles, int AT_num_col_tiles,
          const int *AT_tile_nnz_counts, const int *AT_tile_nnz_offsets, const int *AT_tile_col_offsets,
          const int16 *AT_row_idx_tiled, const int *AT_col_ptr_tiled, const float16 *AT_values_tiled,
          // Tiled Matrix P
          int P_num_row_tiles, int P_num_col_tiles,
          const int *P_tile_nnz_counts, const int *P_tile_nnz_offsets, const int *P_tile_col_offsets,
          const int16 *P_row_idx_tiled, const int *P_col_ptr_tiled, const float16 *P_values_tiled,
          // Remaining standard arguments
          const float *P_diag,
          const float *l_in,
          const float *u_in,
          const float *q_in,
          float sigma,
          float alpha,
          float *rho_in,
          int admm_max_iterations,
          int pcg_max_iterations,
          bool adaptive_rho,
          float eps_abs,
          float eps_rel,
          float pcg_tol_fraction,
          float *x_out,
          float *y_out,
          int *admm_num_iterations_out,
          int *pcg_num_iterations_out,
          float *r_prim_out,
          float *r_dual_out,
          int *status_out);

#endif // ADMM_H