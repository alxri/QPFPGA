#ifndef PCG_H
#define PCG_H

#include "spmv_csc.h"
#include "hls_math.h"
#include "config.h"

void pcg(int num_rows,
         int num_cols,
         const TiledMatrix &mat_A,
         const TiledMatrix &mat_AT,
         const TiledMatrix &mat_P,
         const float *M_inv,
         const float *rho,
         const float sigma,
         const float epsilon,
         float *x,
         float *b,
         float *scratch1,
         float *scratch2,
         int *pcg_num_iterations,
         int pcg_max_iterations);

#endif // PCG_H