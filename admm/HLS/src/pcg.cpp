#include "pcg.h"

static float inf_norm(const float *v, int size)
{
#pragma HLS INLINE
    float max_val = 0.0f;
    for (int i = 0; i < size; i += RESHAPE_FACTOR)
    {
#pragma HLS PIPELINE II = 1
        float local_max = 0.0f;
        for (int j = 0; j < RESHAPE_FACTOR; ++j)
        {
#pragma HLS UNROLL
            const int idx = i + j;
            if (idx < size)
            {
                local_max = hls::fmax(local_max, hls::fabs(v[idx]));
            }
        }
        max_val = hls::fmax(max_val, local_max);
    }
    return max_val;
}

static float dot_prod(const float *x, const float *y, int size)
{
    // 4 accumulators to hide the final FP addition latency
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    int i = 0;
    for (; i + 3 < size; i += 4) {
#pragma HLS PIPELINE II = 1
        // unroll 4 iterations of the loop
        const float p0 = x[i] * y[i];
        const float p1 = x[i + 1] * y[i + 1];
        const float p2 = x[i + 2] * y[i + 2];
        const float p3 = x[i + 3] * y[i + 3];

        // add into the oldest partial sum, then rotate.
        const float sum_ab = p0 + p1;
        const float sum_cd = p2 + p3;
        const float sum_all = sum_ab + sum_cd;

        const float updated = acc0 + sum_all;
        acc0 = acc1;
        acc1 = acc2;
        acc2 = acc3;
        acc3 = updated;
    }

    for (; i < size; ++i) {
#pragma HLS PIPELINE II = 1
        acc0 += x[i] * y[i];
    }
    // final reduction
    return (acc0 + acc1) + (acc2 + acc3);
}

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
         int pcg_max_iterations)
{
#pragma HLS INLINE

float r[MAX_SIZE];
float p[MAX_SIZE];

#pragma HLS BIND_STORAGE variable=r type=RAM_T2P impl=URAM
#pragma HLS ARRAY_RESHAPE variable=r type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=p type=RAM_T2P impl=URAM
#pragma HLS ARRAY_RESHAPE variable=p type=cyclic factor=RESHAPE_FACTOR dim=1

float alpha;
float beta;

// scratch1 = A*x
spmv_csc_tiled(num_rows, num_cols, mat_A, x, scratch1);

// scratch1 = rho .* (A*x)
for (int i = 0; i < num_rows; i += RESHAPE_FACTOR)
{
#pragma HLS PIPELINE II = 1
    for (int j = 0; j < RESHAPE_FACTOR; ++j)
    {
#pragma HLS UNROLL
        const int idx = i + j;
        if (idx < num_rows)
        {
            scratch1[idx] = rho[idx] * scratch1[idx];
        }
    }
}

// scratch2 = A^T * (rho .* (A*x))
spmv_csc_tiled(num_cols, num_rows, mat_AT, scratch1, scratch2);

// scratch1 = P*x
spmv_csc_tiled(num_cols, num_cols, mat_P, x, scratch1);

// scratch2 = P*x + sigma*x + A^T*(rho*(A*x))
for (int i = 0; i < num_cols; i += RESHAPE_FACTOR)
{
#pragma HLS PIPELINE II = 1
    for (int j = 0; j < RESHAPE_FACTOR; ++j)
    {
#pragma HLS UNROLL
        const int idx = i + j;
        if (idx < num_cols)
        {
            scratch2[idx] = scratch2[idx] + sigma * x[idx] + scratch1[idx];
        }
    }
}

// init r=b-Kx, z=M_inv*r, p=z
for (int i = 0; i < num_cols; ++i)
{
#pragma HLS PIPELINE II = 1
    const float r0 = b[i] - scratch2[i];
    r[i] = r0;
    const float z0 = M_inv[i] * r0;
    scratch2[i] = z0; // z
    p[i] = z0;
}



float rT_y = dot_prod(r, scratch2, num_cols); // r^T y

// Relative residual stop: ||r||inf <= eps * ||b||inf
float r_norm = 0.0f;
r_norm = inf_norm(r, num_cols);


#define K_p b // Reuse b's memory to store K*p result since b is not needed after initialization

int iter_count = 0;

for (int k = 0; k < pcg_max_iterations && r_norm > epsilon; ++k)
{
    // scratch1 = A*p
    spmv_csc_tiled(num_rows, num_cols, mat_A, p, scratch1);

    // scratch1 = rho .* (A*p)
    for (int i = 0; i < num_rows; i += RESHAPE_FACTOR)
    {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < RESHAPE_FACTOR; ++j)
        {
#pragma HLS UNROLL
            const int idx = i + j;
            if (idx < num_rows)
            {
                scratch1[idx] = rho[idx] * scratch1[idx];
            }
        }
    }

    // K_p = A^T * (rho .* (A*p))
    spmv_csc_tiled(num_cols, num_rows, mat_AT, scratch1, K_p);

    // scratch1 = P*p
    spmv_csc_tiled(num_cols, num_cols, mat_P, p, scratch1);

    // K_p = P*p + sigma*p + A^T*(rho*(A*p))
    for (int i = 0; i < num_cols; i += RESHAPE_FACTOR)
    {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < RESHAPE_FACTOR; ++j)
        {
#pragma HLS UNROLL
            const int idx = i + j;
            if (idx < num_cols)
            {
                K_p[idx] = K_p[idx] + sigma * p[idx] + scratch1[idx];
            }
        }
    }

    // alpha = (r^T y) / (p^T K p)
    float pT_K_p = dot_prod(p, K_p, num_cols);
    if (pT_K_p <= 0.0f) 
    {
        break;
    }

    alpha = rT_y / pT_K_p; // Scalar division

    for (int i = 0; i < num_cols; i += RESHAPE_FACTOR)
    {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < RESHAPE_FACTOR; ++j)
        {
#pragma HLS UNROLL
            const int idx = i + j;
            if (idx < num_cols)
            {
                const float p_val = p[idx];
                const float K_p_val = K_p[idx];
                const float x_old = x[idx];
                const float r_old = r[idx];

                const float r_new = r_old - alpha * K_p_val; // r_new = r_old - alpha * K_p

                x[idx] = x_old + alpha * p_val;
                r[idx] = r_new;
                scratch2[idx] = M_inv[idx] * r_new; // y = M_inv * r_new
            }
        }
    }
    // beta = (r_new^T y_new) / (r_old^T y_old)
    float rT_y_next = dot_prod(r, scratch2, num_cols);
    if (rT_y_next <= 0.0f) 
    {
        break;
    }
    beta = rT_y_next / rT_y; // Scalar division

    for (int i = 0; i < num_cols; i += RESHAPE_FACTOR)
    {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < RESHAPE_FACTOR; ++j)
        {
#pragma HLS UNROLL
            const int idx = i + j;
            if (idx < num_cols)
            {
                p[idx] = scratch2[idx] + beta * p[idx]; // p = y + beta * p
            }
        }
    }

    rT_y = rT_y_next;

    // Update r_norm for the loop condition
    r_norm = inf_norm(r, num_cols);

    iter_count++;
}

    *pcg_num_iterations = iter_count;
}