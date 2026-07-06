#include "admm.h"


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

inline float clamp(float val, float lower, float upper)
{
    return (val < lower) ? lower : (val > upper) ? upper : val;
}

void update_preconditioner(
    const float *P_diag, const float16 *A_values, const int16 *A_row_idx, const int *A_col_ptr, 
    float *rho, float sigma, float *M_inv, int num_cols) 
{
    // Local caches for the 512-bit wide AXI words
    float16 current_A_vals;
    int16 current_A_rows;
    int current_word_idx = -1; // Tracks which 16-element chunk is currently loaded

    for (int i = 0; i < num_cols; i++) {
        
        // Array of accumulators to break the RAW dependency
        float A_col_sum_lanes[16];
#pragma HLS ARRAY_PARTITION variable=A_col_sum_lanes complete
        
        // Initialize lanes to zero
        for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
            A_col_sum_lanes[k] = 0.0f;
        }
        
        int start = A_col_ptr[i];
        int end = A_col_ptr[i+1];
        
        for (int p = start; p < end; p++) {
#pragma HLS PIPELINE II=1
            // Removed the dangerous RAW false pragma!

            // Fast bitwise math: / 16 is >> 4, % 16 is & 15
            int word_idx = p >> 4;  
            int lane_idx = p & 15;  

            // Fetch new 512-bit word when crossing boundaries
            if (word_idx != current_word_idx) {
                current_A_vals = A_values[word_idx];
                current_A_rows = A_row_idx[word_idx];
                current_word_idx = word_idx;
            }
            
            // Extract from the local cache
            int row_j = current_A_rows[lane_idx];
            float val = current_A_vals[lane_idx];
            
            // Accumulate into the specific lane.
            A_col_sum_lanes[lane_idx] += rho[row_j] * (val * val); 
        }
        
        // Combinational reduction tree at the end of the column
        float A_col_sum = 0.0f;
        for (int k = 0; k < 16; k++) {
#pragma HLS UNROLL
            A_col_sum += A_col_sum_lanes[k];
        }
        
        // Final calculation
        M_inv[i] = 1.0f / (P_diag[i] + sigma + A_col_sum);
    }
}

void update_eps(float b_norm,
                float r_prim,
                float r_dual,
                float &eps,
                float &eps_old,
                int num_iterations,
                int num_pcg_iterations,
                int &zero_pcg_iters,
                float &pcg_tol_fraction)
{
#pragma HLS INLINE
    eps_old = eps;

    // Tighten tolerance if PCG solves in 0 iterations repeatedly
    // Do not tighten if it hits max iterations
    if (num_iterations > 0)
    {
        if (num_pcg_iterations == 0) {
            zero_pcg_iters++;
        } else {
            zero_pcg_iters = 0;
        }

        if (zero_pcg_iters >= 5) {
            pcg_tol_fraction *= 0.5f;
            zero_pcg_iters = 0;
        }
    }

    float eps_new;
    if (num_iterations == 0)
    {
        eps_new = b_norm * pcg_tol_fraction; 
    }
    else
    {
        eps_new = pcg_tol_fraction * hls::sqrt(r_prim * r_dual);
    }

    if (eps_new < OSQP_CG_TOL_MIN) {
        eps_new = OSQP_CG_TOL_MIN;
    }
    if (num_iterations > 0 && eps_new > eps_old) {
        eps_new = eps_old;
    }

    eps = eps_new;
}

void update_b(int num_rows, int num_cols, float sigma, float *x, float *q, const TiledMatrix &mat_AT, float *rho, float *y, float *z, float *b, float *scratch_in, float *scratch_out, float *b_norm)
{
#pragma HLS INLINE
    *b_norm = 0.0f;

    for (int i = 0; i < num_rows; i++) {
#pragma HLS PIPELINE II=1
        scratch_in[i] = rho[i] * z[i] - y[i]; // AT * (rho * z - y)
    }

    spmv_csc_tiled(num_cols, num_rows, mat_AT, scratch_in, scratch_out); // scratch_out = AT * (rho * z - y)

    for (int i = 0; i < num_cols; i++) {
#pragma HLS PIPELINE II=1
        b[i] = sigma * x[i] - q[i] + scratch_out[i]; // b = sigma*x - q + AT*(rho*z - y)
    }
    *b_norm = inf_norm(b, num_cols);
}

static float estimate_new_rho(float r_prim, float r_dual, 
                              float norm_z, float norm_Ax, 
                              float norm_q, float ATy_norm, float Px_norm, 
                              float current_rho_base)
{
#pragma HLS INLINE

    // Scale the primal and dual residuals by their respective norms
    float prim_res_norm = hls::fmax(norm_z, norm_Ax);
    float prim_res_scaled = r_prim / (prim_res_norm + OSQP_DIVISION_TOL);

    float dual_res_norm = hls::fmax(norm_q, hls::fmax(ATy_norm, Px_norm));
    float dual_res_scaled = r_dual / (dual_res_norm + OSQP_DIVISION_TOL);

    // Compute the scaling factor 
    float S = hls::sqrt(prim_res_scaled / hls::fmax(dual_res_scaled, OSQP_DIVISION_TOL)); 
    
    // Compute new base rho
    float rho_base_new = current_rho_base * S; 
    rho_base_new = clamp(rho_base_new, OSQP_RHO_MIN, OSQP_RHO_MAX); 
    
    return rho_base_new;
}

void apply_new_rho(float rho_base_new, float &current_rho_base, 
                   float *rho, float *y, float *rho_inv, int num_rows)
{
#pragma HLS INLINE off
    float S_eff = rho_base_new / current_rho_base; 

    // Update vectors (Optimized for Latency with Block Unrolling)
    for (int i = 0; i < num_rows; i += RESHAPE_FACTOR) 
    {
#pragma HLS PIPELINE II=1
        for (int j = 0; j < RESHAPE_FACTOR; ++j) 
        {
#pragma HLS UNROLL
            const int idx = i + j;
            if (idx < num_rows) 
            {
                const float rho_old = rho[idx];
                float rho_new = rho_old;

                if (rho_old > (OSQP_RHO_MIN * 1.1f)) 
                {
                    rho_new = rho_old * S_eff;
                    rho_new = clamp(rho_new, OSQP_RHO_MIN, OSQP_RHO_MAX * OSQP_RHO_EQ_OVER_RHO_INEQ);
                    rho[idx] = rho_new;
                }

                if (rho_new != rho_old)
                {
                    y[idx] = y[idx] * (rho_new / rho_old);
                }
                rho_inv[idx] = 1.0f / rho_new; 
            }
        }
    }
    current_rho_base = rho_base_new;
}

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
          int *status_out)
{
#pragma HLS INTERFACE m_axi port = A_row_idx offset = slave bundle = gmem0 depth = MAX_NNZ_WORDS
#pragma HLS INTERFACE m_axi port = A_col_ptr offset = slave bundle = gmem1 depth = MAX_COL_PTR
#pragma HLS INTERFACE m_axi port = A_values offset = slave bundle = gmem2 depth = MAX_NNZ_WORDS

#pragma HLS INTERFACE m_axi port = A_row_idx_tiled offset = slave bundle = gmem0 depth = MAX_NNZ_WORDS
#pragma HLS INTERFACE m_axi port = A_col_ptr_tiled offset = slave bundle = gmem1 depth = MAX_COL_PTR
#pragma HLS INTERFACE m_axi port = A_values_tiled offset = slave bundle = gmem2 depth = MAX_NNZ_WORDS
#pragma HLS INTERFACE m_axi port = A_tile_nnz_counts offset = slave bundle = gmem1 depth = MAX_TILES
#pragma HLS INTERFACE m_axi port = A_tile_nnz_offsets offset = slave bundle = gmem1 depth = MAX_TILES
#pragma HLS INTERFACE m_axi port = A_tile_col_offsets offset = slave bundle = gmem1 depth = MAX_TILES

#pragma HLS INTERFACE m_axi port = AT_row_idx_tiled offset = slave bundle = gmem0 depth = MAX_NNZ_WORDS
#pragma HLS INTERFACE m_axi port = AT_col_ptr_tiled offset = slave bundle = gmem1 depth = MAX_COL_PTR
#pragma HLS INTERFACE m_axi port = AT_values_tiled offset = slave bundle = gmem2 depth = MAX_NNZ_WORDS
#pragma HLS INTERFACE m_axi port = AT_tile_nnz_counts offset = slave bundle = gmem1 depth = MAX_TILES
#pragma HLS INTERFACE m_axi port = AT_tile_nnz_offsets offset = slave bundle = gmem1 depth = MAX_TILES
#pragma HLS INTERFACE m_axi port = AT_tile_col_offsets offset = slave bundle = gmem1 depth = MAX_TILES

#pragma HLS INTERFACE m_axi port = P_row_idx_tiled offset = slave bundle = gmem0 depth = MAX_NNZ_WORDS
#pragma HLS INTERFACE m_axi port = P_col_ptr_tiled offset = slave bundle = gmem1 depth = MAX_COL_PTR
#pragma HLS INTERFACE m_axi port = P_values_tiled offset = slave bundle = gmem2 depth = MAX_NNZ_WORDS
#pragma HLS INTERFACE m_axi port = P_tile_nnz_counts offset = slave bundle = gmem1 depth = MAX_TILES
#pragma HLS INTERFACE m_axi port = P_tile_nnz_offsets offset = slave bundle = gmem1 depth = MAX_TILES
#pragma HLS INTERFACE m_axi port = P_tile_col_offsets offset = slave bundle = gmem1 depth = MAX_TILES
#pragma HLS INTERFACE m_axi port = P_diag offset = slave bundle = gmem2 depth = MAX_COLS

#pragma HLS INTERFACE m_axi port = q_in offset = slave bundle = gmem3 depth = MAX_COLS
#pragma HLS INTERFACE m_axi port = l_in offset = slave bundle = gmem3 depth = MAX_ROWS
#pragma HLS INTERFACE m_axi port = u_in offset = slave bundle = gmem3 depth = MAX_ROWS

#pragma HLS INTERFACE m_axi port = rho_in offset = slave bundle = gmem3 depth = MAX_ROWS

#pragma HLS INTERFACE m_axi port = x_out offset = slave bundle = gmem3 depth = MAX_COLS
#pragma HLS INTERFACE m_axi port = y_out offset = slave bundle = gmem3 depth = MAX_ROWS

#pragma HLS INTERFACE s_axilite port = A_num_row_tiles bundle = control
#pragma HLS INTERFACE s_axilite port = A_num_col_tiles bundle = control
#pragma HLS INTERFACE s_axilite port = AT_num_row_tiles bundle = control
#pragma HLS INTERFACE s_axilite port = AT_num_col_tiles bundle = control
#pragma HLS INTERFACE s_axilite port = P_num_row_tiles bundle = control
#pragma HLS INTERFACE s_axilite port = P_num_col_tiles bundle = control

#pragma HLS INTERFACE s_axilite port = admm_num_iterations_out bundle = control
#pragma HLS INTERFACE s_axilite port = pcg_num_iterations_out bundle = control
#pragma HLS INTERFACE s_axilite port = adaptive_rho bundle = control
#pragma HLS INTERFACE s_axilite port = status_out bundle = control
#pragma HLS INTERFACE s_axilite port = r_prim_out bundle = control
#pragma HLS INTERFACE s_axilite port = r_dual_out bundle = control

#pragma HLS INTERFACE s_axilite port = eps_abs bundle = control
#pragma HLS INTERFACE s_axilite port = eps_rel bundle = control
#pragma HLS INTERFACE s_axilite port = pcg_tol_fraction bundle = control

#pragma HLS INTERFACE s_axilite port = num_rows bundle = control
#pragma HLS INTERFACE s_axilite port = num_cols bundle = control
#pragma HLS INTERFACE s_axilite port = A_nnz bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = alpha bundle = control
#pragma HLS INTERFACE s_axilite port = pcg_max_iterations bundle = control
#pragma HLS INTERFACE s_axilite port = admm_max_iterations bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

// Only one SpMV engine
#pragma HLS ALLOCATION function instances=spmv_csc limit=1


    float M_inv[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=M_inv type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=M_inv type=RAM_T2P impl=URAM

    float b[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=b type=cyclic factor=RESHAPE_FACTOR dim=1 //Necessary for K_p in PCG
#pragma HLS BIND_STORAGE variable=b type=RAM_T2P impl=URAM

    float x[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=x type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=x type=RAM_T2P impl=URAM

    float x_tilde[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=x_tilde type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=x_tilde type=RAM_T2P impl=URAM

    float y[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=y type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=y type=RAM_T2P impl=URAM

    float z[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=z type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=z type=RAM_T2P impl=URAM

    float z_tilde[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=z_tilde type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=z_tilde type=RAM_T2P impl=URAM

    float rho[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=rho type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=rho type=RAM_T2P impl=URAM

    float tmp1[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=tmp1 type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=tmp1 type=RAM_T2P impl=URAM

    float tmp2[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=tmp2 type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=tmp2 type=RAM_T2P impl=URAM

    float rho_inv[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=rho_inv type=cyclic factor=RESHAPE_FACTOR dim=1
#pragma HLS BIND_STORAGE variable=rho_inv type=RAM_T2P impl=URAM

    float l[MAX_SIZE];
#pragma HLS BIND_STORAGE variable=l type=RAM_T2P impl=URAM

    float u[MAX_SIZE];
#pragma HLS BIND_STORAGE variable=u type=RAM_T2P impl=URAM

    float q[MAX_SIZE];
#pragma HLS BIND_STORAGE variable=q type=RAM_T2P impl=URAM


    TiledMatrix mat_A = {
        A_num_row_tiles, A_num_col_tiles, 
        A_tile_nnz_counts, A_tile_nnz_offsets, A_tile_col_offsets, 
        A_row_idx_tiled, A_col_ptr_tiled, A_values_tiled
    };

    TiledMatrix mat_AT = {
        AT_num_row_tiles, AT_num_col_tiles, 
        AT_tile_nnz_counts, AT_tile_nnz_offsets, AT_tile_col_offsets, 
        AT_row_idx_tiled, AT_col_ptr_tiled, AT_values_tiled
    };

    TiledMatrix mat_P = {
        P_num_row_tiles, P_num_col_tiles, 
        P_tile_nnz_counts, P_tile_nnz_offsets, P_tile_col_offsets, 
        P_row_idx_tiled, P_col_ptr_tiled, P_values_tiled
    };

    float b_norm = 0.0f;

    float eps = 0.0f;
    float eps_old = 0.0f;

    float r_prim = 0.0f;
    float r_dual = 0.0f;

    float eps_prim = 0.0f;
    float eps_dual = 0.0f;

    int num_iterations = 0;
    int num_pcg_iterations = 0;
    int total_pcg_iterations = 0;
    int zero_pcg_iters = 0;

    float norm_q = 0.0f;
    float norm_z = 0.0f;

    float ATy_norm = 0.0f;
    float Px_norm = 0.0f;

    float Ax_norm = 0.0f; 

    for (int i = 0; i < num_cols; i++)
    {
#pragma HLS PIPELINE II = 1
        x[i] = 0.0f;      // initial x
        x_tilde[i] = 0.0f; // initial guess for PCG (warm-started across ADMM iterations)

        float q_val = q_in[i];
        q[i] = q_val; // copy q from DDR to BRAM

        float abs_q = hls::fabs(q_val); // inf_norm of q
    }
    norm_q = inf_norm(q, num_cols);

    for (int i = 0; i < num_rows; i++)
    {
#pragma HLS PIPELINE II = 1
        y[i] = 0.0f;
        z[i] = 0.0f;
        rho[i] = rho_in[i];
        rho_inv[i] = 1.0f / rho[i];
        l[i] = l_in[i];
        u[i] = u_in[i];

    }

    update_preconditioner(P_diag, A_values, A_row_idx, A_col_ptr, rho, sigma, M_inv, num_cols); // Compute diagonal preconditioner M_inv

    float one_minus_alpha = 1.0f - alpha;

    float current_rho_base = OSQP_RHO;

    do
    {   
        // Adaptive Rho Update every OSQP_ADAPTIVE_RHO_INTERVAL iterations
        if (adaptive_rho && (num_iterations > 0 && num_iterations % OSQP_ADAPTIVE_RHO_INTERVAL == 0)) 
        {
            // Calculate the proposed new rho
            float rho_new = estimate_new_rho(r_prim, r_dual, 
                                             norm_z, Ax_norm, 
                                             norm_q, ATy_norm, Px_norm, 
                                             current_rho_base);
            
            // Check if the proposed rho is outside the tolerance window
            float upper_bound = current_rho_base * OSQP_ADAPTIVE_RHO_TOLERANCE;
            float lower_bound = current_rho_base / OSQP_ADAPTIVE_RHO_TOLERANCE;

            if ((rho_new > upper_bound) || (rho_new < lower_bound)) 
            {
                apply_new_rho(rho_new, current_rho_base, rho, y, rho_inv, num_rows);
                update_preconditioner(P_diag, A_values, A_row_idx, A_col_ptr, rho, sigma, M_inv, num_cols); 
            }
        }

        update_b(num_rows, num_cols, sigma, x, q, mat_AT, rho, y, z, b, tmp1, tmp2, &b_norm); // Compute vector b for PCG and its norm for eps calculation
        
        update_eps(b_norm, r_prim, r_dual, eps, eps_old, num_iterations, num_pcg_iterations, zero_pcg_iters, pcg_tol_fraction);

        pcg(num_rows, num_cols, mat_A, mat_AT, mat_P, M_inv, rho, sigma, eps, x_tilde, b, tmp1, tmp2, &num_pcg_iterations, pcg_max_iterations);

        total_pcg_iterations += num_pcg_iterations;


        spmv_csc_tiled(num_rows, num_cols, mat_A, x_tilde, tmp1); // z_tilde = A*x_tilde
        
        for (int i = 0; i < num_cols; i++)
        {
#pragma HLS PIPELINE II = 1
            x[i] = alpha * x_tilde[i] + one_minus_alpha * x[i]; // x = alpha * x_tilde + (1 - alpha) * x 
        }

        for (int i = 0; i < num_rows; i++)
        {
#pragma HLS PIPELINE II = 1
            float z_prev = z[i];
            tmp2[i] = z_prev; // Store z_prev for dual update later
            float v = alpha * tmp1[i] + one_minus_alpha * z_prev; // v = alpha * z_tilde + (1 - alpha) * z_prev

            float new_z = clamp(v + y[i] * rho_inv[i], l[i], u[i]);
            z[i] = new_z; // z = clamp(v, l, u)
            y[i] = y[i] + rho[i] * (v - new_z); // y = y + rho*(v - z)
        }
        norm_z = inf_norm(z, num_rows);


        // r_prim, note: inf_norm replaced inside loop to reduce an extra loop
        spmv_csc_tiled(num_rows, num_cols, mat_A, x, tmp1); // tmp1 = A*x
        Ax_norm = inf_norm(tmp1, num_rows);
        for (int i = 0; i < num_rows; i += RESHAPE_FACTOR)
        {
#pragma HLS PIPELINE II = 1
            for (int j = 0; j < RESHAPE_FACTOR; ++j)
            {
#pragma HLS UNROLL
                const int idx = i + j;
                if (idx < num_rows)
                {
                    tmp1[idx] = tmp1[idx] - z[idx]; // tmp1 now holds (Ax - z)
                }
            }
        }
        r_prim = inf_norm(tmp1, num_rows);

        //r_dual calculation ||P*x + q + AT*y||inf
        spmv_csc_tiled(num_cols, num_rows, mat_AT, y, tmp1); // tmp1 = AT*y
        spmv_csc_tiled(num_cols, num_cols, mat_P, x, tmp2);    // tmp2 = P*x
        ATy_norm = inf_norm(tmp1, num_cols); // for convergence check and adaptive rho
        Px_norm = inf_norm(tmp2, num_cols);
        for (int i = 0; i < num_cols; i += RESHAPE_FACTOR)
        {
#pragma HLS PIPELINE II=1
            for (int j = 0; j < RESHAPE_FACTOR; ++j)
            {
#pragma HLS UNROLL
                const int idx = i + j;
                if (idx < num_cols)
                {
                    tmp2[idx] = tmp2[idx] + q[idx] + tmp1[idx]; 
                }
            }
        }
        r_dual = inf_norm(tmp2, num_cols);

        // Check for convergence every OSQP_CHECK_TERMINATION iterations
        if (num_iterations > 0 && num_iterations % OSQP_CHECK_TERMINATION == 0) {
            float max_prim = hls::fmax(Ax_norm, norm_z);
            float max_dual = hls::fmax(Px_norm, hls::fmax(norm_q, ATy_norm));

            eps_prim = eps_abs + eps_rel*max_prim;
            eps_dual = eps_abs + eps_rel*max_dual;

            if (r_prim <= eps_prim && r_dual <= eps_dual) {
                break; // Converged
            }
        }

        num_iterations++;

    } while (num_iterations < admm_max_iterations);
     
    // Write outputs back to DDR
    *admm_num_iterations_out = num_iterations;
    *pcg_num_iterations_out = total_pcg_iterations;
    *r_prim_out = r_prim;
    *r_dual_out = r_dual;
    *status_out = (num_iterations == admm_max_iterations) ? 0 : 1; // 0 if max iterations reached, 1 if converged


    // Write final solution back to DDR
    for (int i = 0; i < num_cols; i++)
    {
#pragma HLS PIPELINE II = 1
        x_out[i] = x[i];
    }

    for (int i = 0; i < num_rows; i++)
    {
#pragma HLS PIPELINE II = 1
        y_out[i] = y[i];
    }
}