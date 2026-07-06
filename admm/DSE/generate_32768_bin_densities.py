#!/usr/bin/env python3
"""
Generate three 32768x32768 datasets at different densities and export in .bin format
(compatible with the C++ testbench / existing binary loaders).

Creates directories:
 DSE/32768x32768_data_bin_dse/low/
 DSE/32768x32768_data_bin_dse/med/
 DSE/32768x32768_data_bin_dse/high/

Each contains: q.bin, l.bin, u.bin, P_diag.bin,
                 A_data.bin, A_indices.bin, A_indptr.bin,
                 P_data.bin, P_indices.bin, P_indptr.bin,
                 x_true.bin
"""
import os
import numpy as np
import scipy.sparse as sp

NUM_ROWS = 32768
NUM_COLS = 32768
OSQP_INFTY = np.float32(1e17)
SEED = 123

# Output directory targeting the 32768 dataset
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '32768x32768_data_bin_dse')

DENSITIES = {
    'low': 1,    # average 1 nnz per column
    'med': 8,    # average 8 nnz per column
    'high': 32,  # average 32 nnz per column
}

np.random.seed(SEED)

os.makedirs(OUT_ROOT, exist_ok=True)

for name, nnz_per_col in DENSITIES.items():
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f'Generating density {name} ({nnz_per_col} nnz/col) -> {out_dir}')

    # Ground truth x* and diagonal P
    x_true = np.random.uniform(-2.0, 2.0, NUM_COLS).astype(np.float32)
    P_diag = np.random.uniform(1.0, 3.0, NUM_COLS).astype(np.float32)
    q = -P_diag * x_true

    # Build P sparse (diagonal)
    P_sparse = sp.diags(P_diag).tocsc()
    P_data = P_sparse.data.astype(np.float32)
    P_indices = P_sparse.indices.astype(np.int32)
    P_indptr = P_sparse.indptr.astype(np.int32)

    # Build A as a CSC with probabilistic nnz per column
    rows = []
    vals = []
    col_ptr = [0]
    for c in range(NUM_COLS):
        k = max(1, int(np.random.poisson(nnz_per_col)))
        # sample k distinct row indices
        r = np.random.choice(NUM_ROWS, size=k, replace=False)
        v = np.random.uniform(-1.0, 1.0, size=k).astype(np.float32)
        # avoid tiny values
        v[np.abs(v) < 1e-3] = 1e-3
        rows.extend(r.tolist())
        vals.extend(v.tolist())
        col_ptr.append(len(rows))

    A_col_ptr = np.array(col_ptr, dtype=np.int32)
    A_indices = np.array(rows, dtype=np.int32)
    A_data = np.array(vals, dtype=np.float32)
    A_sp = sp.csc_matrix((A_data, A_indices, A_col_ptr), shape=(NUM_ROWS, NUM_COLS))

    # ------------------------------------------------------------------
    # PERFORMANCE FIX: Replaced the slow nested Python loops with Scipy's 
    # highly optimized native dot product to calculate z_true
    # ------------------------------------------------------------------
    z_true = A_sp.dot(x_true).astype(np.float32)

    # bounds l,u
    l = np.zeros(NUM_ROWS, dtype=np.float32)
    u = np.zeros(NUM_ROWS, dtype=np.float32)
    constraint_types = np.random.randint(0, 3, NUM_ROWS)
    slack1 = np.random.uniform(0.5, 5.0, NUM_ROWS).astype(np.float32)
    slack2 = np.random.uniform(0.5, 5.0, NUM_ROWS).astype(np.float32)
    for i in range(NUM_ROWS):
        ct = constraint_types[i]
        if ct == 0:
            # equality constraint
            l[i] = z_true[i]
            u[i] = z_true[i]
        elif ct == 1:
            # lower-bound constraint: l = z_true - slack, u = +inf
            l[i] = z_true[i] - slack1[i]
            u[i] = OSQP_INFTY
        else:
            # upper-bound constraint: l = -inf, u = z_true + slack
            l[i] = -OSQP_INFTY
            u[i] = z_true[i] + slack2[i]

    # --- write binary files compatible with C++ loader ---
    def write_bin(arr, filename, dtype=None):
        path = os.path.join(out_dir, filename)
        if dtype is not None:
            arr = arr.astype(dtype)
        arr.tofile(path)

    write_bin(q, 'q.bin', np.float32)
    write_bin(l, 'l.bin', np.float32)
    write_bin(u, 'u.bin', np.float32)
    write_bin(P_diag, 'P_diag.bin', np.float32)

    write_bin(A_data, 'A_data.bin', np.float32)
    write_bin(A_indices, 'A_indices.bin', np.int32)
    write_bin(A_col_ptr, 'A_indptr.bin', np.int32)

    write_bin(P_data, 'P_data.bin', np.float32)
    write_bin(P_indices, 'P_indices.bin', np.int32)
    write_bin(P_indptr, 'P_indptr.bin', np.int32)

    write_bin(x_true, 'x_true.bin', np.float32)

    print(f'Wrote binaries to {out_dir}')