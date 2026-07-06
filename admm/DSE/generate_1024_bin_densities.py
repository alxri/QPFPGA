#!/usr/bin/env python3
"""
Generate three 1024x1024 datasets at different densities and export in .bin format
(compatible with the C++ testbench / existing binary loaders).

Creates directories:
 DSE/1024x1024_data_bin_dse/low/
 DSE/1024x1024_data_bin_dse/med/
 DSE/1024x1024_data_bin_dse/high/

Each contains: q.bin, l.bin, u.bin, P_diag.bin,
                 A_data.bin, A_indices.bin, A_indptr.bin,
                 P_data.bin, P_indices.bin, P_indptr.bin,
                 x_true.bin

Usage: python3 DSE/generate_1024_bin_densities.py
"""
import os
import numpy as np
import scipy.sparse as sp

NUM_ROWS = 1024
NUM_COLS = 1024
OSQP_INFTY = np.float32(1e17)
SEED = 123

OUT_ROOT = os.path.join(os.path.dirname(__file__), '1024x1024_data_bin_dse')
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
    # Build A column-by-column and construct correct CSC arrays
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

    # Build z_true = A * x_true
    z_true = np.zeros(NUM_ROWS, dtype=np.float32)
    idx = 0
    for c in range(NUM_COLS):
        for off in range(A_col_ptr[c], A_col_ptr[c + 1]):
            r = A_indices[off]
            v = A_data[off]
            z_true[r] += v * x_true[c]
            idx += 1

    # bounds l,u
    l = np.zeros(NUM_ROWS, dtype=np.float32)
    u = np.zeros(NUM_ROWS, dtype=np.float32)
    constraint_types = np.random.randint(0, 3, NUM_ROWS)
    slack1 = np.random.uniform(0.5, 5.0, NUM_ROWS).astype(np.float32)
    slack2 = np.random.uniform(0.5, 5.0, NUM_ROWS).astype(np.float32)
    for i in range(NUM_ROWS):
        if constraint_types[i] == 0:
            l[i] = z_true[i]
            u[i] = z_true[i]
        elif constraint_types[i] == 1:
            l[i] = z_true[i] - slack1[i]
            u[i] = z_true[i] + slack2[i]
        else:
            l[i] = -OSQP_INFTY
            u[i] = OSQP_INFTY

    # Export binaries
    q.tofile(os.path.join(out_dir, 'q.bin'))
    l.tofile(os.path.join(out_dir, 'l.bin'))
    u.tofile(os.path.join(out_dir, 'u.bin'))
    P_diag.tofile(os.path.join(out_dir, 'P_diag.bin'))

    A_data.tofile(os.path.join(out_dir, 'A_data.bin'))
    A_indices.tofile(os.path.join(out_dir, 'A_indices.bin'))
    A_col_ptr.tofile(os.path.join(out_dir, 'A_indptr.bin'))

    P_data.tofile(os.path.join(out_dir, 'P_data.bin'))
    P_indices.tofile(os.path.join(out_dir, 'P_indices.bin'))
    P_indptr.tofile(os.path.join(out_dir, 'P_indptr.bin'))

    x_true.tofile(os.path.join(out_dir, 'x_true.bin'))

    print('Wrote', out_dir)

print('All densities generated.')
