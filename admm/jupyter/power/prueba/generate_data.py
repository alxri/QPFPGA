import os
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NUM_ROWS = 1024
NUM_COLS = 1024
OUT_DIR = "./data_bin/"

OSQP_INFTY = np.float32(1e17)

# Ensure the output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Generating {NUM_ROWS}x{NUM_COLS} procedural data...")

# ---------------------------------------------------------------------
# Data Generation (Identical logic to the C++ testbench)
# ---------------------------------------------------------------------
np.random.seed(123)

# 1. Generate Ground Truth x* and Matrix P
x_true = np.random.uniform(-2.0, 2.0, NUM_COLS).astype(np.float32)
P_diag = np.random.uniform(1.0, 3.0, NUM_COLS).astype(np.float32)
q = -P_diag * x_true

P_sparse = sp.diags(P_diag).tocsc()
P_data = P_sparse.data.astype(np.float32)
P_indices = P_sparse.indices.astype(np.int32)
P_indptr = P_sparse.indptr.astype(np.int32)

# 2. Generate Matrix A
A_row_idx = np.random.randint(0, NUM_ROWS, NUM_COLS, dtype=np.int32)
A_vals = np.random.uniform(-1.0, 1.0, NUM_COLS).astype(np.float32)
A_vals[np.abs(A_vals) < 1e-3] = 1e-3
A_col_ptr = np.arange(NUM_COLS + 1, dtype=np.int32)

# Build z* (A * x_true)
z_true = np.zeros(NUM_ROWS, dtype=np.float32)
for c in range(NUM_COLS):
    z_true[A_row_idx[c]] += A_vals[c] * x_true[c]

# 3. Generate Bounds based on z*
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

# ---------------------------------------------------------------------
# Export to Binary
# ---------------------------------------------------------------------
print(f"Exporting files to {OUT_DIR}...")

# Vector exports
q.tofile(os.path.join(OUT_DIR, "q.bin"))
l.tofile(os.path.join(OUT_DIR, "l.bin"))
u.tofile(os.path.join(OUT_DIR, "u.bin"))
P_diag.tofile(os.path.join(OUT_DIR, "P_diag.bin"))

# Matrix A components
A_vals.tofile(os.path.join(OUT_DIR, "A_data.bin"))
A_row_idx.tofile(os.path.join(OUT_DIR, "A_indices.bin"))
A_col_ptr.tofile(os.path.join(OUT_DIR, "A_indptr.bin"))

# Matrix P components
P_data.tofile(os.path.join(OUT_DIR, "P_data.bin"))
P_indices.tofile(os.path.join(OUT_DIR, "P_indices.bin"))
P_indptr.tofile(os.path.join(OUT_DIR, "P_indptr.bin"))

# Export Ground Truth so you can verify it in other scripts if needed
x_true.tofile(os.path.join(OUT_DIR, "x_true.bin"))

print("Export complete! You can now run your C++ testbench using these files.")