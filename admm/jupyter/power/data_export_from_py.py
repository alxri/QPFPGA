import numpy as np
import scipy.sparse as sp
import os

def print_vector(name, vec):
    """Mimics the C++ print_vector formatting."""
    print(f"=== {name} (Size: {len(vec)}) ===")
    if len(vec) == 0:
        print("[Empty or Not Found]\n")
        return
    for i, val in enumerate(vec):
        print(f"{val}", end=" ")
        if (i + 1) % 10 == 0:
            print()
    if len(vec) % 10 != 0:
        print()
    print("\n")

data_dir = "./data/"
out_dir = "./data_bin/"
os.makedirs(out_dir, exist_ok=True)

# Define OSQP hardware-safe infinity
OSQP_INFTY = np.float32(1e17)

# 1. Load original data
P_sparse = sp.load_npz(os.path.join(data_dir, "P.npz")).tocsc()
A_sparse = sp.load_npz(os.path.join(data_dir, "A.npz")).tocsc()
q = np.load(os.path.join(data_dir, "q.npy")).astype(np.float32)
l = np.load(os.path.join(data_dir, "l.npy")).astype(np.float32)
u = np.load(os.path.join(data_dir, "u.npy")).astype(np.float32)

# --- CLAMP INFINITY VALUES ---
# Replace positive infinity with OSQP_INFTY
l[np.isposinf(l)] = OSQP_INFTY
u[np.isposinf(u)] = OSQP_INFTY

# Replace negative infinity with -OSQP_INFTY
l[np.isneginf(l)] = -OSQP_INFTY
u[np.isneginf(u)] = -OSQP_INFTY
# -----------------------------

# Extract P matrix components
P_data = P_sparse.data.astype(np.float32)
P_indices = P_sparse.indices.astype(np.int32)
P_indptr = P_sparse.indptr.astype(np.int32)
P_diag = P_sparse.diagonal().astype(np.float32)

# Extract A matrix components
A_data = A_sparse.data.astype(np.float32)
A_indices = A_sparse.indices.astype(np.int32)
A_indptr = A_sparse.indptr.astype(np.int32)

# 2. Print everything BEFORE exporting
print_vector("Vector l", l)
print_vector("Vector q", q)
print_vector("Vector u", u)
print_vector("P_diag", P_diag)

print_vector("A_data", A_data)
print_vector("A_indices", A_indices)
print_vector("A_indptr", A_indptr)

print_vector("P_data", P_data)
print_vector("P_indices", P_indices)
print_vector("P_indptr", P_indptr)

# 3. Export to binary files
q.tofile(os.path.join(out_dir, "q.bin"))
l.tofile(os.path.join(out_dir, "l.bin"))
u.tofile(os.path.join(out_dir, "u.bin"))

P_data.tofile(os.path.join(out_dir, "P_data.bin"))
P_indices.tofile(os.path.join(out_dir, "P_indices.bin"))
P_indptr.tofile(os.path.join(out_dir, "P_indptr.bin"))
P_diag.tofile(os.path.join(out_dir, "P_diag.bin"))

A_data.tofile(os.path.join(out_dir, "A_data.bin"))
A_indices.tofile(os.path.join(out_dir, "A_indices.bin"))
A_indptr.tofile(os.path.join(out_dir, "A_indptr.bin"))

# 4. Dimension checks and outputs
print(f"NUM_ROWS: {A_sparse.shape[0]}, NUM_COLS: {A_sparse.shape[1]}, A_NNZ: {A_sparse.nnz}")

A = sp.load_npz("data/A.npz")
A_csc = A.tocsc()
A_csr = A.tocsr()

print("CSC nnz match:", (A_csc != A_sparse).nnz == 0)
print("CSR nnz match:", (A_csr != A_sparse).nnz == 0)