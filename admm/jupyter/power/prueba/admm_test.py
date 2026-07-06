import os
import numpy as np
import scipy.sparse as sp
import time
import struct
import math
from pynq import Overlay, allocate, MMIO

# ---------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------
PACK_SIZE = 16
TILE_ROWS = 8192
TILE_COLS = 8192

ADMM_IP_CONTROL_BASE   = 0xA0000000 
ADMM_IP_CONTROL_R_BASE = 0xA0010000
ADDR_RANGE             = 0x20000

OSQP_RHO = 1.0
OSQP_RHO_EQ_OVER_RHO_INEQ = 100
OSQP_RHO_TOL = 0.01
OSQP_INFTY = np.float32(1e17)
OSQP_RHO_MAX = 1e6
OSQP_RHO_MIN = 1e-6

NUM_ROWS = 1024
NUM_COLS = 1024
DATA_DIR = "./data_bin/"

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def float_to_uint(f_val):
    return struct.unpack('<I', struct.pack('<f', float(f_val)))[0]

def uint_to_float(u_val):
    return struct.unpack('<f', struct.pack('<I', int(u_val)))[0]

def write_64bit_address(ip, base_offset, address):
    ip.write(base_offset, address & 0xFFFFFFFF)
    ip.write(base_offset + 0x04, (address >> 32) & 0xFFFFFFFF)

def ceil_div(a, b):
    return (a + b - 1) // b

def flush_denormals(arr):
    mask = (np.abs(arr) > 0.0) & (np.abs(arr) < 1e-15)
    arr[mask] = 0.0
    return arr

def allocate_and_copy(np_array, dtype):
    buf = allocate(shape=np_array.shape, dtype=dtype, cacheable=False)
    buf[:] = np_array[:]
    return buf

# ---------------------------------------------------------------------
# Matrix Tiling Engine
# ---------------------------------------------------------------------
def build_tiled_csc(sp_mat, tile_rows, tile_cols, pack_size=16):
    global_rows, global_cols = sp_mat.shape
    num_row_tiles = ceil_div(global_rows, tile_rows)
    num_col_tiles = ceil_div(global_cols, tile_cols)
    num_tiles = num_row_tiles * num_col_tiles

    g_col_ptr, g_row_idx, g_values = sp_mat.indptr, sp_mat.indices, sp_mat.data

    tiles_rows = [[[] for _ in range(tile_cols)] for _ in range(num_tiles)]
    tiles_vals = [[[] for _ in range(tile_cols)] for _ in range(num_tiles)]

    for c in range(global_cols):
        tc, local_c = c // tile_cols, c % tile_cols
        for idx in range(g_col_ptr[c], g_col_ptr[c+1]):
            r, v = g_row_idx[idx], g_values[idx]
            tr, local_r = r // tile_rows, r % tile_rows
            tile_idx = tr * num_col_tiles + tc
            tiles_rows[tile_idx][local_c].append(local_r)
            tiles_vals[tile_idx][local_c].append(v)

    tile_nnz_counts  = np.zeros(num_tiles, dtype=np.int32)
    tile_nnz_offsets = np.zeros(num_tiles, dtype=np.int32)
    tile_col_offsets = np.zeros(num_tiles, dtype=np.int32)
    col_ptr_tiled = []
    row_idx_tiled = []
    values_tiled  = []
    
    nnz_word_cursor = 0
    for tile_idx in range(num_tiles):
        tile_col_offsets[tile_idx] = len(col_ptr_tiled)
        
        local_col_ptr = [0] * (tile_cols + 1)
        tile_nnz = 0
        for c in range(tile_cols):
            tile_nnz += len(tiles_rows[tile_idx][c])
            local_col_ptr[c+1] = tile_nnz

        tile_nnz_counts[tile_idx] = tile_nnz
        tile_nnz_offsets[tile_idx] = nnz_word_cursor
        col_ptr_tiled.extend(local_col_ptr)

        t_rows, t_vals = [], []
        for c in range(tile_cols):
            t_rows.extend(tiles_rows[tile_idx][c])
            t_vals.extend(tiles_vals[tile_idx][c])

        words = ceil_div(tile_nnz, pack_size)
        for w in range(words):
            row_word = np.zeros(pack_size, dtype=np.int32)
            val_word = np.zeros(pack_size, dtype=np.float32)
            for lane in range(pack_size):
                idx = w * pack_size + lane
                if idx < tile_nnz:
                    row_word[lane] = t_rows[idx]
                    val_word[lane] = t_vals[idx]
            row_idx_tiled.append(row_word)
            values_tiled.append(val_word)
        nnz_word_cursor += words

    if len(row_idx_tiled) == 0:
        row_idx_tiled.append(np.zeros(pack_size, dtype=np.int32))
        values_tiled.append(np.zeros(pack_size, dtype=np.float32))

    return (num_row_tiles, num_col_tiles,
            np.array(tile_nnz_counts, dtype=np.int32),
            np.array(tile_nnz_offsets, dtype=np.int32),
            np.array(tile_col_offsets, dtype=np.int32),
            np.array(col_ptr_tiled, dtype=np.int32),
            np.array(row_idx_tiled, dtype=np.int32),
            np.array(values_tiled, dtype=np.float32))

# ---------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------
print("Loading overlay from /home/xilinx/admm/admm_fixed_tiles.bit...")
overlay = Overlay('/home/xilinx/admm/admm_fixed_tiles.bit')
print("Overlay loaded!\n")

control_ip   = MMIO(ADMM_IP_CONTROL_BASE, ADDR_RANGE)
control_r_ip = MMIO(ADMM_IP_CONTROL_R_BASE, ADDR_RANGE)

# 1. Load Data from Binaries
print(f"Loading {NUM_ROWS}x{NUM_COLS} data from {DATA_DIR}...")
x_true    = np.fromfile(os.path.join(DATA_DIR, "x_true.bin"), dtype=np.float32)
P_diag    = np.fromfile(os.path.join(DATA_DIR, "P_diag.bin"), dtype=np.float32)
q         = np.fromfile(os.path.join(DATA_DIR, "q.bin"), dtype=np.float32)
A_row_idx = np.fromfile(os.path.join(DATA_DIR, "A_indices.bin"), dtype=np.int32)
A_vals    = np.fromfile(os.path.join(DATA_DIR, "A_data.bin"), dtype=np.float32)
A_col_ptr = np.fromfile(os.path.join(DATA_DIR, "A_indptr.bin"), dtype=np.int32)
l         = np.fromfile(os.path.join(DATA_DIR, "l.bin"), dtype=np.float32)
u         = np.fromfile(os.path.join(DATA_DIR, "u.bin"), dtype=np.float32)

A_sparse = sp.csc_matrix((A_vals, A_row_idx, A_col_ptr), shape=(NUM_ROWS, NUM_COLS))
AT_sparse = A_sparse.T.tocsc()
P_sparse = sp.diags(P_diag).tocsc()

# 2. Scaling
print("Applying Scaling...\n")
def apply_scaling(P_d, A_sp, q_vec, l_vec, u_vec, iters=10):
    n, m = len(P_d), A_sp.shape[0]
    E = np.ones(n, dtype=np.float32)
    D = np.ones(m, dtype=np.float32)
    P_scaled, A_scaled = P_d.copy(), A_sp.copy()

    for _ in range(iters):
        A_abs = abs(A_scaled)
        A_col_norm = A_abs.max(axis=0).toarray().flatten()
        x_norm = np.maximum(np.maximum(np.abs(P_scaled), A_col_norm), 1e-4)
        z_norm = np.maximum(A_abs.max(axis=1).toarray().flatten(), 1e-4)

        E_new, D_new = 1.0 / np.sqrt(x_norm), 1.0 / np.sqrt(z_norm)
        E *= E_new
        D *= D_new

        P_scaled = P_scaled * (E_new ** 2)
        A_scaled = sp.diags(D_new).dot(A_scaled).dot(sp.diags(E_new)).tocsc()

    q_scaled = q_vec * E
    
    # Avoid true IEEE infinity multiplication errors
    l_scaled = np.where(l_vec <= -OSQP_INFTY * 0.9, -OSQP_INFTY, l_vec * D)
    u_scaled = np.where(u_vec >=  OSQP_INFTY * 0.9,  OSQP_INFTY, u_vec * D)

    c = 1.0 / np.maximum(np.max(np.abs(P_scaled)), np.max(np.abs(q_scaled)))
    c = max(c, 1e-4)
    return P_scaled * c, A_scaled, q_scaled * c, l_scaled, u_scaled, E, D, c

(P_diag_s, A_sp_s, q_s, l_s, u_s, E_scale, D_scale, c_scale) = apply_scaling(P_diag, A_sparse, q, l, u)
AT_sp_s = A_sp_s.T.tocsc()
P_sp_s = sp.diags(P_diag_s).tocsc()

# Flush denormals
A_sp_s.data = flush_denormals(A_sp_s.data)
P_sp_s.data = flush_denormals(P_sp_s.data)
P_diag_s = flush_denormals(P_diag_s)
q_s = flush_denormals(q_s)

# Create Rho
rho_in = np.full(NUM_ROWS, OSQP_RHO, dtype=np.float32)
fin_l = np.abs(l_s) < (OSQP_INFTY * 0.9)
fin_u = np.abs(u_s) < (OSQP_INFTY * 0.9)
rho_in[~fin_l & ~fin_u] = OSQP_RHO_MIN
rho_in[fin_l & fin_u & ((u_s - l_s) < 0.01)] = OSQP_RHO * OSQP_RHO_EQ_OVER_RHO_INEQ
rho_in = np.clip(rho_in, OSQP_RHO_MIN, OSQP_RHO_MAX)

# 3. Allocating Tiled Matrices
print("Slicing and Allocating Tiled Matrices... (This may take a moment)")
buffers = []

# Regular Matrix A
A_nnz = A_sp_s.nnz
A_words = ceil_div(A_nnz, PACK_SIZE)
A_row_words = np.zeros((A_words, PACK_SIZE), dtype=np.int32)
A_val_words = np.zeros((A_words, PACK_SIZE), dtype=np.float32)
A_row_words.reshape(-1)[:A_nnz] = A_sp_s.indices
A_val_words.reshape(-1)[:A_nnz] = A_sp_s.data

A_r_hw = allocate_and_copy(A_row_words, np.int32)
A_v_hw = allocate_and_copy(A_val_words, np.float32)
A_c_hw = allocate_and_copy(A_sp_s.indptr, np.int32)
buffers.extend([A_r_hw, A_v_hw, A_c_hw])

for name, sp_mat in [("A", A_sp_s), ("AT", AT_sp_s), ("P", P_sp_s)]:
    rtiles, ctiles, counts, noff, coff, cptr, ridx, vals = build_tiled_csc(sp_mat, TILE_ROWS, TILE_COLS, PACK_SIZE)
    globals()[f"{name}_rt"] = rtiles
    globals()[f"{name}_ct"] = ctiles
    globals()[f"{name}_cnt_hw"]  = allocate_and_copy(counts, np.int32)
    globals()[f"{name}_noff_hw"] = allocate_and_copy(noff, np.int32)
    globals()[f"{name}_coff_hw"] = allocate_and_copy(coff, np.int32)
    globals()[f"{name}_cptr_hw"] = allocate_and_copy(cptr, np.int32)
    globals()[f"{name}_ridx_hw"] = allocate_and_copy(ridx, np.int32)
    globals()[f"{name}_vals_hw"] = allocate_and_copy(vals, np.float32)
    buffers.extend([globals()[f"{name}_{x}"] for x in ["cnt_hw", "noff_hw", "coff_hw", "cptr_hw", "ridx_hw", "vals_hw"]])

Pdiag_hw = allocate_and_copy(P_diag_s, np.float32)
l_hw     = allocate_and_copy(l_s, np.float32)
u_hw     = allocate_and_copy(u_s, np.float32)
q_hw     = allocate_and_copy(q_s, np.float32)
rho_hw   = allocate_and_copy(rho_in, np.float32)
x_hw     = allocate(shape=(NUM_COLS,), dtype=np.float32, cacheable=False)
y_hw     = allocate(shape=(NUM_ROWS,), dtype=np.float32, cacheable=False)
buffers.extend([Pdiag_hw, l_hw, u_hw, q_hw, rho_hw, x_hw, y_hw])

# 4. Hardware Run
sigma, alpha = 1e-2, 1.8
eps_abs, eps_rel, pcg_tol_frac = 1e-3, 1e-3, 1.0
admm_max, pcg_max = 2000, 5

results = []

for ad_rho in [0, 1]:
    print(f"=== HW Run (adaptive_rho={ad_rho}) ===")
    
    x_hw[:] = 0.0
    y_hw[:] = 0.0

    control_ip.write(0x10, NUM_ROWS)
    control_ip.write(0x18, NUM_COLS)
    control_ip.write(0x20, A_nnz)
    control_ip.write(0x28, A_rt);  control_ip.write(0x30, A_ct)
    control_ip.write(0x38, AT_rt); control_ip.write(0x40, AT_ct)
    control_ip.write(0x48, P_rt);  control_ip.write(0x50, P_ct)
    control_ip.write(0x58, float_to_uint(sigma))
    control_ip.write(0x60, float_to_uint(alpha))
    control_ip.write(0x68, admm_max)
    control_ip.write(0x70, pcg_max)
    control_ip.write(0x78, ad_rho)
    control_ip.write(0x80, float_to_uint(eps_abs))
    control_ip.write(0x88, float_to_uint(eps_rel))
    control_ip.write(0x90, float_to_uint(pcg_tol_frac))

    write_64bit_address(control_r_ip, 0x010, A_r_hw.device_address)
    write_64bit_address(control_r_ip, 0x01c, A_c_hw.device_address)
    write_64bit_address(control_r_ip, 0x028, A_v_hw.device_address)
    
    write_64bit_address(control_r_ip, 0x034, A_cnt_hw.device_address);  write_64bit_address(control_r_ip, 0x040, A_noff_hw.device_address)
    write_64bit_address(control_r_ip, 0x04c, A_coff_hw.device_address); write_64bit_address(control_r_ip, 0x058, A_ridx_hw.device_address)
    write_64bit_address(control_r_ip, 0x064, A_cptr_hw.device_address); write_64bit_address(control_r_ip, 0x070, A_vals_hw.device_address)
    
    write_64bit_address(control_r_ip, 0x07c, AT_cnt_hw.device_address);  write_64bit_address(control_r_ip, 0x088, AT_noff_hw.device_address)
    write_64bit_address(control_r_ip, 0x094, AT_coff_hw.device_address); write_64bit_address(control_r_ip, 0x0a0, AT_ridx_hw.device_address)
    write_64bit_address(control_r_ip, 0x0ac, AT_cptr_hw.device_address); write_64bit_address(control_r_ip, 0x0b8, AT_vals_hw.device_address)
    
    write_64bit_address(control_r_ip, 0x0c4, P_cnt_hw.device_address);  write_64bit_address(control_r_ip, 0x0d0, P_noff_hw.device_address)
    write_64bit_address(control_r_ip, 0x0dc, P_coff_hw.device_address); write_64bit_address(control_r_ip, 0x0e8, P_ridx_hw.device_address)
    write_64bit_address(control_r_ip, 0x0f4, P_cptr_hw.device_address); write_64bit_address(control_r_ip, 0x100, P_vals_hw.device_address)
    
    write_64bit_address(control_r_ip, 0x10c, Pdiag_hw.device_address)
    write_64bit_address(control_r_ip, 0x118, l_hw.device_address)
    write_64bit_address(control_r_ip, 0x124, u_hw.device_address)
    write_64bit_address(control_r_ip, 0x130, q_hw.device_address)
    write_64bit_address(control_r_ip, 0x13c, rho_hw.device_address)
    write_64bit_address(control_r_ip, 0x148, x_hw.device_address)
    write_64bit_address(control_r_ip, 0x154, y_hw.device_address)

    hw_start = time.time()
    control_ip.write(0x00, 0x01)
    
    # Fast Python polling loop
    while (control_ip.read(0x00) & 0x02) == 0:
        pass
        
    hw_end = time.time()

    admm_iters = int(control_ip.read(0x98))
    pcg_iters  = int(control_ip.read(0xa8))
    p_res      = float(uint_to_float(control_ip.read(0xb8)))
    d_res      = float(uint_to_float(control_ip.read(0xc8)))
    status     = int(control_ip.read(0xd8))

    # Calculate Unscaled X, MAE, and Objective Function
    x_unscaled = np.array(x_hw) * E_scale
    mae = np.mean(np.abs(x_unscaled - x_true))
    obj = float(0.5 * np.sum(P_diag * x_unscaled * x_unscaled) + np.dot(q, x_unscaled))

    print(f"HW execution time: {(hw_end - hw_start)*1000:.4f} ms")
    print(f"Status: {'Converged' if status == 1 else 'Max Iterations'}")
    print(f"ADMM Iterations: {admm_iters}")
    print(f"PCG Iterations : {pcg_iters}")
    print(f"Primal Residual: {p_res:.5e}")
    print(f"Dual Residual  : {d_res:.5e}")
    print(f"Objective Value: {obj:.6e}")
    print(f"Mean Abs Error : {mae:.5e}")
    
    print("\n--- First 10 Elements of x_unscaled ---")
    for i in range(min(10, NUM_COLS)):
        print(f"  x[{i:2d}] = {x_unscaled[i]:13.6f} | Expected: {x_true[i]:13.6f}")
    print("")
    
    results.append({
        "admm_iters": admm_iters, "pcg_iters": pcg_iters, 
        "obj": obj, "mae": mae, "hw_ms": (hw_end - hw_start)*1000
    })

print("=== Summary ===")
off, on = results[0], results[1]
print(f"ADMM iterations: off={off['admm_iters']} | on={on['admm_iters']}")
print(f"PCG iterations:  off={off['pcg_iters']} | on={on['pcg_iters']}")
print(f"Objective Value: off={off['obj']:.6e} | on={on['obj']:.6e}")
print(f"MAE from Truth:  off={off['mae']:.3e} | on={on['mae']:.3e}")
print(f"HW time (ms):    off={off['hw_ms']:.3f} | on={on['hw_ms']:.3f}\n")

for b in buffers:
    b.freebuffer()
print("Buffers released.")