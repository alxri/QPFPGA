import os
import numpy as np
import osqp
from scipy import sparse

def load_bin_file(path, dtype):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.fromfile(path, dtype=dtype)

def run_comparative_benchmark(data_dir):
    print(f"--- Loading data from {data_dir} ---")

    # Data types: Match your binary generation
    f_type = np.float32  
    i_type = np.int32    

    # 1. Load Vectors
    q = load_bin_file(os.path.join(data_dir, 'q.bin'), dtype=f_type)
    l = load_bin_file(os.path.join(data_dir, 'l.bin'), dtype=f_type)
    u = load_bin_file(os.path.join(data_dir, 'u.bin'), dtype=f_type)
    
    # 2. Load Sparse Matrices
    p_data = load_bin_file(os.path.join(data_dir, 'P_data.bin'), dtype=f_type)
    p_indices = load_bin_file(os.path.join(data_dir, 'P_indices.bin'), dtype=i_type)
    p_indptr = load_bin_file(os.path.join(data_dir, 'P_indptr.bin'), dtype=i_type)
    
    a_data = load_bin_file(os.path.join(data_dir, 'A_data.bin'), dtype=f_type)
    a_indices = load_bin_file(os.path.join(data_dir, 'A_indices.bin'), dtype=i_type)
    a_indptr = load_bin_file(os.path.join(data_dir, 'A_indptr.bin'), dtype=i_type)
    
    n, m = len(q), len(l)
    P = sparse.csc_matrix((p_data, p_indices, p_indptr), shape=(n, n))
    A = sparse.csc_matrix((a_data, a_indices, a_indptr), shape=(m, n))

    print(f"Problem Loaded: n={n}, m={m}")

    # Common Settings for comparison
    # adaptive_rho is False to match your FPGA config
    common_settings = {
        'verbose': True,
        'eps_abs': 1e-05,
        'eps_rel': 1e-05,
        'adaptive_rho': False, 
        'rho': 0.1  # Ensure rho matches your FPGA's constant rho
    }

    # --- 1. BUILTIN DIRECT SOLVER (QDLDL) ---
    print("\n" + "="*50)
    print("RUNNING: Builtin Direct Solver (QDLDL)")
    print("="*50)
    prob_direct = osqp.OSQP(algebra='builtin')
    prob_direct.setup(P, q, A, l, u, **common_settings)
    res_direct = prob_direct.solve()

    # --- 2. MKL INDIRECT SOLVER (CG) ---
    print("\n" + "="*50)
    print("RUNNING: MKL Indirect Solver (CG)")
    print("="*50)
    prob_indirect = osqp.OSQP(algebra='mkl')
    # Update settings specifically for MKL indirect
    mkl_settings = common_settings.copy()
    mkl_settings['solver_type'] = 'indirect'
    
    prob_indirect.setup(P, q, A, l, u, **mkl_settings)
    res_indirect = prob_indirect.solve()

    # --- SUMMARY ---
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Solver':<20} | {'Status':<10} | {'Iter':<10} | {'Time (s)':<10}")
    print("-" * 55)
    print(f"{'Builtin Direct':<20} | {res_direct.info.status:<10} | {res_direct.info.iter:<10} | {res_direct.info.run_time:.4f}")
    print(f"{'MKL Indirect':<20} | {res_indirect.info.status:<10} | {res_indirect.info.iter:<10} | {res_indirect.info.run_time:.4f}")

    return res_direct, res_indirect

if __name__ == "__main__":
    # DATA_PATH = "../32768x32768_data_bin_dse/high/"
    DATA_PATH = "../32768x32768_data_bin_dse/med"
    run_comparative_benchmark(DATA_PATH)