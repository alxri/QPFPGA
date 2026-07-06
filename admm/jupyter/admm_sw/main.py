import time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp

# Import the custom solver functions from your separate file
from admm import apply_scaling, init_rho_vec, solve_admm_pcg

# =====================================================================
# 1. Define the MPC Problem using CVXPY
# =====================================================================
print("--- Formulating MPC Problem ---")
H, n, m = 42, 6, 3

# Define variables
U = cp.Variable((m, H), name='U')
X = cp.Variable((n, H+1), name='X')

# Define parameters
Psqrt = cp.Parameter((n, n), name='Psqrt')
Qsqrt = cp.Parameter((n, n), name='Qsqrt')
Rsqrt = cp.Parameter((m, m), name='Rsqrt')
A_param = cp.Parameter((n, n), name='A')
B_param = cp.Parameter((n, m), name='B')
x_init = cp.Parameter(n, name='x_init')

# Define objective
objective = cp.Minimize(cp.sum_squares(Psqrt @ X[:, H]) + 
                        cp.sum_squares(Qsqrt @ X[:, :H]) + 
                        cp.sum_squares(Rsqrt @ U))

# Define constraints
constraints = [
    X[:, 1:] == A_param @ X[:, :H] + B_param @ U,
    cp.abs(U) <= 1,
    X[:, 0] == x_init
]

problem = cp.Problem(objective, constraints)

# =====================================================================
# 2. Assign Parameter Values (System Physics)
# =====================================================================
A_cont = np.concatenate((
    np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ]),
    np.zeros((3, 6))
), axis=0)

mass = 1.0
B_cont = np.concatenate((
    np.zeros((3, 3)), 
    (1/mass) * np.diag(np.ones(3))
), axis=0)

td = 0.1
A_param.value = np.eye(n) + td * A_cont
B_param.value = td * B_cont

Psqrt.value = np.eye(n)
Qsqrt.value = np.eye(n)
Rsqrt.value = np.sqrt(0.1) * np.eye(m)
x_init.value = np.array([2, 2, 2, -1, -1, 1])

# =====================================================================
# 3. Solve Locally with CVXPY (Baseline)
# =====================================================================
print("Solving with CVXPY (OSQP)...")
t0 = time.time()
val_cvxpy = problem.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, max_iter=4000)
t1 = time.time()
print(f"CVXPY Objective: {val_cvxpy:.6f}")
print(f"CVXPY Solve Time: {(t1 - t0) * 1000:.3f} ms\n")

# =====================================================================
# 4. Extract Canonical OSQP Data
# =====================================================================
print("--- Extracting Canonical Data ---")
data, chain, inverse_data = problem.get_problem_data(cp.OSQP)

P_osqp = data['P']
q_osqp = data['q']
A_eq = data['A']
b_eq = data['b']
F_ineq = data['F']
G_ineq = data['G']

# Build full matrices
A_osqp = sp.vstack([A_eq, F_ineq], format='csc')
l_osqp = np.hstack([b_eq, -np.inf * np.ones(F_ineq.shape[0])])
u_osqp = np.hstack([b_eq, G_ineq])

print(f"P shape: {P_osqp.shape}, nnz: {P_osqp.nnz}")
print(f"A shape: {A_osqp.shape}, nnz: {A_osqp.nnz}")

# =====================================================================
# 5. Prepare Data for Custom SW ADMM Solver
# =====================================================================
print("\n--- Running SW ADMM+PCG Solver ---")

# Ensure everything is float32 to match hardware behavior
P_diag_orig = P_osqp.diagonal().astype(np.float32)
A_sparse_f32 = A_osqp.astype(np.float32)
q_f32 = q_osqp.astype(np.float32)
l_f32 = l_osqp.astype(np.float32)
u_f32 = u_osqp.astype(np.float32)

# Apply Scaling
(P_scaled_diag, A_scaled, q_scaled, l_scaled, u_scaled, 
 E_scale, D_scale, c_scale) = apply_scaling(
    P_diag_orig, A_sparse_f32, q_f32, l_f32, u_f32, iters=10
)

# Convert scaled P diagonal back to sparse matrix for the solver
P_sparse_scaled = sp.diags(P_scaled_diag).tocsc()

# Initialize rho
rho_in = init_rho_vec(l_scaled, u_scaled, rho_base=0.1)

# =====================================================================
# 6. Execute Custom Python Solver
# =====================================================================
t0 = time.time()
res_py = solve_admm_pcg(
    P_diag=P_scaled_diag,
    P_sparse=P_sparse_scaled,
    A_sparse=A_scaled,
    q=q_scaled,
    l=l_scaled,
    u=u_scaled,
    rho_in=rho_in,
    sigma=1e-6,
    alpha=1.6,
    admm_max_iters=4000,
    pcg_max_iters=20,
    adaptive_rho=True,
    eps_abs=1e-3,
    eps_rel=1e-3
)
t1 = time.time()

# =====================================================================
# 7. Unscale & Evaluate Results
# =====================================================================
x_py_unscaled = res_py["x"] * E_scale

# Objective: (1/2) * x^T * P * x + q^T * x
obj_py = 0.5 * np.sum(P_diag_orig * x_py_unscaled * x_py_unscaled) + np.dot(q_f32, x_py_unscaled)

# Violation check
Ax_py = A_sparse_f32.dot(x_py_unscaled)
viol_py = float(np.max(np.maximum(0.0, np.maximum(l_f32 - Ax_py, Ax_py - u_f32))))

print(f"ADMM Iterations: {res_py['admm_iters']}")
print(f"PCG Iterations : {res_py['pcg_iters']}")
print(f"Primal Residual: {res_py['r_prim']:.5e} (Scaled)")
print(f"Dual Residual  : {res_py['r_dual']:.5e} (Scaled)")
print(f"Constraint Viol: {viol_py:.3e} (Unscaled)")
print(f"Objective Value: {obj_py:.6f} (Unscaled)")
print(f"Python SW Time : {(t1 - t0) * 1000:.3f} ms")
