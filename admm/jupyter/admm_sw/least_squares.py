import cvxpy as cp
import numpy
import time
import numpy as np
import scipy.sparse as sp

from admm import apply_scaling, init_rho_vec, solve_admm_pcg


# Problem data.
m = 300
n = 200
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective is returned by prob.solve().
print("Solving with CVXPY (OSQP)...")
t0 = time.time()
result = prob.solve(solver=cp.OSQP, eps_abs=1e-3, eps_rel=1e-3, max_iter=4000)
t1 = time.time()
print(f"CVXPY Objective: {result:.6f}")
print(f"CVXPY Solve Time: {(t1 - t0) * 1000:.3f} ms\n")


# =====================================================================
# 4. Extract Canonical OSQP Data
# =====================================================================
print("--- Extracting Canonical Data ---")
data, chain, inverse_data = prob.get_problem_data(cp.OSQP)

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


# # The optimal value for x is stored in x.value.
# print(x.value)
# # The optimal Lagrange multiplier for a constraint
# # is stored in constraint.dual_value.
# print(constraints[0].dual_value)

