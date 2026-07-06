import os
import time
import numpy as np
import scipy.sparse as sp

# =====================================================================
# 1. ADMM+PCG Behavioral Tracker Class
# =====================================================================

class ADMMPCGTracker:
    def __init__(self, P, q, A, l, u, rho_vec, rho_base=0.1):
        # Problem dimensions
        self.P = P.tocsc()
        self.A = A.tocsc()
        self.AT = A.T.tocsc()
        self.q = q
        self.l = l
        self.u = u
        
        self.num_cols = P.shape[0]
        self.num_rows = A.shape[0]
        
        # P diagonal for preconditioner
        self.P_diag = self.P.diagonal()
        
        # Hyperparameters (Matching C++ defines)
        self.OSQP_INFTY = 1e17
        self.OSQP_CG_TOL_MIN = 1e-5
        self.OSQP_DIVISION_TOL = 1e-10
        self.OSQP_RHO_MIN = 1e-6
        self.OSQP_RHO_MAX = 1e6
        self.OSQP_RHO_EQ_OVER_RHO_INEQ = 1e3
        
        # ADMM State Variables
        self.x = np.zeros(self.num_cols)
        self.x_tilde = np.zeros(self.num_cols)
        self.z = np.zeros(self.num_rows)
        self.y = np.zeros(self.num_rows)
        
        self.rho = rho_vec.copy()
        self.current_rho_base = rho_base
        
        # Tracking history
        self.history = {
            'admm_iter': [], 'pcg_iters': [], 'r_prim': [], 
            'r_dual': [], 'pcg_eps': [], 'rho_base': []
        }

    def update_preconditioner(self, sigma):
        A_sq = self.A.copy()
        A_sq.data = A_sq.data ** 2
        A_col_sum = A_sq.T.dot(self.rho)
        self.M_inv = 1.0 / (self.P_diag + sigma + A_col_sum)

    def K_op(self, v, sigma):
        return self.P.dot(v) + sigma * v + self.AT.dot(self.rho * self.A.dot(v))

    def pcg(self, b, sigma, epsilon, max_iters):
        K_x = self.K_op(self.x_tilde, sigma)
        r = b - K_x
        z = self.M_inv * r
        p = z.copy()
        
        rT_z = np.dot(r, z)
        r_norm = np.linalg.norm(r, np.inf)
        
        iter_count = 0
        for k in range(max_iters):
            if r_norm <= epsilon:
                break
                
            K_p = self.K_op(p, sigma)
            pT_K_p = np.dot(p, K_p)
            
            if pT_K_p <= 0.0:
                break # Negative curvature / precision loss
                
            alpha = rT_z / pT_K_p
            self.x_tilde = self.x_tilde + alpha * p
            r = r - alpha * K_p
            z = self.M_inv * r
            
            rT_z_next = np.dot(r, z)
            if rT_z_next <= 0.0:
                break
                
            beta = rT_z_next / rT_z
            p = z + beta * p
            
            rT_z = rT_z_next
            r_norm = np.linalg.norm(r, np.inf)
            iter_count += 1
            
        return iter_count

    def solve(self, sigma=1e-6, alpha_relax=1.6, pcg_tol_fraction=0.15, 
              admm_max_iter=2000, pcg_max_iter=10, eps_abs=1e-3, eps_rel=1e-3, 
              adaptive_rho=True, verbose=True):
        
        self.update_preconditioner(sigma)
        
        eps = 0.0
        zero_pcg_iters = 0
        total_pcg_iters = 0
        pcg_iters_last = 0
        
        for num_iterations in range(admm_max_iter):
            # 1. Adaptive Rho Update
            if adaptive_rho and num_iterations > 0 and num_iterations % 50 == 0:
                norm_z = np.linalg.norm(self.z, np.inf)
                norm_Ax = np.linalg.norm(self.A.dot(self.x), np.inf)
                norm_q = np.linalg.norm(self.q, np.inf)
                ATy_norm = np.linalg.norm(self.AT.dot(self.y), np.inf)
                Px_norm = np.linalg.norm(self.P.dot(self.x), np.inf)
                
                prim_res_scaled = self.history['r_prim'][-1] / (max(norm_z, norm_Ax) + self.OSQP_DIVISION_TOL)
                dual_res_scaled = self.history['r_dual'][-1] / (max(norm_q, ATy_norm, Px_norm) + self.OSQP_DIVISION_TOL)
                
                S = np.sqrt(prim_res_scaled / max(dual_res_scaled, self.OSQP_DIVISION_TOL))
                rho_new = np.clip(self.current_rho_base * S, self.OSQP_RHO_MIN, self.OSQP_RHO_MAX)
                
                if (rho_new > self.current_rho_base * 2.0) or (rho_new < self.current_rho_base / 2.0):
                    S_eff = rho_new / self.current_rho_base
                    mask = self.rho > (self.OSQP_RHO_MIN * 1.1)
                    rho_updated = np.where(mask, self.rho * S_eff, self.rho)
                    rho_updated = np.clip(rho_updated, self.OSQP_RHO_MIN, self.OSQP_RHO_MAX * self.OSQP_RHO_EQ_OVER_RHO_INEQ)
                    
                    self.y = np.where(rho_updated != self.rho, self.y * (rho_updated / self.rho), self.y)
                    self.rho = rho_updated
                    self.current_rho_base = rho_new
                    self.update_preconditioner(sigma)

            # 2. Update b
            b = sigma * self.x - self.q + self.AT.dot(self.rho * self.z - self.y)
            b_norm = np.linalg.norm(b, np.inf)
            
            # 3. Update EPS for PCG
            if num_iterations > 0:
                if pcg_iters_last == 0:
                    zero_pcg_iters += 1
                else:
                    zero_pcg_iters = 0
                    
                if zero_pcg_iters >= 5:
                    pcg_tol_fraction *= 0.5
                    zero_pcg_iters = 0
            
            if num_iterations == 0:
                eps_new = b_norm * pcg_tol_fraction
            else:
                eps_new = pcg_tol_fraction * np.sqrt(self.history['r_prim'][-1] * self.history['r_dual'][-1])
            
            eps_new = max(eps_new, self.OSQP_CG_TOL_MIN)
            if num_iterations > 0 and eps_new > eps:
                eps_new = eps
            eps = eps_new

            # 4. PCG Solve
            pcg_iters_last = self.pcg(b, sigma, eps, pcg_max_iter)
            total_pcg_iters += pcg_iters_last
            
            # 5. ADMM Updates
            A_xtilde = self.A.dot(self.x_tilde)
            self.x = alpha_relax * self.x_tilde + (1.0 - alpha_relax) * self.x
            
            z_prev = self.z.copy()
            v = alpha_relax * A_xtilde + (1.0 - alpha_relax) * z_prev
            
            new_z = np.clip(v + self.y / self.rho, self.l, self.u)
            self.z = new_z
            self.y = self.y + self.rho * (v - new_z)
            
            # 6. Compute Residuals
            Ax = self.A.dot(self.x)
            r_prim = np.linalg.norm(Ax - self.z, np.inf)
            r_dual = np.linalg.norm(self.P.dot(self.x) + self.q + self.AT.dot(self.y), np.inf)
            
            # Tracking
            self.history['admm_iter'].append(num_iterations)
            self.history['pcg_iters'].append(pcg_iters_last)
            self.history['r_prim'].append(r_prim)
            self.history['r_dual'].append(r_dual)
            self.history['pcg_eps'].append(eps)
            self.history['rho_base'].append(self.current_rho_base)

            # 7. Convergence Check
            if num_iterations > 0 and num_iterations % 5 == 0:
                max_prim = max(np.linalg.norm(Ax, np.inf), np.linalg.norm(self.z, np.inf))
                max_dual = max(np.linalg.norm(self.P.dot(self.x), np.inf), np.linalg.norm(self.q, np.inf), np.linalg.norm(self.AT.dot(self.y), np.inf))
                
                eps_prim = eps_abs + eps_rel * max_prim
                eps_dual = eps_abs + eps_rel * max_dual
                
                if r_prim <= eps_prim and r_dual <= eps_dual:
                    if verbose: print(f"Converged at iter {num_iterations}")
                    break

            if verbose and num_iterations % 50 == 0:
                print(f"Iter {num_iterations:4d} | PCG: {pcg_iters_last:2d} | p_res: {r_prim:.2e} | d_res: {r_dual:.2e} | rho: {self.current_rho_base:.2e}")

        return self.x, num_iterations, total_pcg_iters

# =====================================================================
# 2. Load Canonicalized CVXPY QP Data
# =====================================================================

VERBOSE_TRACKER = True # Set to False to disable step-by-step prints

data_dir = "./data"
print(f"Loading QP data from: {os.path.abspath(data_dir)}")

# Load sparse matrices
P_sparse = sp.load_npz(os.path.join(data_dir, "P.npz")).tocsc()
A_sparse = sp.load_npz(os.path.join(data_dir, "A.npz")).tocsc()

# Load vectors
q = np.load(os.path.join(data_dir, "q.npy")).astype(np.float32)
l_in = np.load(os.path.join(data_dir, "l.npy")).astype(np.float32)
u_in = np.load(os.path.join(data_dir, "u.npy")).astype(np.float32)

num_rows, num_cols = A_sparse.shape
print(f"Problem size: {num_rows} x {num_cols}")

P_diag_orig = P_sparse.diagonal().astype(np.float32)

# =====================================================================
# 3. Matrix Scaling (Scaling + Cost Scaling)
# =====================================================================

print("Applying Scaling...")
OSQP_INFTY = 1e17

def apply_scaling(P_diag, A_sparse, q, l, u, iters=10):
    n = len(P_diag)
    m = A_sparse.shape[0]

    E = np.ones(n, dtype=np.float32)
    D = np.ones(m, dtype=np.float32)

    P_scaled = P_diag.copy()
    A_scaled = A_sparse.copy()

    for _ in range(iters):
        A_abs = abs(A_scaled)
        A_col_norm = A_abs.max(axis=0).toarray().flatten()
        P_col_norm = np.abs(P_scaled)
        x_norm = np.maximum(np.maximum(P_col_norm, A_col_norm), 1e-4)

        z_norm = np.maximum(A_abs.max(axis=1).toarray().flatten(), 1e-4)

        E_new = 1.0 / np.sqrt(x_norm)
        D_new = 1.0 / np.sqrt(z_norm)
        E *= E_new
        D *= D_new

        P_scaled = P_scaled * (E_new ** 2)
        A_scaled = sp.diags(D_new).dot(A_scaled).dot(sp.diags(E_new)).tocsc()

    q_scaled = q * E
    
    # Using OSQP_INFTY logic for bounds scaling
    OSQP_MIN_SCALING = 1e-4
    l_scaled = np.where(l <= -(OSQP_INFTY * OSQP_MIN_SCALING), -OSQP_INFTY, l * D)
    u_scaled = np.where(u >=  (OSQP_INFTY * OSQP_MIN_SCALING),  OSQP_INFTY, u * D)

    c = max(1.0 / np.maximum(np.max(np.abs(P_scaled)), np.max(np.abs(q_scaled))), 1e-4)
    P_scaled *= c
    q_scaled *= c

    return P_scaled, A_scaled, q_scaled, l_scaled, u_scaled, E, D, c

(P_diag_scaled, A_sparse_scaled, q_scaled, 
 l_scaled, u_scaled, E_scale, D_scale, c_scale) = apply_scaling(
    P_diag_orig, A_sparse, q, l_in, u_in
)

# Convert diagonal back to sparse matrix for the tracker
P_sparse_scaled = sp.diags(P_diag_scaled).tocsc()

# =====================================================================
# 4. Solver Parameters & Rho Initialization
# =====================================================================

alpha = 1.8
sigma = 1e-2

eps_abs = 5e-3
eps_rel = 5e-3

pcg_tol_fraction = 1.0

admm_max_iterations = 2000
pcg_max_iterations = 5

OSQP_RHO = 1.0
OSQP_RHO_MIN = 1e-6
OSQP_RHO_MAX = 1e6
OSQP_RHO_EQ_OVER_RHO_INEQ = 100
OSQP_RHO_TOL = 0.01


def init_rho_vec(l, u, rho_base=OSQP_RHO):
    rho_base = float(np.clip(rho_base, OSQP_RHO_MIN, OSQP_RHO_MAX))
    rho = np.full(l.shape, rho_base, dtype=np.float32)

    loose_l = l <= -(OSQP_INFTY * 1e-4)
    loose_u = u >=  (OSQP_INFTY * 1e-4)

    loose = loose_l & loose_u
    eq = (~loose_l) & (~loose_u) & ((u - l) < OSQP_RHO_TOL)

    rho[loose] = OSQP_RHO_MIN
    rho[eq] = rho_base * OSQP_RHO_EQ_OVER_RHO_INEQ
    rho = np.clip(rho, OSQP_RHO_MIN, OSQP_RHO_MAX * OSQP_RHO_EQ_OVER_RHO_INEQ).astype(np.float32)

    stats = {
        "rho_base": rho_base,
        "num_loose": int(np.sum(loose)),
        "num_eq": int(np.sum(eq)),
    }
    return rho, stats

rho_in, rho_stats = init_rho_vec(l_scaled, u_scaled, rho_base=OSQP_RHO)
print(f"Initial rho: base={rho_stats['rho_base']:.3e} | eq={rho_stats['num_eq']} | loose={rho_stats['num_loose']}")

# =====================================================================
# 5. Utility Functions
# =====================================================================

def objective_val(x_vec):
    return float(0.5 * np.sum(P_diag_orig * x_vec * x_vec) + np.dot(q, x_vec))

def max_box_violation(z, l, u):
    return float(np.max(np.maximum(0.0, np.maximum(l - z, z - u))))

# =====================================================================
# 6. Software Tracker Execution
# =====================================================================

def run_sw_tracker(adaptive_rho):
    print(f"\n=== SW Tracker Run (adaptive_rho={adaptive_rho}) ===")
    
    # Initialize the Tracker with scaled problem data
    tracker = ADMMPCGTracker(
        P=P_sparse_scaled, 
        q=q_scaled, 
        A=A_sparse_scaled, 
        l=l_scaled, 
        u=u_scaled, 
        rho_vec=rho_in, 
        rho_base=OSQP_RHO
    )

    t0 = time.time()
    
    # Solve
    x_hw_scaled, admm_iters, pcg_iters = tracker.solve(
        sigma=sigma, 
        alpha_relax=alpha, 
        pcg_tol_fraction=pcg_tol_fraction, 
        admm_max_iter=admm_max_iterations, 
        pcg_max_iter=pcg_max_iterations, 
        eps_abs=eps_abs, 
        eps_rel=eps_rel, 
        adaptive_rho=adaptive_rho, 
        verbose=VERBOSE_TRACKER
    )
    
    t1 = time.time()
    hw_ms = (t1 - t0) * 1000.0

    # -------------------------------------------------------------
    # UNSCALE THE SOLUTION
    # -------------------------------------------------------------
    x_hw_unscaled = x_hw_scaled * E_scale

    # Compute violations and objective using ORIGINAL matrices
    Ax_hw = (A_sparse @ x_hw_unscaled).astype(np.float32)
    viol_hw = max_box_violation(Ax_hw, l_in, u_in)
    obj_hw = objective_val(x_hw_unscaled)
    
    # Extract final residuals from the tracker history
    r_prim_out = tracker.history['r_prim'][-1] if admm_iters > 0 else 0
    r_dual_out = tracker.history['r_dual'][-1] if admm_iters > 0 else 0

    print("\n--- Results ---")
    print(f"ADMM Iterations: {admm_iters}")
    print(f"PCG Iterations : {pcg_iters}")
    if admm_iters > 0:
        print(f"PCG/ADMM       : {pcg_iters / float(admm_iters):.2f} (max={pcg_max_iterations})")
    print(f"Primal Residual: {r_prim_out:.5e}")
    print(f"Dual Residual  : {r_dual_out:.5e}")
    print(f"Constraint Violation: {viol_hw:.3e}")
    print(f"Objective Value: {obj_hw:.6e}")
    print(f"SW execution time: {hw_ms:.4f} ms")

    return {
        "adaptive_rho": adaptive_rho,
        "admm_iters": admm_iters,
        "pcg_iters": pcg_iters,
        "r_prim": r_prim_out,
        "r_dual": r_dual_out,
        "viol": viol_hw,
        "obj": obj_hw,
        "hw_ms": hw_ms,
    }

# =====================================================================
# 7. Run Tests
# =====================================================================

RUN_BOTH_ADAPTIVE_RHO = True

adaptive_rho_list = [False, True] if RUN_BOTH_ADAPTIVE_RHO else [False]
results = []
for adaptive_rho in adaptive_rho_list:
    results.append(run_sw_tracker(adaptive_rho))

# =====================================================================
# 8. Summary
# =====================================================================

if RUN_BOTH_ADAPTIVE_RHO and len(results) == 2:
    off = results[0]
    on = results[1]
    print("\n=== Summary ===")
    print(f"ADMM iterations: off={off['admm_iters']} | on={on['admm_iters']}")
    print(f"Primal residual: off={off['r_prim']:.3e} | on={on['r_prim']:.3e}")
    print(f"Dual residual: off={off['r_dual']:.3e} | on={on['r_dual']:.3e}")
    print(f"Violation: off={off['viol']:.3e} | on={on['viol']:.3e}")
    print(f"Objective: off={off['obj']:.6e} | on={on['obj']:.6e}")
    print(f"SW time (ms): off={off['hw_ms']:.3f} | on={on['hw_ms']:.3f}")

    print(f"alpha = {alpha}")
    print(f"sigma = {sigma}")

    print(f"eps_abs = {eps_abs}")
    print(f"eps_rel = {eps_rel}")

    print(f"pcg_tol_fraction = {pcg_tol_fraction}")
    print(f"admm_max_iterations = {admm_max_iterations}")
    print(f"pcg_max_iterations = {pcg_max_iterations}")

    print(f"OSQP_RHO = {OSQP_RHO}")
    print(f"OSQP_RHO_MIN = {OSQP_RHO_MIN}")
    print(f"OSQP_RHO_MAX = {OSQP_RHO_MAX}")
    print(f"OSQP_RHO_EQ_OVER_RHO_INEQ = {OSQP_RHO_EQ_OVER_RHO_INEQ}")
    print(f"OSQP_RHO_TOL = {OSQP_RHO_TOL}")