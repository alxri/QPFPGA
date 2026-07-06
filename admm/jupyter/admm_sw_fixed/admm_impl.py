import numpy as np
import scipy.sparse as sp
import time

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
        """Mirrors the C++ update_preconditioner logic using A_col_sum = AT * (rho * A_squared)"""
        # A.copy() and square data to simulate the (val * val) in C++
        A_sq = self.A.copy()
        A_sq.data = A_sq.data ** 2
        
        # A_col_sum_lanes reduction equivalent
        A_col_sum = A_sq.T.dot(self.rho)
        self.M_inv = 1.0 / (self.P_diag + sigma + A_col_sum)

    def K_op(self, v, sigma):
        """The KKT matrix operator: K*v = P*v + sigma*v + A^T*(rho*(A*v))"""
        return self.P.dot(v) + sigma * v + self.AT.dot(self.rho * self.A.dot(v))

    def pcg(self, b, sigma, epsilon, max_iters, verbose=False):
        K_x = self.K_op(self.x_tilde, sigma)
        r = b - K_x
        z = self.M_inv * r
        p = z.copy()
        
        rT_z = np.dot(r, z)
        r_norm = np.linalg.norm(r, np.inf)
        
        if verbose:
            print(f"    [PCG Start] Target eps: {epsilon:.2e} | Initial r_norm: {r_norm:.2e}")
            
        iter_count = 0
        for k in range(max_iters):
            if r_norm <= epsilon:
                if verbose:
                    print(f"    [PCG Exit] Converged at iter {k} | r_norm: {r_norm:.2e}")
                break
                
            K_p = self.K_op(p, sigma)
            pT_K_p = np.dot(p, K_p)
            
            if pT_K_p <= 0.0:
                if verbose:
                    print(f"    [PCG Exit] Negative Curvature at iter {k}")
                break 
                
            alpha = rT_z / pT_K_p
            self.x_tilde = self.x_tilde + alpha * p
            r = r - alpha * K_p
            z = self.M_inv * r
            
            rT_z_next = np.dot(r, z)
            if rT_z_next <= 0.0:
                if verbose:
                    print(f"    [PCG Exit] Stagnation (rT_z_next <= 0) at iter {k}")
                break
                
            beta = rT_z_next / rT_z
            p = z + beta * p
            
            rT_z = rT_z_next
            r_norm = np.linalg.norm(r, np.inf)
            iter_count += 1
            
            if verbose and (k % 5 == 0 or k == max_iters - 1):
                print(f"      pcg_step {k:2d} | r_norm: {r_norm:.2e}")
                
        if iter_count == max_iters and verbose:
            print(f"    [PCG Exit] Max Iters Reached | Final r_norm: {r_norm:.2e} (Target: {epsilon:.2e})")
            
        return iter_count

    def solve(self, sigma=1e-6, alpha_relax=1.6, pcg_tol_fraction=0.15, 
              admm_max_iter=2000, pcg_max_iter=10, eps_abs=1e-3, eps_rel=1e-3, 
              adaptive_rho=True, verbose=True):
        
        self.update_preconditioner(sigma)
        
        eps = 0.0
        zero_pcg_iters = 0
        total_pcg_iters = 0
        
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
            pcg_iters_last = self.pcg(b, sigma, eps, pcg_max_iter, verbose=(verbose and num_iterations % 50 == 0))
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

            # 7. Convergence Check (Every 5 iters matching OSQP_CHECK_TERMINATION)
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