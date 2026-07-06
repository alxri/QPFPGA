import numpy as np
import scipy.sparse as sp

# --- Constants ---
PCG_TOL_MIN_SQ = 1E-14
NUM_ITERATIONS_CHECK_TERMINATION = 25 # Matched to OSQP interval
ADMM_ADAPTIVE_RHO_INTERVAL = 50
ADMM_ADAPTIVE_RHO_TOLERANCE = 2.0
ADMM_RHO_MAX = 1e6
ADMM_RHO_MIN = 1e-6
ADMM_RHO_EQ_OVER_RHO_INEQ = 1e3
OSQP_RHO_TOL = 1e-4

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
        x_norm = np.maximum(P_col_norm, A_col_norm)
        x_norm = np.maximum(x_norm, 1e-4)
        z_norm = A_abs.max(axis=1).toarray().flatten()
        z_norm = np.maximum(z_norm, 1e-4)

        E_new = 1.0 / np.sqrt(x_norm)
        D_new = 1.0 / np.sqrt(z_norm)
        E *= E_new
        D *= D_new

        P_scaled = P_scaled * (E_new ** 2)
        A_scaled = sp.diags(D_new).dot(A_scaled).dot(sp.diags(E_new)).tocsc()

    q_scaled = q * E
    l_scaled = np.where(l == -np.inf, -np.inf, l * D)
    u_scaled = np.where(u ==  np.inf,  np.inf, u * D)

    c = 1.0 / np.maximum(np.max(np.abs(P_scaled)), np.max(np.abs(q_scaled)))
    c = max(c, 1e-4)
    P_scaled *= c
    q_scaled *= c

    return P_scaled, A_scaled, q_scaled, l_scaled, u_scaled, E, D, c

def init_rho_vec(l, u, rho_base=0.1):
    rho_base = float(np.clip(rho_base, ADMM_RHO_MIN, ADMM_RHO_MAX))
    rho = np.full(l.shape, rho_base, dtype=np.float32)
    finite_l = np.isfinite(l)
    finite_u = np.isfinite(u)
    loose = (~finite_l) & (~finite_u)
    eq = finite_l & finite_u & ((u - l) < OSQP_RHO_TOL)
    rho[loose] = ADMM_RHO_MIN
    rho[eq] = rho_base * ADMM_RHO_EQ_OVER_RHO_INEQ
    rho = np.clip(rho, ADMM_RHO_MIN, ADMM_RHO_MAX * ADMM_RHO_EQ_OVER_RHO_INEQ).astype(np.float32)
    return rho

def update_preconditioner(P_diag, A, rho, sigma):
    A_sq = A.copy()
    A_sq.data = A_sq.data ** 2
    A_col_sum = A_sq.T.dot(rho)
    return 1.0 / (P_diag + sigma + A_col_sum)

def update_eps_sq(b_norm, r_prim, r_dual, eps_sq_old, num_iterations, num_pcg_iterations, pcg_max_iterations, pcg_tol_fraction):
    if num_iterations > 0 and num_pcg_iterations >= pcg_max_iterations:
        pcg_tol_fraction *= 0.5
    if num_iterations == 0:
        new_eps_sq = pcg_tol_fraction * pcg_tol_fraction
    else:
        denom = (b_norm * b_norm) if b_norm > 0.0 else 1.0
        new_eps_sq = (pcg_tol_fraction * pcg_tol_fraction) * (r_prim * r_dual) / denom

    if new_eps_sq < PCG_TOL_MIN_SQ:
        new_eps_sq = PCG_TOL_MIN_SQ
    if num_iterations > 0 and new_eps_sq > eps_sq_old:
        new_eps_sq = eps_sq_old
    return new_eps_sq, pcg_tol_fraction

def pcg(A, AT, P, M_inv, rho, sigma, eps_sq, x, b, pcg_max_iterations):
    num_cols = len(x)
    Ax = A.dot(x)
    AT_rho_Ax = AT.dot(rho * Ax)
    Kx = P.dot(x) + sigma * x + AT_rho_Ax
    
    r = b - Kx
    z = M_inv * r
    p = z.copy()
    
    rT_y = np.dot(r, z)
    rT_r = np.dot(r, r)
    threshold = eps_sq * np.dot(b, b)
    
    iter_count = 0
    while iter_count < pcg_max_iterations and rT_r > threshold:
        Ap = A.dot(p)
        AT_rho_Ap = AT.dot(rho * Ap)
        Kp = P.dot(p) + sigma * p + AT_rho_Ap
        
        alpha = rT_y / np.dot(p, Kp)
        x += alpha * p
        r -= alpha * Kp
        
        z = M_inv * r
        rT_y_next = np.dot(r, z)
        rT_r = np.dot(r, r)
        
        beta = rT_y_next / rT_y
        p = z + beta * p
        rT_y = rT_y_next
        iter_count += 1
        
    return iter_count

def solve_admm_pcg(P_diag, P_sparse, A_sparse, q, l, u, rho_in, 
                   sigma=1e-3, alpha=1.6, admm_max_iters=4000, 
                   pcg_max_iters=20, adaptive_rho=True, 
                   eps_abs=1e-3, eps_rel=1e-3, pcg_tol_fraction=0.1):
    
    num_rows, num_cols = A_sparse.shape
    AT_sparse = A_sparse.transpose().tocsc()
    
    x = np.zeros(num_cols, dtype=np.float32)
    x_tilde = np.zeros(num_cols, dtype=np.float32)
    y = np.zeros(num_rows, dtype=np.float32)
    z = np.zeros(num_rows, dtype=np.float32)
    
    rho = rho_in.copy()
    rho_inv = 1.0 / rho
    current_rho_base = 0.1 
    
    M_inv = update_preconditioner(P_diag, A_sparse, rho, sigma)
    norm_q = np.max(np.abs(q))
    eps_sq = 0.0
    r_prim, r_dual = 0.0, 0.0
    
    num_iterations = 0
    total_pcg_iterations = 0
    
    while num_iterations < admm_max_iters:
        if adaptive_rho and num_iterations > 0 and num_iterations % ADMM_ADAPTIVE_RHO_INTERVAL == 0:
            if (r_prim > ADMM_ADAPTIVE_RHO_TOLERANCE * r_dual) or (r_dual > ADMM_ADAPTIVE_RHO_TOLERANCE * r_prim):
                S = np.sqrt(r_prim / max(r_dual, 1e-15))
                rho_base_new = np.clip(current_rho_base * S, ADMM_RHO_MIN, ADMM_RHO_MAX)
                S_eff = rho_base_new / current_rho_base
                
                mask = rho > (ADMM_RHO_MIN * 1.1)
                rho_new = rho.copy()
                rho_new[mask] = np.clip(rho[mask] * S_eff, ADMM_RHO_MIN, ADMM_RHO_MAX * ADMM_RHO_EQ_OVER_RHO_INEQ)
                
                changed_mask = rho_new != rho
                y[changed_mask] = y[changed_mask] * (rho_new[changed_mask] / rho[changed_mask])
                
                rho = rho_new
                rho_inv = 1.0 / rho
                current_rho_base = rho_base_new
                M_inv = update_preconditioner(P_diag, A_sparse, rho, sigma)

        AT_y_rho_z = AT_sparse.dot(rho * z - y)
        b = sigma * x - q + AT_y_rho_z
        b_norm = np.max(np.abs(b))
        
        eps_sq, pcg_tol_fraction = update_eps_sq(b_norm, r_prim, r_dual, eps_sq, 
                                                 num_iterations, total_pcg_iterations, 
                                                 pcg_max_iters, pcg_tol_fraction)

        pcg_iters = pcg(A_sparse, AT_sparse, P_sparse, M_inv, rho, sigma, eps_sq, x_tilde, b, pcg_max_iters)
        total_pcg_iterations += pcg_iters

        Ax_tilde = A_sparse.dot(x_tilde)
        x = alpha * x_tilde + (1.0 - alpha) * x
        
        z_prev = z.copy()
        v = alpha * Ax_tilde + (1.0 - alpha) * z_prev
        z = np.clip(v + y * rho_inv, l, u)
        y = y + rho * (v - z)
        
        Ax = A_sparse.dot(x)
        r_prim = np.max(np.abs(Ax - z))
        r_dual = np.max(np.abs(AT_sparse.dot(rho * (z - z_prev))))
        
        if num_iterations > 0 and num_iterations % NUM_ITERATIONS_CHECK_TERMINATION == 0:
            ATy = AT_sparse.dot(y)
            Px = P_sparse.dot(x)
            r_dual_kkt = np.max(np.abs(Px + q + ATy))
            max_prim = max(np.max(np.abs(Ax)), np.max(np.abs(z)))
            max_dual = max(np.max(np.abs(Px)), norm_q, np.max(np.abs(ATy)))
            
            if r_prim <= (eps_abs + eps_rel * max_prim) and r_dual_kkt <= (eps_abs + eps_rel * max_dual):
                break
                
        num_iterations += 1

    r_dual_kkt = np.max(np.abs(P_sparse.dot(x) + q + AT_sparse.dot(y)))
    return {"x": x, "admm_iters": num_iterations, "pcg_iters": total_pcg_iterations, "r_prim": r_prim, "r_dual": r_dual_kkt}