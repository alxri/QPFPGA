#ifndef ADMM_PCG_PARAMS_H
#define ADMM_PCG_PARAMS_H

#define PACK_SIZE 16

#define SCALING_ITER_DEFAULT 10

#define ADMM_RHO 1.0f
#define ADMM_RHO_EQ_OVER_RHO_INEQ 100
#define ADMM_RHO_TOL 0.01f
#define ADMM_INFTY 1e17f
#define ADMM_RHO_MAX 1e6f
#define ADMM_RHO_MIN 1e-6f

// ADMM and PCG parameters, can be tuned when calling solver as solver options
#define ADMM_SIGMA 1e-2f
#define ADMM_ALPHA 1.8f
#define ADMM_EPS_ABS 5e-3f
#define ADMM_EPS_REL 5e-3f
#define ADMM_PCG_TOL_FRACTION 1.0f
#define ADMM_MAX_ITER 2000
#define PCG_MAX_ITER 5


struct RunResult {
    int admm_iters;
    int pcg_iters;
    float r_prim;
    float r_dual;
    float viol;
    double obj;
    double hw_ms;
    float mae;
};

#endif // ADMM_PCG_PARAMS_H