// Auto-generated DSE Config
#ifndef DSE_CONFIG_H
#define DSE_CONFIG_H

// SpMV Engine
#define NUM_PES 6
#define MAX_ROWS 32768
#define MAX_COLS 32768

#define RESHAPE_FACTOR 8

// Accelerator supported maximum problem size
#define MAX_SIZE 32768

// ADMM - PCG Default Parameters
#define OSQP_CG_TOL_MIN 1e-5f
#define OSQP_INFTY 1e17f
#define OSQP_DIVISION_TOL 1e-10f

#define OSQP_CHECK_TERMINATION 5

#define OSQP_ADAPTIVE_RHO_INTERVAL 50
#define OSQP_ADAPTIVE_RHO_TOLERANCE 2

#define OSQP_RHO 0.1f
#define OSQP_RHO_MAX 1e6f
#define OSQP_RHO_MIN 1e-6f
#define OSQP_RHO_EQ_OVER_RHO_INEQ 1e03


#define MAX_TILES ( (MAX_COLS / MAX_ROWS) * (MAX_COLS / MAX_COLS) )



#endif // DSE_CONFIG_H
