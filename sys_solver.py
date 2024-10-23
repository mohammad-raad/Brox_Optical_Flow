import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, spilu
import config as cfg

def sor_solver_sparse(A, b, x0, M=None, tol=1e-3, maxiter=500, omega=1.8):
    if not sp.isspmatrix_csr(A):
        A = sp.csr_matrix(A)
    
    n = len(b)
    x = x0.copy()
    
    for k in range(maxiter):
        x_old = x.copy()
        for i in range(n):
            row_start, row_end = A.indptr[i], A.indptr[i+1]
            A_row = A.indices[row_start:row_end]
            A_data = A.data[row_start:row_end]
            
            sum1 = np.dot(A_data[A_row < i], x[A_row[A_row < i]])
            sum2 = np.dot(A_data[A_row > i], x_old[A_row[A_row > i]])
            
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sum1 - sum2)
        
        error = np.linalg.norm(x - x_old, ord=np.inf)
        if error < tol:
            return x, {'iterations': k+1, 'error': error, 'converged': True}
    
    return x, {'iterations': maxiter, 'error': error, 'converged': False}



def solve_system(solver, A, b, uv_residual_prev):
    A = A.tocsc()
    ilu = spilu(A)
    M = lambda x: ilu.solve(x)
    M_linear_operator = LinearOperator((A.shape[0], A.shape[0]), M, dtype='float64')
    
    uv_residual, info = solver(A, b, x0=uv_residual_prev, M=M_linear_operator, tol=cfg.TOLERANCE, maxiter=cfg.MAX_ITER)
    
    return uv_residual, info


