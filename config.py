from sys_solver import sor_solver_sparse
from scipy.sparse.linalg import gmres,bicgstab, cg


# Directories
GT_DIR = 'path/to/gt'
INPUT_DIR = 'path/to/input'
RESULTS_DIR = "results"


# Sparse Linear System Solver.
SOLVERS = {
    "SOR": sor_solver_sparse,
    "GMRES": gmres,
    "BICGSTAB": bicgstab,
    "CG": cg
}


NUM_LEVELS = 40           # Number of pyramid levels for multi-resolution approach. (4)
SCALING_FACTOR = 0.95     # Factor by which the image is scaled at each pyramid level.(0.95)
NUM_INNER_ITER =  5       # Number of inner iterations of the fixed-point iteration. (5)
ALPHA = 8                 # Regularization parameter that controls the smoothness of the flow field. (8) (Higher values results in smoother flows)
GAMMA = 1                 # Weighting factor for the gradient terms in the smoothness term. (1)
#CONVERGENCE_THRESH = 0.5


# Sparse Linear System Solver
OMEGA = 1.8       # Relaxation parameter for the SOR (Successive Over-Relaxation) method used to solve the linear system
MAX_ITER = 100     # Maximum number of iterations for the solver.
TOLERANCE = 1e-3  # Tolerance threshold.