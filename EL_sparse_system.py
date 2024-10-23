import numpy as np
from scipy import sparse
import config as cfg


def construct_sparse_sys(I_x, I_y, I_z, I_xx, I_xy, I_yy, I_xz, I_yz, data_pds, smooth_pds, u_initial, v_initial):
    H, W = u_initial.shape
    smooth_pds[0, :] = smooth_pds[-1, :] = smooth_pds[:, 0] = smooth_pds[:, -1] = 0
    
    base_indices = np.arange(0, 2 * H * W, dtype=np.int32)
    sparse_rows = np.tile(base_indices, (6,1)).T.flatten()
    sparse_cols = sparse_rows.copy()
    coeffs = np.zeros_like(sparse_rows, dtype=np.float32)
    
    sparse_cols[0::6] -= 2
    sparse_cols[1::6] -= 2*W
    sparse_cols[8::12] -= 1
    sparse_cols[3::12] += 1
    sparse_cols[4::6] += 2*W
    sparse_cols[5::6] += 2
    
    smooth_pdsum = smooth_pds[0:2*H:2, 1::2] + smooth_pds[1::2, 0:2*W:2] + smooth_pds[2::2, 1::2] + smooth_pds[1::2, 2::2]
    
    du_coeff_1 = data_pds * (I_x**2 + cfg.GAMMA*(I_xx**2 + I_xy**2)) + smooth_pdsum
    dv_coeff_1 = data_pds * (I_x*I_y + cfg.GAMMA*(I_xx*I_xy + I_yy*I_xy))
    du_coeff_2 = data_pds * (I_y*I_x + cfg.GAMMA*(I_xy*I_xx + I_yy*I_xy))
    dv_coeff_2 = data_pds * (I_y**2 + cfg.GAMMA*(I_yy**2 + I_xy**2)) + smooth_pdsum
    
    coeffs[0::12] = -smooth_pds[1::2, :2*W:2].flatten()
    coeffs[6::12] = -smooth_pds[1::2, :2*W:2].flatten()
    
    coeffs[1::12] = -smooth_pds[:2*H:2, 1::2].flatten()
    coeffs[7::12] = -smooth_pds[:2*H:2, 1::2].flatten()
    
    coeffs[2::12] = du_coeff_1.flatten()
    coeffs[8::12] = du_coeff_2.flatten()
    coeffs[3::12] = dv_coeff_1.flatten()
    coeffs[9::12] = dv_coeff_2.flatten()
    
    coeffs[4::12] = -smooth_pds[2::2, 1::2].flatten()
    coeffs[10::12] = -smooth_pds[2::2, 1::2].flatten()
    
    coeffs[5::12] = -smooth_pds[1::2, 2::2].flatten()
    coeffs[11::12] = -smooth_pds[1::2, 2::2].flatten()
    
    upad = np.pad(u_initial, ((1,1), (1,1)), mode="constant")
    vpad = np.pad(v_initial, ((1,1), (1,1)), mode="constant")
    
    smoothness_const = lambda pad: (
        smooth_pds[1::2, 0:2*W:2]*(pad[1:H+1, 0:W] - pad[1:H+1, 1:W+1]) +
        smooth_pds[1::2, 2::2]*(pad[1:H+1, 2:] - pad[1:H+1, 1:W+1]) +
        smooth_pds[0:2*H:2, 1::2]*(pad[0:H, 1:W+1] - pad[1:H+1, 1:W+1]) +
        smooth_pds[2::2, 1::2]*(pad[2:, 1:W+1] - pad[1:H+1, 1:W+1])
    )
    smoothness_const_1 = smoothness_const(upad)
    smoothness_const_2 = smoothness_const(vpad)
    
    const_EL_1 = data_pds * (I_x*I_z + cfg.GAMMA*(I_xx*I_xz + I_xy*I_yz)) - smoothness_const_1
    const_EL_2 = data_pds * (I_y*I_z + cfg.GAMMA*(I_xy*I_xz + I_yy*I_yz)) - smoothness_const_2
    
    # Construct the right-hand side vector 'b' for the linear system Ax = b
    b = np.zeros(2*H*W, dtype=np.float32)
    b[0::2] = -const_EL_1.flatten()
    b[1::2] = -const_EL_2.flatten()
    
    valid_indices = (sparse_cols >= 0) & (sparse_cols < 2*H*W)
    sparse_cols = sparse_cols[valid_indices]
    sparse_rows = sparse_rows[valid_indices]  
    coeffs = coeffs[valid_indices]
    A = sparse.csr_matrix((coeffs, (sparse_rows, sparse_cols)), shape=(2*H*W, 2*H*W))   # Ax = b

    return A, b