import numpy as np
from utils import psi_derivative


def compute_smooth_pds(u, v):
    H, W = u.shape
    diffusivity_map = np.zeros((2*H+1, 2*W+1), dtype=np.float32)
    
    u_y, u_x = np.gradient(u)
    v_y, v_x = np.gradient(v)
    #u_t = u - u_prev
    #v_t = v - v_prev
    
    ux_avg = np.pad((u_x[:, :-1] + u_x[:, 1:]) / 2, ((0, 0), (0, 1)), mode='constant')
    vx_avg = np.pad((v_x[:, :-1] + v_x[:, 1:]) / 2, ((0, 0), (0, 1)), mode='constant')
    uy_avg = np.pad((u_y[:-1, :] + u_y[1:, :]) / 2, ((0, 1), (0, 0)), mode='constant')
    vy_avg = np.pad((v_y[:-1, :] + v_y[1:, :]) / 2, ((0, 1), (0, 0)), mode='constant')
    
    ux_pd = u_x**2
    uy_pd = u_y**2
    vx_pd = v_x**2
    vy_pd = v_y**2
    
    np.add(ux_pd, np.pad((uy_avg[:, :-1] + uy_avg[:, 1:]) / 2, ((0, 0), (0, 1)), mode='constant')**2, out=ux_pd)
    np.add(uy_pd, np.pad((ux_avg[:-1, :] + ux_avg[1:, :]) / 2, ((0, 1), (0, 0)), mode='constant')**2, out=uy_pd)
    np.add(vx_pd, np.pad((vy_avg[:, :-1] + vy_avg[:, 1:]) / 2, ((0, 0), (0, 1)), mode='constant')**2, out=vx_pd)
    np.add(vy_pd, np.pad((vx_avg[:-1, :] + vx_avg[1:, :]) / 2, ((0, 1), (0, 0)), mode='constant')**2, out=vy_pd)

    diffusivity_map[:-1:2, 1::2] = psi_derivative(uy_pd + vy_pd)
    diffusivity_map[1::2, :-1:2] = psi_derivative(ux_pd + vx_pd)
    
    return diffusivity_map