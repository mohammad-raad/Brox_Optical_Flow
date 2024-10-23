import argparse
import os
import multiprocessing as mp

import numpy as np
import cv2
from scipy.ndimage import map_coordinates

import config as cfg
import utils 
from EL_sparse_system import construct_sparse_sys
from diffusivity_map import compute_smooth_pds


class BroxOpticalFlow:
    def __init__(self, num_levels, scaling_factor, solver):
        self.num_levels = num_levels
        self.scaling_factor = scaling_factor
        self.solver = cfg.SOLVERS[solver]


    def _compute_residual(self,I_z, I_x, I_y, I_xx, I_xy, I_yy, I_xz, I_yz, u_initial, v_initial):
        "Computes the flow residuals exploiting the fixed point iteration strategy."
        H, W = I_z.shape
        
        u_residual = np.zeros((H, W))  # Initialize u_residual and v_residual to zero
        v_residual = np.zeros((H, W))
        uv_residual = np.zeros(2*H*W)  # The change in the flow field between successive iterations of the SOR algorithm.
        #uv_residual_prev = np.zeros_like(uv_residual)
        
        for l in range(cfg.NUM_INNER_ITER):
            
            I_z_updated = I_z + I_x * u_residual + I_y * v_residual
            I_xz_updated = I_xz + I_xx * u_residual + I_xy * v_residual
            I_yz_updated = I_yz + I_xy * u_residual + I_yy * v_residual
            
            robustness_factor = utils.psi_derivative((I_z_updated**2) + cfg.GAMMA*((I_xz_updated**2) + (I_yz_updated**2)))
            diffusivity = compute_smooth_pds(u_initial + u_residual, v_initial + v_residual)

            A, b = construct_sparse_sys(
                I_x, I_y, I_z_updated, I_xx, I_xy, I_yy, I_xz_updated, I_yz_updated,
                robustness_factor, cfg.ALPHA * diffusivity,
                u_initial, v_initial
            )
            
            #uv_residual_prev[:] = uv_residual
            #uv_residual, info = solve_system(self.solver, A, b, uv_residual_prev)
            
            uv_residual, info = self.solver(A, b, x0=uv_residual, rtol=cfg.TOLERANCE, maxiter=cfg.MAX_ITER)
            
            #if np.linalg.norm(uv_residual - uv_residual_prev) < cfg.CONVERGENCE_THRESH:
                #print(f"Converged at iteration {l}...")
                #break

            u_residual = uv_residual[0::2].reshape(H, W)
            v_residual = uv_residual[1::2].reshape(H, W)
            
        return u_residual, v_residual
    
    
    
    def _update_flow(self, I_1, I_2, u_initial, v_initial):  # u, v are the scaled-up flows coming frome the coarser level...
        # Warp I_2 based on the current flow (u_initial, v_initial)
        height, width = I_1.shape
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        coords_x = grid_x + u_initial
        coords_y = grid_y + v_initial
        
        I_2_warped = map_coordinates(I_2, [coords_y, coords_x], order=1, mode='nearest')
        #I_2_warped = cv2.remap(I_2, coords_x.astype(np.float32), coords_y.astype(np.float32), cv2.INTER_LINEAR)
        
        I_y, I_x = np.gradient(I_2_warped)
        I_y1, I_x1 = np.gradient(I_1)
        I_z = I_2_warped - I_1     
        I_xz = I_x - I_x1
        I_yz = I_y - I_y1
        _, I_xx = np.gradient(I_x)
        I_yy, I_xy = np.gradient(I_y)
        
        u_residual, v_residual = self._compute_residual(I_z, I_x, I_y, I_xx, I_xy, I_yy, I_xz, I_yz, u_initial, v_initial)
        
        u_updated = u_initial + u_residual
        v_updated = v_initial + v_residual
        
        return u_updated, v_updated
   
   
   
    def _estimate_flow(self, images):
        u_ests = []
        v_ests = []
        
        for i in range(len(images)-1):
            img1 = images[i]
            img2 = images[i+1]

            # Initializing Flow at the coarsest level with zero...
            coarsest_scale = utils.img_scaling(img1, self.scaling_factor**self.num_levels).shape
            u = np.zeros(coarsest_scale)
            v = np.zeros(coarsest_scale)
            
            for k in range(1, self.num_levels + 1):
                scale_factor = self.scaling_factor ** (self.num_levels - k)
                scaled_up_1 = utils.img_scaling(img1, scale_factor)
                scaled_up_2 = utils.img_scaling(img2, scale_factor)
                
                u_initial = cv2.resize(u, (scaled_up_1.shape[1], scaled_up_1.shape[0]), interpolation=cv2.INTER_LINEAR)
                v_initial = cv2.resize(v, (scaled_up_1.shape[1], scaled_up_1.shape[0]), interpolation=cv2.INTER_LINEAR)
                
                u, v = self._update_flow(scaled_up_1, scaled_up_2, u_initial, v_initial)   # Input arguments are at the same dimentions.
                
            u_ests.append(u)
            v_ests.append(v)
            
        return u_ests, v_ests
    


    def _process_sequence(self, sequence_key, sequences):
        sequence_files = sequences[sequence_key][:2]
        batch_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) for img_path in sequence_files]
        print(f"The sequence {sequence_key} has {len(sequence_files)} frames in the shape of {batch_images[0].shape}.")
        
        u_ests, v_ests = self._estimate_flow(batch_images)
        
        filenames = [os.path.basename(file).split('.')[0] for file in sequence_files]
        utils.save_flow(u_ests, v_ests, filenames[:-1], format="flo")


def main():
    parser = argparse.ArgumentParser(description="Brox Optical Flow Estimation.")
    parser.add_argument("--task", type=str, default="estimate", choices=["estimate", "evaluate"], help="Choose the task to perform.")
    parser.add_argument("--input_dir", type=str, default=cfg.INPUT_DIR, help="Directory containing input frames or estimated flows or gt flows.")
    parser.add_argument("--gt_dir", type=str, default=cfg.GT_DIR, help="Directory containing ground truth flows.")
    parser.add_argument("--solver", type=str, default="SOR", choices=list(cfg.SOLVERS.keys()), help="Choose the solver: SOR, GMRES, BICGSTAB, CG")
    args = parser.parse_args()

    sequences, sequence_keys = utils.process_files(args.input_dir)
    num_sequences = len(sequence_keys)
    print(f"Number of sequences: {num_sequences}")
    
    model = BroxOpticalFlow(num_levels=cfg.NUM_LEVELS, scaling_factor=cfg.SCALING_FACTOR, solver=args.solver)

    if args.task == "estimate":
        with mp.Pool(processes=mp.cpu_count()) as pool:
            pool.starmap(model._process_sequence, [(seq_key, sequences) for seq_key in sequence_keys])
        pool.close()
        pool.join()


    if args.task == "evaluate":
        avg_epe = []   
        num_outliers = 0
        num_valids = []  
        
        for sequence_key in sequence_keys:
            est_files = sequences[sequence_key]   # The estimated flows path (.flo format)
            gt_files = [os.path.join(args.gt_dir, os.path.basename(file)) for file in est_files]
            
            u_ests, v_ests, _ = utils.read_flow(est_files)
            u_gts, v_gts, valid_gts = utils.read_flow(gt_files)
            
            average_epe, pixwise_epe, outlier_count, valid_count = utils.compute_epe(u_ests, v_ests, u_gts, v_gts, valid_gts) # epe_full is a list of EPE maps for each sequence.

            utils.visualize_flow(u_ests, v_ests, est_files, valid_gts)
            utils.visualize_error(pixwise_epe, est_files, valid_gts)
            
            print(f"EPE for sequence {sequence_key}: {average_epe}\n\n")
            
            avg_epe.append(average_epe)
            num_valids.append(valid_count)
            num_outliers += outlier_count
            
        print(f"EPE_all: {np.sum(np.array(avg_epe) * np.array(num_valids)) / np.sum(num_valids)}")
        print(f"F1_all: {num_outliers / np.sum(num_valids)}")
        
if __name__ == "__main__":
    main()