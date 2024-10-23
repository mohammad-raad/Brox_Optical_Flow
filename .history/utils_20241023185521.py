import os
import numpy as np
import config as cfg
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt




def save_flow(u_ests, v_ests, filenames, format):
    """Save optical flow to .flo or .png file."""
    
    dest_dir = os.path.join(cfg.RESULTS_DIR, "est_flows")
    os.makedirs(dest_dir, exist_ok=True)
    
    if format == "flo":
        for i, filename in enumerate(filenames):
            flow = np.dstack((u_ests[i], v_ests[i])).astype(np.float32)
            flow_file_path = os.path.join(dest_dir, f"{filename}.flo")
            with open(flow_file_path, 'wb') as f:
                # Write the header: magic number
                f.write(np.array([202021.25], dtype=np.float32).tobytes())
                height, width = flow.shape[:2]
                # Write the width and height
                f.write(np.array([width, height], dtype=np.int32).tobytes())
                # Write the flow data
                flow.tofile(f)

    elif format == "png":
        for i, filename in enumerate(filenames):
            # Scale u and v to the range [0, 65535] for 16-bit PNG
            u_scaled = np.clip((u_ests[i] * 64.0 + 2**15), 0, 65535).astype(np.uint16)
            v_scaled = np.clip((v_ests[i] * 64.0 + 2**15), 0, 65535).astype(np.uint16)

            # Create a validity mask (all ones indicating valid flow)
            valid_mask = np.ones_like(u_scaled, dtype=np.uint16)

            # Stack u, v, and the validity mask into a 3-channel image
            flow_png = np.dstack((u_scaled, v_scaled, valid_mask))

            # Save the image as a 16-bit PNG
            flow_file_path = os.path.join(dest_dir, f"{filename}.png")
            cv2.imwrite(flow_file_path, flow_png)

    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are 'flo' and 'png'.")


def read_flow(flow_sequence):
    format = flow_sequence[0].split('.')[-1]
    if not format in ["png", "flo"]:
        raise ValueError(f"Unsupported file format: {format}. Supported formats are 'flo' and 'png'.")

    u_ext = []
    v_ext = []
    validity_ext = []

    if format == "flo":
        for flow_file in flow_sequence:
            with open(flow_file, 'rb') as f:
                # Read the header
                magic = np.frombuffer(f.read(4), dtype=np.float32)[0]
                if magic != 202021.25:
                    raise ValueError(f"Invalid .flo file: {flow_file}")

                width, height = np.frombuffer(f.read(8), dtype=np.int32)
                flow_data = np.frombuffer(f.read(), dtype=np.float32).reshape((height, width, 2))

                u = flow_data[:, :, 0]
                v = flow_data[:, :, 1]

                # Identify and handle invalid flow vectors
                invalid = np.logical_or(u>1e+3, v>1e+3)  # Check for NaN (Not a Number) which can also represent invalid data.
                
                u_ext.append(u)
                v_ext.append(v)
                validity_ext.append(~invalid) # Store validity information

                
    elif format == "png":
        for flow_file in flow_sequence:
            # Read the flow image as a 16-bit image
            flow_png = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED)
            
            # Extract the three channels
            u_raw = flow_png[:, :, 2].astype(np.float32) 
            v_raw = flow_png[:, :, 1].astype(np.float32)
            valid = flow_png[:, :, 0].astype(bool)

            # Convert from PNG integer values to flow values
            u = (u_raw - 2**15) / 64.0
            v = (v_raw - 2**15) / 64.0
            
            # Set flow to zero for invalid pixels
            u[~valid] = 0
            v[~valid] = 0

            # Store u and v components of the flow
            u_ext.append(u)
            v_ext.append(v)
            validity_ext.append(valid)
    
    return u_ext, v_ext, validity_ext
  
  
def visualize_flow(u_ests, v_ests, filenames, valid_gts, percentile=90):
    dest_dir = os.path.join(cfg.RESULTS_DIR, "visual_flows")
    os.makedirs(dest_dir, exist_ok=True)

    for i, (u, v, filename, valid_mask) in enumerate(zip(u_ests, v_ests, filenames, valid_gts)):
        flow = np.stack([u, v], axis=-1)
        magnitude, angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

        v_max = np.percentile(magnitude, percentile)
        v_max = max(v_max, 1e-5)  

        magnitude = np.clip(magnitude, 0, v_max)
        magnitude = (magnitude / v_max) * 255  

        angle = angle * 180 / np.pi 
        angle = angle / 2

        hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle.astype(np.uint8)  # Hue: direction
        hsv[..., 1] = magnitude.astype(np.uint8)*0.5  # Saturation
        hsv[..., 2] = 255  # Value
        


        flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if valid_mask is not None:
            invalid_pixels = ~valid_mask
            flow_image[invalid_pixels] = [128, 128, 128]  # Set invalid pixels to gray

        output_filename = os.path.join(dest_dir, os.path.splitext(os.path.basename(filename))[0] + "_flow.png")
        cv2.imwrite(output_filename, flow_image)

        if valid_mask is not None:
            mask_filename = os.path.join(dest_dir, os.path.splitext(os.path.basename(filename))[0] + "_mask.png")
            cv2.imwrite(mask_filename, (valid_mask * 255).astype(np.uint8))


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



def visualize_error(sequence_epe, filenames, valid_gts, max_err=20):
    dest_dir = os.path.join(cfg.RESULTS_DIR, "visual_errors")
    os.makedirs(dest_dir, exist_ok=True)
    
    for epe, filename, valid_gt in zip(sequence_epe, filenames, valid_gts):
        normalized_epe = np.clip(epe / max_err, 0, 1)
        error_map = (normalized_epe * 255).astype(np.uint8)
        
        # Create a white background
        error_image = np.ones((error_map.shape[0], error_map.shape[1], 3), dtype=np.uint8) * 255
        
        # Set the error pixels to red with varying intensity
        error_image[..., 0] = 255  # Red channel maximum
        error_image[..., 1] = 255 - error_map  # Green channel inverse of error
        error_image[..., 2] = 255 - error_map  # Blue channel inverse of error
        
        # Set non-valid pixels to gray
        invalid_pixels = ~valid_gt
        error_image[invalid_pixels] = [128, 128, 128]
        
        # Create a figure matching the error map size and add the color bar
        fig, ax = plt.subplots(figsize=(error_map.shape[1] / 100, error_map.shape[0] / 100), dpi=100)
        ax.imshow(error_image)
        ax.axis('off')  # Hide axes

        # Create a red color bar that matches the error map
        norm = Normalize(vmin=0, vmax=max_err)
        sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        sm.set_array([])

        # Add the color bar, making it smaller in height to match the image
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)  # Adjust fraction to control size
        cbar.set_label('(EPE)', rotation=270, labelpad=15)
        cbar.set_ticks([0, max_err / 2, max_err])
        cbar.set_ticklabels([f'0', f'{max_err / 2:.1f}', f'{max_err}'])

        # Save the figure with the color bar, keeping the image size unchanged
        output_filename = os.path.join(dest_dir, f"{os.path.splitext(os.path.basename(filename))[0]}_error.png")
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()




            
def compute_epe(u_ests, v_ests, u_gts, v_gts, valid_gts):
    avg_epe = [] # Stacking average EPE for each frame.
    pixwise_epe = []
    num_outliers = 0
    num_valids = []
    
    for u_est, v_est, u_gt, v_gt, valid_gt in zip(u_ests, v_ests, u_gts, v_gts, valid_gts):
        epe = np.sqrt((u_est - u_gt)**2 + (v_est - v_gt)**2)
        gt_magnitude = np.sqrt(u_gt**2 + v_gt**2)
        relative_error = epe / (gt_magnitude + 1e-6)    # Add a small value to avoid division by zero
        
        outliers_mask = ((epe > 3) | (relative_error > 0.05)) & valid_gt   # True if the pixel is an outlier.
        
        d0_10_mask = (gt_magnitude <=10) & valid_gt
        d10_40_mask = (gt_magnitude > 10) & (gt_magnitude <= 40) & valid_gt
        d40_inf_mask = (gt_magnitude > 40) & valid_gt
        
        print(f"d0_10: {np.sum(d0_10_mask)}, d10_40: {np.sum(d10_40_mask)}, d40_inf: {np.sum(d40_inf_mask)}, valid: {np.sum(valid_gt)/valid_gt.size*100:.2f}%")
        
        avg_epe.append(epe[valid_gt].mean())
        pixwise_epe.append(epe)
        
        num_outliers += np.sum(outliers_mask)
        num_valids.append(np.sum(valid_gt))
        
        d0_10_epe = epe[d0_10_mask & valid_gt].mean()
        d10_40_epe = epe[d10_40_mask & valid_gt].mean()
        d40_inf_epe = epe[d40_inf_mask & valid_gt].mean()
        print(f"d0_10_epe: {d0_10_epe}, d10_40_epe: {d10_40_epe}, d40_inf_epe: {d40_inf_epe}")
        
    avg_epe = np.sum(np.array(avg_epe) * np.array(num_valids)) / np.sum(num_valids)

    return avg_epe, pixwise_epe, num_outliers, np.sum(num_valids)


def img_scaling(image, scale_factor, method='bilinear'):
    #scaled_image = ndimage.zoom(image, (scale_factor, scale_factor), order=1 if method == 'bilinear' else 0, mode='nearest')
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled_image


def psi_derivative(x, epsilon=1e-4):
    return 0.5 / np.sqrt(x + epsilon)


def process_files(input_dir):
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.png', '.flo'))])
    sequences = defaultdict(list)
    
    for file in input_files:
        sequence_number = os.path.basename(file).split('_')[0]
        sequences[sequence_number].append(file)
        
    sequence_keys = list(sequences.keys())
    return sequences, sequence_keys


