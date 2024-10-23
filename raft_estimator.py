import torch
import torchvision
import os
import argparse
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.io import read_image
from torchvision import transforms
import utils

def predict_flow(img1, img2, model, original_size=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # Add batch dimension if not present
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # Predict flow
    with torch.no_grad():
        flow_predictions = model(img1, img2)
        flow = flow_predictions[-1]  # Get the final flow prediction
        
        # Resize flow back to original dimensions if needed
        if original_size is not None:
            flow = torch.nn.functional.interpolate(flow, size=original_size, mode='bilinear', align_corners=False)
            
            # Scale the flow values according to the resize ratio
            original_h, original_w = original_size
            current_h, current_w = flow.shape[2:]
            flow[:, 0] *= (original_w / current_w)
            flow[:, 1] *= (original_h / current_h)
            
    return flow


def main():
    
    parser = argparse.ArgumentParser(description='Compute optical flow using RAFT model')
    parser.add_argument("--img1_path", type=str, required=True, help="Path to first image")
    parser.add_argument("--img2_path", type=str, required=True, help="Path to second image")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for flow visualization")
    args = parser.parse_args()

    if not os.path.exists(args.img1_path):
        raise FileNotFoundError(f"First image not found: {args.img1_path}")
    if not os.path.exists(args.img2_path):
        raise FileNotFoundError(f"Second image not found: {args.img2_path}")

    try:
        # Load RAFT model
        weights = Raft_Large_Weights.DEFAULT
        model = torchvision.models.optical_flow.raft_large(weights=weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        img1 = read_image(args.img1_path)/255.0
        img2 = read_image(args.img2_path)/255.0
        
        # Store original dimensions
        original_size = (img1.shape[1], img1.shape[2])

        # Pad images to make dimensions divisible by 8
        h, w = img1.shape[1:3]
        new_h = ((h + 7) // 8) * 8
        new_w = ((w + 7) // 8) * 8
        transform = transforms.Resize((new_h, new_w))
        img1_resized = transform(img1)
        img2_resized = transform(img2)

        # Compute flow with original size for rescaling
        flow = predict_flow(img1_resized, img2_resized, model, original_size=original_size)
        
        flow = flow.cpu().numpy()
        u = flow[0, 0].squeeze()  # First channel (horizontal flow)
        v = flow[0, 1].squeeze()  # Second channel (vertical flow)
        
        u_ests = [u]
        v_ests = [v]
        filenames = [os.path.basename(args.img1_path)]
        valid_gts = [None]  # No validity mask in this case
        
        utils.save_flow(u_ests, v_ests, filenames, format="flo")
        print(f"Flow computation and visualization completed successfully")
        return 0

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return 1

if __name__ == "__main__":
    main()