import torch
import json
import os
import numpy as np
from VSViG import VSViG_base

# Paths
patches_dir = "./data/patches/"
keypoints_dir = "./data/keypoints/"
model_path = "VSViG-base.pth"

# Load the pre-trained model
model = VSViG_base()
model.load_state_dict(torch.load(model_path))
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

#print(f"before_patches_shape: {patches.shape}")
#print(f"before_keypoint_shape: {keypoints.shape}")
# Inference on a single clip
def infer_clip(clip_id):
    # Load patches and keypoints
    patches_path = f"{patches_dir}/patches_{clip_id}.npy"
    keypoints_path = f"{keypoints_dir}/keypoints_{clip_id}.json"

    patches = np.load(patches_path)  # Shape: (frames, 15, patch_height, patch_width, 3)
    with open(keypoints_path, "r") as f:
        keypoints = np.array(json.load(f)).reshape(-1, 18, 2)[:, :15, :]  # Shape: (frames, 15, 2)

    print(f"before_patches_shape: {patches.shape}")
    print(f"before_keypoint_shape: {keypoints.shape}")

    # Reshape and permute data
    data = torch.tensor(patches).permute(0, 1, 4, 2, 3).float()  # Shape: (frames, 15, 3, height, width)
    data = data.unsqueeze(0)  # Add batch dimension: (1, frames, 15, 3, height, width)
    keypoints = torch.tensor(keypoints).float().unsqueeze(0)  # Shape: (1, frames, 15, 2)

    if torch.cuda.is_available():
        data, keypoints = data.cuda(), keypoints.cuda()

    # Debug shapes
    print(f"Reshaped data shape: {data.shape}, Keypoints shape: {keypoints.shape}")

    # Run inference
    with torch.no_grad():
        output = model(data, keypoints)  # Output is seizure probabilities

    return output.cpu().numpy()

# Test on all clips
for clip_id in range(len(os.listdir(patches_dir))):
    print(f"Running inference on Clip {clip_id}...")
    seizure_probabilities = infer_clip(str(clip_id).zfill(1))
    print(f"Seizure Probabilities for Clip {clip_id}: {seizure_probabilities}")
