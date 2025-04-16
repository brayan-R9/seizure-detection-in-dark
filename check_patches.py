import numpy as np
import matplotlib.pyplot as plt
import os

# Directory where patches are saved
patches_dir = "./data/patches/"
patch_file = "patches_clip.npy"  # Change this to the file you want to inspect

# Load the patches
patch_path = os.path.join(patches_dir, patch_file)
patches = np.load(patch_path)  # Shape: (frames, 15, patch_height, patch_width, 3)

# Function to visualize patches for a frame
def visualize_frame_patches(patches, frame_idx):
    frame_patches = patches[frame_idx]  # Shape: (15, patch_height, patch_width, 3)

    plt.figure(figsize=(15, 10))
    for i in range(15):  # 15 patches per frame
        patch = frame_patches[i]
        plt.subplot(3, 5, i + 1)
        plt.imshow(patch.astype('uint8'))  # Ensure patch is in the right format
        plt.title(f"Patch {i+1}")
        plt.axis("off")
    plt.suptitle(f"Patches for Frame {frame_idx}")
    plt.show()

# Visualize patches for a specific frame
frame_idx = 50  # Adjust this to inspect different frames
visualize_frame_patches(patches, frame_idx)
