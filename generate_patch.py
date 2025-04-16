import cv2
import json
import numpy as np
import os
from extract_patches import extract_patches

# Paths
clips_dir = "./data/clips/"
keypoints_dir = "./data/keypoints/"
patches_dir = "./data/patches/"

# Ensure the patches directory exists
os.makedirs(patches_dir, exist_ok=True)

def process_clip(clip_name):
    video_path = os.path.join(clips_dir, clip_name)
    keypoints_path = os.path.join(keypoints_dir, f"keypoints_{clip_name.split('.')[0].split('_')[1]}.json")
    patches_path = os.path.join(patches_dir, f"patches_{clip_name.split('.')[0].split('_')[1]}.npy")

    # Load keypoints
    with open(keypoints_path, "r") as f:
        keypoints_data = json.load(f)

    # Validate keypoints
    validated_keypoints = []
    for frame_idx, frame_data in enumerate(keypoints_data):
        if isinstance(frame_data, list) and len(frame_data) == 1 and len(frame_data[0]) == 18:
            validated_keypoints.append(frame_data[0])  # Extract the keypoints for the person
        else:
            print(f"Inconsistent keypoints in frame {frame_idx} of {clip_name}. Adding placeholder keypoints.")
            validated_keypoints.append([[0, 0]] * 18)  # Placeholder for missing/inconsistent frames

    keypoints = np.array(validated_keypoints).reshape(-1, 18, 2)  # Shape: (frames, 18, 2)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    patches_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the first 15 keypoints for each frame
        frame_keypoints = keypoints[frame_idx, :15, :]  # (15, 2)

        # Generate patches for the frame
        patches = extract_patches(frame, frame_keypoints)
        patches_list.append(patches)

        frame_idx += 1

    cap.release()

    # Save patches as a .npy file
    patches_list = np.array(patches_list)  # Shape: (frames, 15, patch_height, patch_width, 3)
    np.save(patches_path, patches_list)
    print(f"Patches saved to {patches_path}.")

# Process all clips in the clips directory
for clip in os.listdir(clips_dir):
    if clip.endswith(".mp4"):
        print(f"Processing {clip}...")
        process_clip(clip)
