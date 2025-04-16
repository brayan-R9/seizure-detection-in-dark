import cv2
import numpy as np
import torch
import json
import os
import winsound
import time
import tkinter as tk
import plyer
from plyer import notification
from scipy import stats
from scipy.ndimage import uniform_filter1d
import subprocess
from alert import loud_siren,show_alert,flash_screen
from VSViG import VSViG_base  # Import seizure detection model
from gic_clahe import apply_gic_with_clahe  # Import enhancement functions
from extract_patches import extract_patches  # Patch extraction

# Define paths
IMAGE_DIR = "./datasets/images/"
CLIPS_DIR = "./data/clips/"
KEYPOINTS_DIR = "./data/keypoints/"
PATCHES_DIR = "./data/patches/"
MODELS_DIR = "./models/"
INPUT_VIDEO = "./data/input_video.mp4"

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(KEYPOINTS_DIR, exist_ok=True)
os.makedirs(PATCHES_DIR, exist_ok=True)

###  Video Enhancement & Save Frames for OpenPose
def enhance_video_and_save_frames(input_video_path, image_output_dir, output_clip_path, gamma=0.5):
    """
    Enhances video frame-by-frame using GIC + CLAHE and saves frames for OpenPose.
    Also saves the enhanced video for patch extraction.
    """
    
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_clip_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Enhance frame
        #enhanced_frame = apply_gic_with_clahe(frame, gamma)
        enhanced_frame = frame
       

        # Save frame as image for OpenPose
        frame_filename = os.path.join(image_output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, enhanced_frame)

        # Save frame to output video for patch extraction
        out.write(enhanced_frame)

        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}", end="\r")

    cap.release()
    out.release()
    print(f"\n Frames saved to {image_output_dir} & Enhanced video saved to {output_clip_path}!")

###  Run OpenPose to Extract Keypoints
def extract_keypoints_with_openpose(image_dir, output_json):
    """
    Run OpenPose (demo.py) on enhanced images.
    """
    command = f'python demo.py --checkpoint-path {MODELS_DIR}/pose.pth --images {image_dir}/*.jpg --output-file {output_json}'
    print(f" Running OpenPose on enhanced images...")

    try:
        subprocess.run(command, shell=True, check=True)
        print(f" Keypoints saved to: {output_json}")
    except subprocess.CalledProcessError as e:
        print(f" OpenPose failed: {e}")

###  Generate Patches for Seizure Detection
def generate_patches(clip_name, keypoints_file):
    """
    Generates patches from the enhanced video using extracted keypoints.
    """
    video_path = os.path.join(CLIPS_DIR, clip_name)
    patches_path = os.path.join(PATCHES_DIR, f"patches_{clip_name.split('.')[0].split('_')[1]}.npy")

    # Load keypoints
    with open(keypoints_file, "r") as f:
        keypoints_data = json.load(f)

    # Validate keypoints
    validated_keypoints = []
    for frame_data in keypoints_data:
        if isinstance(frame_data, list) and len(frame_data) == 1 and len(frame_data[0]) == 18:
            validated_keypoints.append(frame_data[0])  # Extract keypoints
        else:
            validated_keypoints.append([[0, 0]] * 18)  # Placeholder for missing frames

    keypoints = np.array(validated_keypoints).reshape(-1, 18, 2)  # Shape: (frames, 18, 2)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    patches_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_keypoints = keypoints[frame_idx, :15, :]  # Take first 15 keypoints
        patches = extract_patches(frame, frame_keypoints)
        patches_list.append(patches)

        frame_idx += 1

    cap.release()

    # Save patches as .npy file
    patches_list = np.array(patches_list)  # Shape: (frames, 15, patch_height, patch_width, 3)
    np.save(patches_path, patches_list)
    print(f"Patches saved to {patches_path}.")

    return patches_path

###  Run Seizure Detection Model
def detect_seizure(patches_path, keypoints_path, model):
    """ Run the seizure detection model """
    patches = np.load(patches_path).astype(np.float32)
    patches = torch.tensor(patches).permute(0, 1, 4, 2, 3).float() # (Frames, 15, 3, 32, 32)

    # Load and preprocess keypoints
    with open(keypoints_path, "r") as f:
        keypoints_data = json.load(f)

    keypoints = []
    for frame_kpts in keypoints_data:
        if isinstance(frame_kpts, list) and len(frame_kpts) == 1 and len(frame_kpts[0]) == 18:
            keypoints.append(frame_kpts[0])  # Use valid keypoints
        else:
            print("Warning: Invalid keypoints detected. Adding placeholder keypoints.")
            keypoints.append([[0, 0]] * 18)  # Placeholder for missing keypoints

    keypoints = np.array(keypoints, dtype=np.float32)  # Convert to float32
    keypoints = keypoints.reshape(-1, 15, 2)  # Ensure shape is (Frames, 15, 2)
    keypoints = torch.tensor(keypoints).float()  # Convert to PyTorch tensor

    # FIX: Add a dummy channel of zeros
    dummy_channel = torch.zeros((keypoints.shape[0], keypoints.shape[1], 1), dtype=torch.float32)
    keypoints = torch.cat([keypoints, dummy_channel], dim=-1)  # Shape: (Frames, 15, 3)


    # Match keypoints to patch frames
    #keypoints = keypoints[:patches.shape[0]]
    num_frames = min(len(patches), len(keypoints))
    patches = patches[:num_frames]
    keypoints = keypoints[:num_frames]

    with torch.no_grad():
        seizure_probabilities = model(patches.unsqueeze(0), keypoints.unsqueeze(0))

    return seizure_probabilities.squeeze().tolist()


###  Alerting System
def trigger_alert():
    """ Call all alerts when probability > 0.7 """
    print(f" ******* SEIZURE ALERT *******")
    flash_screen()
    loud_siren()  # Play a beep sound on Windows
    show_alert()

###  Main Function: Run Everything Together
def main(input_video):
    """ Full pipeline: Enhance Video → Extract Keypoints → Generate Patches → Detect Seizure → Alert """

    # Step 1: Enhance video & save frames for OpenPose
    enhanced_video = os.path.join(CLIPS_DIR, "enhanced_clip.mp4")
    enhance_video_and_save_frames(input_video, IMAGE_DIR, enhanced_video, gamma=0.5)

    # Step 2: Extract Keypoints using OpenPose
    keypoints_path = os.path.join(KEYPOINTS_DIR, "keypoints.json")
    extract_keypoints_with_openpose(IMAGE_DIR, keypoints_path)
    
    # Step 3: Generate Patches
    patches_path = generate_patches("enhanced_clip.mp4", keypoints_path)

    # Step 4: Load VSViG model
    model = VSViG_base()
    model.load_state_dict(torch.load('./VSViG-base.pth'))
    model.eval()

    # Step 5: Run seizure detection
    seizure_probs = detect_seizure(patches_path, keypoints_path, model)
    #print(f"Seizure Probabilities : {seizure_probs}")

    '''
    c = sum(1 for i in seizure_probs if i==1.0)
    if c > 1:
        mod = np.mean(seizure_probs) 
    else:
        mode_result = stats.mode(seizure_probs, keepdims=False)
        mod = mode_result.mode if hasattr(mode_result, "mode") else mode_result[0]
        mod = min(mod + 0.2, 1.0)

    print("seizure prob:",mod)'''

    def adaptive_smoothing(probs):
        window_size = 3 if np.mean(probs) < 0.5 else 5  # Dynamic window selection
        return uniform_filter1d(probs, size=window_size)

    
    p=cumulative_prob = np.mean(seizure_probs)
    print("Probability of Seizure:",cumulative_prob)

    
    # Step 6: Trigger Alert
    ALERT_THRESHOLD = 0.75
    if cumulative_prob > ALERT_THRESHOLD:
            trigger_alert()
            

### Run the pipeline
if __name__ == "__main__":
    input_video = "./data/clip_4.mp4"
    main(input_video)
