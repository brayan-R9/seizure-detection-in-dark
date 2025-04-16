import torch
import numpy as np
import json
from alert import loud_siren,show_alert,flash_screen
from VSViG import VSViG_base

# Load Pretrained Model
model = VSViG_base()
model.load_state_dict(torch.load('./VSViG-base.pth'))
model.eval()  # Set model to evaluation mode

# Load Patches for a Test Clip
patches_path = './data/patches/patches_clip.npy'  # Change for other clips
patches = np.load(patches_path).astype(np.float32)  # Ensure float32
patches = torch.tensor(patches).permute(0, 1, 4, 2, 3).float()  # Convert and set dtype to float32

# Load Keypoints for the Same Clip
keypoints_path = './data/keypoints/keypoints.json'
with open(keypoints_path, 'r') as f:
    keypoints1 = json.load(f)

keypoints = [] #new part
for frame_kpts in keypoints1:
    if isinstance(frame_kpts, list) and len(frame_kpts) == 1 and len(frame_kpts[0]) == 18:
        keypoints.append(frame_kpts[0])  # Use valid keypoints
    else:
        print("Warning: Invalid keypoints detected. Adding placeholder keypoints.")
        keypoints.append([[0, 0]] * 18)  # Placeholder for missing keypoints

# Convert keypoints to float32 and ensure proper shape
keypoints = np.array(keypoints, dtype=np.float32)  # Ensure float32
keypoints = keypoints.reshape(-1, 15, 2)  # Shape: (Frames, 15, 2)
keypoints = torch.tensor(keypoints).float()  # Convert to float32

# ✅ FIX: Add a third "dummy" channel of zeros
dummy_channel = torch.zeros((keypoints.shape[0], keypoints.shape[1], 1), dtype=torch.float32)
keypoints = torch.cat([keypoints, dummy_channel], dim=-1)  # Shape: (Frames, 15, 3)

# Ensure patches and keypoints have the same number of frames
num_frames = min(len(patches), len(keypoints))
patches = patches[:num_frames]
keypoints = keypoints[:num_frames]

# ✅ FIX: Add missing batch dimension to both patches and keypoints
patches = patches.unsqueeze(0)  # Shape: (1, Frames, 15, 3, 32, 32)
keypoints = keypoints.unsqueeze(0)  # Shape: (1, Frames, 15, 3) ✅ Fixed!

#print(f"Final Shape of patches: {patches.shape}")   # Expected: (1, Frames, 15, 3, 32, 32)
#print(f"Final Shape of keypoints: {keypoints.shape}")  # Expected: (1, Frames, 15, 3)

# Inference
with torch.no_grad():
    seizure_probabilities = model(patches, keypoints)
    #print(f"Seizure Probabilities: {seizure_probabilities.squeeze().numpy()}")

# Display per-frame probabilities
#print(f"Seizure Probabilities: {seizure_probabilities.squeeze().tolist()}")
print(f"Seizure Probability aggregate: {seizure_probabilities.squeeze().numpy()}")
probabilities = seizure_probabilities.squeeze().tolist()
print(probabilities)

'''
# Alert System (Trigger when probability > 0.7)
ALERT_THRESHOLD = 0.7
if seizure_probabilities>ALERT_THRESHOLD:
    flash_screen()
    loud_siren()  # Play a beep sound on Windows
    show_alert()

'''
'''
for idx, prob in enumerate(probabilities):
    if prob > ALERT_THRESHOLD:
        print(f" ALERT: Seizure Probability {prob:.3f} exceeded threshold at index {idx}")
        flash_screen()
        loud_siren()  # Play a beep sound on Windows
        show_alert()
        break  # Stop checking after first alert
'''


