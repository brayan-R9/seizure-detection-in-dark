'''import cv2
import numpy as np
import json

# Paths
video_path = './data/clips/clip_5.mp4'
keypoints_path = './data/keypoints/keypoints_5.json'

# Load video and keypoints
cap = cv2.VideoCapture(video_path)
with open(keypoints_path, "r") as f:
    keypoints = json.load(f)

print(len(keypoints))  # Number of elements
print([len(kp) for kp in keypoints])  # Length of each element

#print(keypoints)  # Actual data

keypoints = np.array(keypoints).reshape(-1, 18, 2)  # Ensure proper shape

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Debug: Print keypoints for the current frame
    print(f"Keypoints for Frame {frame_idx}: {keypoints[frame_idx]}")

    for keypoint in keypoints[frame_idx]:
        if len(keypoint) >= 2:  # Ensure the keypoint has at least x and y values
            x, y = keypoint[:2]  # Extract x and y coordinates
            if x is None or y is None or np.isnan(x) or np.isnan(y):  # Skip invalid keypoints
                continue
            x, y = int(float(x)), int(float(y))  # Convert to integers
            if x < 0 or y < 0:  # Skip keypoints with negative coordinates
                continue
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Keypoints", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()'''

import cv2
import numpy as np
import json

# Paths
video_path = './data/clips/enhanced_clip.mp4'
keypoints_path = './data/keypoints/keypoints.json'

# Load video and keypoints
cap = cv2.VideoCapture(video_path)
with open(keypoints_path, "r") as f:
    keypoints = json.load(f)

print(f"Total frames in keypoints: {len(keypoints)}")
print([len(kp) for kp in keypoints])  # Length of each keypoint entry

# Fallback: Ensure all frames have proper keypoints
processed_keypoints = []
for frame_kpts in keypoints:
    if isinstance(frame_kpts, list) and len(frame_kpts) == 1 and len(frame_kpts[0]) == 18:
        processed_keypoints.append(frame_kpts[0])  # Use valid keypoints
    else:
        print("Warning: Invalid keypoints detected. Adding placeholder keypoints.")
        processed_keypoints.append([[0, 0]] * 18)  # Placeholder for missing keypoints

keypoints = np.array(processed_keypoints).reshape(-1, 18, 2)  # Reshape to (frames, 18, 2)
print(keypoints.shape)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Debug: Print keypoints for the current frame
    print(f"Keypoints for Frame {frame_idx}: {keypoints[frame_idx]}")

    for keypoint in keypoints[frame_idx]:
        if len(keypoint) >= 2:  # Ensure the keypoint has at least x and y values
            x, y = keypoint[:2]  # Extract x and y coordinates
            if x is None or y is None or np.isnan(x) or np.isnan(y):  # Skip invalid keypoints
                continue
            x, y = int(float(x)), int(float(y))  # Convert to integers
            if x < 0 or y < 0:  # Skip keypoints with negative coordinates
                continue
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Keypoints", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
