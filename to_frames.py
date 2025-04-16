import cv2
import os

# Input video path
video_path = './data/clips/enhanced_clip.mp4'  # Replace with your video path
output_folder = './datasets/images'  # Folder to save frames

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file!")
    exit()

# Extract frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Save frame as image
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    print(f"Saved: {frame_filename}")
    frame_count += 1

# Release resources
cap.release()
print(f"Frames extracted: {frame_count}")
print(f"Frames saved in folder: {output_folder}")
