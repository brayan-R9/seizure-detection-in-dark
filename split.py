import cv2
import os

def split_video(video_path, output_folder, clip_duration=5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_length = fps * clip_duration

    for i in range(0, frame_count, clip_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        output_clip = os.path.join(output_folder, f"clip_{i//clip_length}.mp4")
        out = cv2.VideoWriter(output_clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        for _ in range(clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
    cap.release()

video_pat = './data/Sz1PG.mp4'  # Replace with your video path
output_pat = './splited'  # Folder to save frames
split_video(video_pat, output_pat, clip_duration=5)