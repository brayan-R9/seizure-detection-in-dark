import cv2
import tkinter as tk
from tkinter import Label, Button, Text
from PIL import Image, ImageTk
import threading
import subprocess
import os

class SeizureDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Seizure Detection System")

        # Title Label
        self.title_label = Label(root, text="Seizure Detection System", font=("Arial", 16))
        self.title_label.pack(pady=10)

        # Video Display Label
        self.video_label = Label(root)
        self.video_label.pack()

        # Start Processing Button
        self.process_button = Button(root, text="Start Processing", command=self.start_processing, state=tk.NORMAL)
        self.process_button.pack()

        # Close Application Button
        self.close_button = Button(root, text="Close Application", command=self.close_application, fg="white", bg="red")
        self.close_button.pack(pady=10)

        # Console Output
        self.console_output = Text(root, height=10, width=80)
        self.console_output.pack(pady=10)

        # Predefined Video Path (Change this to your actual video file path)
        self.video_path = "E:/vsvig_integrated/clips/clip_4_dark.mp4"

        self.cap = None
        self.playing = False
        self.fps = 30  # Default FPS (will be updated when video is loaded)

        # Start video automatically
        self.play_video()

    def play_video(self):
        """Automatically plays the predefined video."""
        if not self.video_path:
            return

        self.cap = cv2.VideoCapture(self.video_path)
        self.playing = True

        if self.cap.isOpened():
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        def update_frame():
            if not self.playing or not self.cap.isOpened():
                return

            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 360))  # Adjust video size for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(img)
                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk

                # Schedule next frame
                delay = int(1000 / self.fps)  # Convert FPS to milliseconds
                self.root.after(delay, update_frame)
            else:
                self.cap.release()
                self.playing = False

        update_frame()

    def start_processing(self):
        """Starts processing the predefined video."""
        if not self.video_path:
            return
        self.console_output.insert(tk.END, "Processing started...\n")
        self.console_output.see(tk.END)

        def run_pipeline():
            process = subprocess.Popen(
                ["python", "pipeline.py", self.video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )

            for line in iter(process.stdout.readline, ""):
                self.console_output.insert(tk.END, line)
                self.console_output.see(tk.END)
                self.root.update_idletasks()

            process.stdout.close()
            process.wait()
            self.console_output.insert(tk.END, "Processing finished.\n")

        threading.Thread(target=run_pipeline, daemon=True).start()

    def close_application(self):
        """Fully terminates the application."""
        self.playing = False
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()
        os._exit(0)

# Run the application
root = tk.Tk()
app = SeizureDetectionApp(root)
root.mainloop()
