import winsound
import time
import plyer
from plyer import notification
import tkinter as tk


def loud_siren():
    for _ in range(5):  # Play the sound 5 times
        winsound.Beep(1000, 700)  # Higher-pitched beep
        time.sleep(0.3)  # Short delay
        winsound.Beep(1500, 700)  # Even higher-pitched beep
        time.sleep(0.3)

def show_alert():
    notification.notify(
        title="🚨 SEIZURE ALERT!",
        message="Seizure probability exceeded threshold!",
        timeout=10,  # Display for 10 seconds
    )

def flash_screen():
    root = tk.Tk()
    root.attributes('-fullscreen', True)  # Fullscreen mode
    root.configure(bg='red')  # Red screen
    root.after(2000, root.destroy)  # Close after 2 seconds
    root.mainloop()
