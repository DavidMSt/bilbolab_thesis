import os
import threading
import time
import cv2
from datetime import datetime
from pathlib import Path

from robot.sensing.camera.pycamera import PyCameraType, PyCamera

# === Settings ===
SAVE_FOLDER = Path("captured_images")
CAPTURE_INTERVAL = 0.1  # seconds
CAMERA_TYPE = PyCameraType.GS
RESOLUTION = (1456, 1088)

# === Ensure save folder exists ===
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

# === Initialize camera ===
camera = PyCamera(CAMERA_TYPE, RESOLUTION)
camera.start()

# === Capture Thread Function ===
capturing = False
stop_event = threading.Event()

def capture_loop():
    counter = 0
    while not stop_event.is_set():
        frame = camera.takeFrame()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = SAVE_FOLDER / f"image_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        counter += 1
        time.sleep(CAPTURE_INTERVAL)

# === Main control loop ===
print("Press [Enter] to START/STOP capturing images. Press [Ctrl+C] to exit.")

capture_thread = None

try:
    while True:
        input()  # Wait for Enter
        capturing = not capturing

        if capturing:
            print("🟢 Started capturing images...")
            stop_event.clear()
            capture_thread = threading.Thread(target=capture_loop)
            capture_thread.start()
        else:
            print("🔴 Stopping capture...")
            stop_event.set()
            capture_thread.join()
            print("✅ Capture stopped.")

except KeyboardInterrupt:
    print("\nExiting...")
    if capturing:
        stop_event.set()
        capture_thread.join()
    camera.picam.stop()

