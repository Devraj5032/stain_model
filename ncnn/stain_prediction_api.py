import cv2
import time
import os
import psutil

# Prevent Wayland/Qt errors on Raspberry Pi
os.environ["QT_QPA_PLATFORM"] = "offscreen"

print("🎥 Starting camera preview locked at 6 FPS... Press 'q' to quit.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit(1)

TARGET_FPS = 6
FRAME_INTERVAL = 1.0 / TARGET_FPS  # ≈0.166 seconds

last_time = time.time()
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # FPS regulation: wait if frame is too early
    now = time.time()
    elapsed = now - last_time
    if elapsed < FRAME_INTERVAL:
        time.sleep(FRAME_INTERVAL - elapsed)
    last_time = time.time()

    # Count FPS over time (for display only)
    frame_count += 1
    duration = time.time() - start_time
    if duration >= 1:
        fps_display = frame_count / duration
        frame_count = 0
        start_time = time.time()
    else:
        fps_display = TARGET_FPS

    # System stats
    temps = os.popen("vcgencmd measure_temp").read().strip().replace("temp=", "")
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent

    text = f"FPS: {fps_display:.1f} | CPU: {cpu:.1f}% | MEM: {mem:.1f}% | {temps}"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Camera Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
