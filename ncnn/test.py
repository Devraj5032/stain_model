import cv2
import time
import os
import numpy as np
import ncnn

# ---------------- Load NCNN Model ----------------
net = ncnn.Net()
net.load_param("model.ncnn.param")   # update if needed
net.load_model("model.ncnn.bin")     # update if needed

# ---------------- Create output folder ----------------
os.makedirs("detections", exist_ok=True)

# ---------------- Select Camera ----------------
# For Windows
if os.name == "nt":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# For Jetson Nano / Raspberry Pi
else:
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------- FPS Control ----------------
target_fps = 6
frame_interval = 1.0 / target_fps
last_time = time.time()

print("✅ Running real-time detection (~6 FPS). Press Ctrl+C to stop.")

save_count = 0

try:
    while True:
        # Maintain ~6 FPS
        if time.time() - last_time < frame_interval:
            continue
        last_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            continue

        # ---------------- Preprocess frame ----------------
        ncnn_mat = ncnn.Mat.from_pixels_resize(
            frame, ncnn.Mat.PixelType.PIXEL_BGR,
            frame.shape[1], frame.shape[0],
            640, 640
        )

        # ---------------- Run inference ----------------
        ex = net.create_extractor()
        ex.input("in0", ncnn_mat)   # change to your model’s input blob name
        ret, output = ex.extract("out0")  # change to your model’s output blob name

        detections = []
        for i in range(output.h):
            values = np.array(output.row(i)).flatten()
            if len(values) >= 6:
                x1, y1, x2, y2, score, label = values[:6]
                if score > 0.5:
                    detections.append((x1, y1, x2, y2, score, int(label)))

        # ---------------- Draw & Save detections ----------------
        if detections:
            for (x1, y1, x2, y2, score, label) in detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{label} {score:.2f}",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            filename = f"detections/detect_{save_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved detection: {filename}")
            save_count += 1

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    cap.release()
    print("✅ Exited cleanly.")
