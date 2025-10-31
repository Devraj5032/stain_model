import cv2
import time
import os
import numpy as np
from tflite_runtime.interpreter import Interpreter

# ---------------- Load TFLite Model ----------------
MODEL_PATH = "stain_detection_model.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- Define Classes ----------------
CLASSES = [
    "footprint",
    "glass",
    "stain",
    "stairs"
]  # 🧽 update this list with your own labels

# ---------------- Create output folder ----------------
os.makedirs("detections", exist_ok=True)

# ---------------- Select Camera ----------------
if os.name == "nt":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

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
        input_shape = input_details[0]['shape']  # [1, h, w, 3]
        h, w = input_shape[1], input_shape[2]
        input_data = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(input_data, axis=0)
        input_data = (np.float32(input_data) / 255.0)

        # ---------------- Run inference ----------------
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # ---------------- Get output ----------------
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Assuming output is [N, 6] → x1, y1, x2, y2, score, label
        detections = []

        if len(output_data.shape) == 2:
            for det in output_data:
                if len(det) >= 6:
                    x1, y1, x2, y2, score, label = det[:6]
                    if score > 0.5:
                        detections.append((x1, y1, x2, y2, score, int(label)))

        # ---------------- Draw & Save detections ----------------
        if detections:
            for (x1, y1, x2, y2, score, label) in detections:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                label_name = CLASSES[label] if label < len(CLASSES) else str(label)
                cv2.putText(frame, f"{label_name} {score:.2f}",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            filename = f"detections/detect_{save_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved detection: {filename}")
            save_count += 1

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    cap.release()
    print("✅ Exited cleanly.")
