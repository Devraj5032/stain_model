import cv2
import time
import os
import numpy as np
import tensorflow as tf

# ---------------- Load TFLite Model ----------------
MODEL_PATH = "stain_detection_model.tflite"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

print(f"🧠 Loading TFLite model: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- Define Classes ----------------
CLASSES = ["footprint", "glass", "stain", "stairs"]

# ---------------- Create output folder ----------------
os.makedirs("detections", exist_ok=True)


# ---------------- List Available Cameras ----------------
def list_cameras(max_tested=10):
    """Scan for connected cameras."""
    print("🔍 Scanning for available cameras...")
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


# ---------------- Get Camera ----------------
cameras = list_cameras()
if not cameras:
    print("❌ No cameras detected. Please connect one and try again.")
    exit()

print("\n🎦 Available Cameras:")
for idx in cameras:
    print(f"  [{idx}] Camera {idx}")

while True:
    try:
        cam_index = int(input("\nEnter camera index to use: "))
        if cam_index in cameras:
            break
        else:
            print("⚠️ Invalid index. Try again.")
    except ValueError:
        print("⚠️ Please enter a valid number.")


# ---------------- Initialize Camera ----------------
print(f"\n📸 Starting camera {cam_index} ...")
cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError(f"❌ Failed to open camera {cam_index}")

# ---------------- FPS Control ----------------
target_fps = 6
frame_interval = 1.0 / target_fps
last_time = time.time()
save_count = 0

print(f"\n✅ Running real-time detection (~{target_fps} FPS). Press 'q' or 'Esc' to quit.\n")

# ---------------- Detection Loop ----------------
try:
    while True:
        if time.time() - last_time < frame_interval:
            continue
        last_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            continue

        # ---------------- Preprocess frame ----------------
        input_shape = input_details[0]['shape']  # [1, height, width, 3]
        h, w = input_shape[1], input_shape[2]

        resized = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

        # ---------------- Run inference ----------------
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        detections = []
        if len(output_data.shape) == 2:
            for det in output_data:
                if len(det) >= 6:
                    x1, y1, x2, y2, score, label = det[:6]
                    if score > 0.5:
                        detections.append((x1, y1, x2, y2, score, int(label)))

        # ---------------- Draw detections ----------------
        for (x1, y1, x2, y2, score, label) in detections:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label_name = CLASSES[label] if label < len(CLASSES) else str(label)
            cv2.putText(frame, f"{label_name} {score:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---------------- FPS Display ----------------
        fps = 1.0 / (time.time() - last_time + 1e-5)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # ---------------- Show Frame ----------------
        cv2.imshow("🧠 Real-Time Stain Detection", frame)

        # Save frame if detection
        if detections:
            filename = f"detections/detect_{save_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            save_count += 1

        # Exit
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            print("\n🛑 Exiting...")
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cleanup complete. Exited cleanly.")
