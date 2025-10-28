import cv2
import time
import numpy as np
import ncnn
from PIL import Image

# ✅ Load NCNN model
net = ncnn.Net()
net.load_param("./model.ncnn.param")
net.load_model("./model.ncnn.bin")

# Define input and output layer names (from your NCNN export)
INPUT_NAME = "in0"
OUTPUT_NAME = "out0"

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # CHW
    return img

def infer(frame):
    img = preprocess_frame(frame)

    with net.create_extractor() as ex:
        ex.input(INPUT_NAME, ncnn.Mat(img))
        ret, out = ex.extract(OUTPUT_NAME)
        return np.array(out)

def real_time_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        output = infer(frame)
        inference_time = (time.time() - start_time) * 1000

        # TODO: Post-process detections (depends on your model output)
        # For now, just show frame and FPS
        fps = 1000 / inference_time if inference_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("NCNN Real-time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
