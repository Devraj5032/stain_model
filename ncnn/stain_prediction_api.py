from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import ncnn
import time
import cv2
import yaml
import torch

app = Flask(__name__)

# ================================
# 🔹 Load NCNN Model at Startup
# ================================
net = ncnn.Net()
net.load_param("./model.ncnn.param")
net.load_model("./model.ncnn.bin")

# Load metadata (contains class names, anchors, etc.)
with open("metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)
CLASS_NAMES = metadata.get("names", [])

# Define input and output names (change if your model uses different ones)
INPUT_NAME = "in0"
OUTPUT_NAME = "out0"

# ================================
# 🔹 Helper: Preprocess Image
# ================================
def preprocess_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # CHW
    return img

# ================================
# 🔹 Helper: Run Inference
# ================================
def infer_ncnn(pil_image):
    img = preprocess_image(pil_image)
    with net.create_extractor() as ex:
        ex.input(INPUT_NAME, ncnn.Mat(img))
        ret, out = ex.extract(OUTPUT_NAME)
        return np.array(out)

# ================================
# 🔹 Flask Prediction Route
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_name = file.filename
    image_bytes = file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))

    start_time = time.time()
    output = infer_ncnn(pil_image)
    inference_time = (time.time() - start_time) * 1000  # ms

    # NOTE ⚠️: This is placeholder logic.
    # Actual bounding boxes depend on your model output structure.
    # Once you show me what `output.shape` looks like, I’ll give you exact postprocessing.
    detections = []
    print("Model output shape:", output.shape)

    response = {
        "image_name": image_name,
        "model": "YOLOv11-NCNN",
        "detections": detections,
        "inference_time_ms": round(inference_time, 2),
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
