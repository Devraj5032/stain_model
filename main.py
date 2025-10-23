from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import time

app = Flask(__name__)

# Load your trained YOLOv11 model once when the app starts
model = YOLO("yolov11.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_name = file.filename
    image_bytes = file.read()

    # Open image
    image = Image.open(io.BytesIO(image_bytes))

    # Run inference
    start_time = time.time()
    results = model(image)
    inference_time = (time.time() - start_time) * 1000  # ms

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

        detections.append({
            "class_id": cls_id,
            "class_name": model.names[cls_id],
            "confidence": round(conf, 3),
            "bbox": {
                "x_min": round(x_min, 2),
                "y_min": round(y_min, 2),
                "x_max": round(x_max, 2),
                "y_max": round(y_max, 2)
            }
        })

    response = {
        "image_name": image_name,
        "model": "yolov11.pt",
        "detections": detections,
        "inference_time_ms": round(inference_time, 2)
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
