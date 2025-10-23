
This document explains how to set up and run a Flask-based API for YOLOv11 object detection.


The API accepts an image and returns structured JSON predictions including detected objects, confidence scores, and bounding box coordinates.
 
 
Features
 • Accepts image uploads via HTTP POST
 • Runs inference using YOLOv11
 • Returns clean, structured JSON predictions
 • Includes inference time (in milliseconds)
 • Easy to extend for web apps or dashboards


 Requirements
 • Python 3.8+
 • pip (Python package manager)
 Installation Steps
 • 1. Clone the repository
 • 2. Install dependencies: pip install -r requirements.txt
 • 3. Place your YOLOv11 model file (yolov11.pt) in project root
 • 4. Run the Flask app: python app.py
 • 5. Access API at http://localhost:5000


 API Usage
 • Endpoint: POST /predict
 • Send image using multipart/form-data


 Sample JSON Response
 {
  "image_name": "test.jpg",
  "model": "yolov11.pt",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.94,
      "bbox": { "x_min": 120.3, "y_min": 180.1, "x_max": 240.7, "y_max": 410.2 }
    },
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.88,
      "bbox": { "x_min": 260.0, "y_min": 200.5, "x_max": 520.3, "y_max": 390.4 }
    }
  ],
  "inference_time_ms": 45.23
 }