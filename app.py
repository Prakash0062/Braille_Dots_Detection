import io
import os
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import torch

app = Flask(__name__)

# ✅ Make sure model loads only once and runs on CPU
try:
    model = YOLO("best.pt")
    model.to(torch.device("cpu"))
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

def detect_braille(img_bytes):
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(image)

        # YOLO inference
        results = model(img_np)[0]

        detected_labels = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_labels.append(label)

            # Draw bounding box + label
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 0, 0), 2)

        # Convert image to byte stream
        _, img_encoded = cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        return io.BytesIO(img_encoded.tobytes()), "".join(detected_labels)

    except Exception as e:
        print("⚠️ Detection error:", e)
        return None, "Detection failed"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        img_bytes = request.files["image"].read()
        processed_img_io, detected_text = detect_braille(img_bytes)

        if processed_img_io is None:
            return jsonify({"error": "Detection failed"}), 500

        output_path = os.path.join("static", "output.png")
        with open(output_path, "wb") as f:
            f.write(processed_img_io.getbuffer())

        return jsonify({
            "image_url": "/" + output_path.replace("\\", "/"),
            "detected_text": detected_text
        })

    except Exception as e:
        print("⚠️ Upload handler error:", e)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=False, host="0.0.0.0", port=5000)
