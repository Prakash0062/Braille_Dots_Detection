import io
import base64
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = YOLO(r"best.pt")

def detect_braille(img_bytes, conf_threshold=0.25):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)

    results = model(img_np, conf=conf_threshold)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            confidence = round(float(box.conf[0]), 2)
            label = model.names[int(box.cls[0])]
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            detections.append({
                'box': [x1, y1, x2, y2],
                'confidence': confidence,
                'label': label,
                'x_center': x_center,
                'y_center': y_center
            })

    if not detections:
        return "", []

    # Step 1: Sort detections by vertical position
    detections.sort(key=lambda d: d['y_center'])

    # Step 2: Group into rows manually
    row_thresh = 20  # Adjust this based on average line spacing
    rows = []
    current_row = []
    last_y = -100

    for det in detections:
        y = det['y_center']
        if abs(y - last_y) > row_thresh:
            if current_row:
                rows.append(current_row)
            current_row = [det]
            last_y = y
        else:
            current_row.append(det)
            last_y = (last_y + y) // 2  # smooth the row height

    if current_row:
        rows.append(current_row)

    # Step 3: Sort each row left to right
    detected_text_rows = []
    for row in rows:
        sorted_row = sorted(row, key=lambda d: d['x_center'])
        row_labels = [d['label'] for d in sorted_row]
        detected_text_rows.append(''.join(row_labels))

    # Step 4: Draw boxes and labels
    colors = [
        (0, 255, 0), (0, 0, 255), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (128, 0, 128), (0, 128, 128)
    ]
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['box']
        conf = det['confidence']
        label = det['label']
        color = colors[idx % len(colors)]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_np, f'{label} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode('utf-8')

    return img_str, detected_text_rows

@app.route('/')
def index():
    return render_template('index.html')

import logging

logging.basicConfig(level=logging.DEBUG)

@app.route('/detect', methods=['POST'])
def detect():
    logging.debug(f"Received /detect request with files: {request.files} and form: {request.form}")
    if 'image' not in request.files:
        logging.error("No image file uploaded in request")
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    conf_threshold = request.form.get('confidence', default=0.25, type=float)

    try:
        detected_image, detected_text_rows = detect_braille(image_bytes, conf_threshold)
        logging.debug(f"Detection successful, returning response")
        return jsonify({
            'detected_image': f'data:image/jpeg;base64,{detected_image}',
            'detected_text_rows': detected_text_rows
        })
    except Exception as e:
        logging.exception("Exception during detection")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

