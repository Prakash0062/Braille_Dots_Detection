import io
import base64
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLO model
model = YOLO("best.pt")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_braille(img_bytes, conf_threshold=0.25):
    """Detect braille characters in an image."""
    try:
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
            return "", ["No braille characters detected"]

        # Group detections into rows
        detections.sort(key=lambda d: d['y_center'])
        row_thresh = 20
        rows, current_row, last_y = [], [], -100

        for det in detections:
            y = det['y_center']
            if abs(y - last_y) > row_thresh:
                if current_row:
                    rows.append(current_row)
                current_row = [det]
                last_y = y
            else:
                current_row.append(det)
                last_y = (last_y + y) // 2

        if current_row:
            rows.append(current_row)

        detected_text_rows = []
        for row in rows:
            sorted_row = sorted(row, key=lambda d: d['x_center'])
            row_labels = [d['label'] for d in sorted_row]
            detected_text_rows.append(''.join(row_labels))

        # Draw boxes on image
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            color = (0, 255, 0)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')

        return img_str, detected_text_rows

    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        return "", [f"Detection error: {str(e)}"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    logging.debug("Received detection request")
    
    # Check if file was uploaded
    if 'image' not in request.files:
        logging.error("No image file in request")
        return jsonify({'error': 'No image file uploaded'}), 400
        
    image_file = request.files['image']
    
    # Validate file
    if not image_file or image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Read and validate image
        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty image file'}), 400
            
        # Get confidence threshold
        conf_threshold = request.form.get('confidence', default=0.25, type=float)
        
        # Process image
        detected_image, detected_text_rows = detect_braille(image_bytes, conf_threshold)
        
        return jsonify({
            'detected_image': f"data:image/jpeg;base64,{detected_image}" if detected_image else "",
            'detected_text_rows': detected_text_rows
        })
        
    except Exception as e:
        logging.exception("Error during detection")
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
