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
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB file size limit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Model loading with optimizations
def load_model():
    try:
        import torch
        torch.set_grad_enabled(False)  # Disable gradients for inference
        
        # Load with reduced memory footprint
        model = YOLO("best.pt", task='detect')
        
        # Optimize model
        model.fuse()  # Fuse conv and bn layers
        model.eval()  # Set to evaluation mode
        
        # Warmup run
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(dummy_img, verbose=False)
        
        return model
    except Exception as e:
        logging.error(f"Model loading failed: {str(e)}")
        raise

# Global model variable
try:
    model = load_model()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    model = None

def process_detections(results):
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
    return detections

def group_into_rows(detections):
    if not detections:
        return []
    
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
    
    return rows

def draw_detections(img_np, detections):
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        color = (0, 255, 0)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_np, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if not model:
        return jsonify({'error': 'Model not available'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    try:
        start_time = time.time()
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Validate image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(img)
        
        # Resize if too large (keep aspect ratio)
        h, w = img_np.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_np = cv2.resize(img_np, (new_w, new_h))
        
        # Process with timeout
        conf_threshold = min(max(float(request.form.get('confidence', 0.25)), 0, 1)
        results = model(img_np, conf=conf_threshold, verbose=False)
        
        # Process results
        detections = process_detections(results)
        rows = group_into_rows(detections)
        
        detected_text_rows = []
        for row in rows:
            sorted_row = sorted(row, key=lambda d: d['x_center'])
            row_labels = [d['label'] for d in sorted_row]
            detected_text_rows.append(''.join(row_labels))
        
        # Draw detections
        draw_detections(img_np, detections)
        
        # Encode result image
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        logging.info(f"Detection completed in {time.time()-start_time:.2f}s")
        return jsonify({
            'detected_image': f"data:image/jpeg;base64,{img_str}",
            'detected_text_rows': detected_text_rows or ["No braille detected"]
        })

    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
