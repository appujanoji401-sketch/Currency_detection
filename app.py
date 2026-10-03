"""
Real-Time Indian Currency Recognition System
Flask Backend API for currency detection with voice output
"""

from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
from collections import Counter, deque
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables
model = None
detection_active = False
latest_detections = []
latest_total_value = 0

# Temporal majority voting: under webcam light the model flips frame-to-frame
# between the visually similar same-size notes (200/500/10). Requiring N
# identical consecutive frames silenced everything (measured live: 1 detection
# in 30s with a held note). Instead we vote over a sliding window of the last
# VOTE_WINDOW frames and report the class(es) with the most frames inside it
# (plurality), provided they reach MIN_VOTES. Coins are detected only in a
# minority of frames at webcam distance, so the bar stays low; the plurality
# rule is what outvotes the 200/500/10 note flips.
VOTE_WINDOW = 10
MIN_VOTES = 2
_frame_history = deque(maxlen=VOTE_WINDOW)
_last_box = {}

# Per-class confidence floors (measured live): body/clothing false positives
# fired ONLY as the 500 note class at conf 0.31-0.39. Other notes (e.g. 200,
# which detects weakly at webcam range) and coins keep the low 0.3 floor.
# For the 500 class the floor is ADAPTIVE: once a genuine 500 note is shown,
# real detections cluster far above the FP band, so the floor drops to just
# below the lowest genuine 500 confidence seen.
NOTE_CLASSES = {0, 1, 2, 3, 4, 5}
MIN_CONF_COINS = 0.3
INITIAL_500_FLOOR = 0.6    # until a genuine 500 is seen (real ones measure 0.8+)
FP_500_MAX_CONF = 0.39     # top of measured body-FP band for class 5
_conf500_min_seen = None   # lowest conf of genuine 500 detections
_conf500_obs = []          # recent 500 detections for hysteresis

# Currency denomination mapping
# Matches final_dataset classes 0-10 (11 denominations)
# Rs 1/2/5 notes and Rs 2000 note are intentionally excluded (obsolete/withdrawn)
CURRENCY_NAMES = {
    0: '10 Rupees Note', 1: '20 Rupees Note', 2: '50 Rupees Note', 3: '100 Rupees Note',
    4: '200 Rupees Note', 5: '500 Rupees Note',
    6: '1 Rupee Coin', 7: '2 Rupees Coin', 8: '5 Rupees Coin', 9: '10 Rupees Coin', 10: '20 Rupees Coin'
}

# Currency value mapping (class ID to numerical value in Rupees)
CURRENCY_VALUES = {
    0: 10, 1: 20, 2: 50, 3: 100, 4: 200, 5: 500,
    6: 1, 7: 2, 8: 5, 9: 10, 10: 20
}

#Loads the trained YOLOv8 model by checking the available model paths and initializes the model for currency detection
def load_model():
    """Load the trained YOLO model"""
    global model
    
    # Check for trained model (newest training run first)
    model_paths = [
        'runs/detect/currency_detection/weights/best.pt',
        'runs/detect/runs/detect/currency_detection/weights/best.pt',
        'models/best.pt',
        'yolov8n.pt'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Loading model from: {path}")
            model = YOLO(path)
            return True
    
    print("No trained model found. Please train the model first using train_model.py")
    return False

# Processes each camera frame using YOLOv8, identifies all visible currency objects, maps their class IDs to denominations,
#  draws bounding boxes and calculates the total values.

def detect_currency(frame):
    """Detect currency in the given frame and return all detections with total value"""
    global model, _conf500_obs, _conf500_min_seen
    
    if model is None:
        return frame, [], 0
    
    # Perform detection (higher imgsz so small objects like coins stay visible)
    results = model(frame, verbose=False, conf=0.3, imgsz=1024)
    
    detections = []
    total_value = 0

    # Collect raw boxes first (before drawing) to apply the temporal filter
    raw = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Per-class confidence floor. Only the 500-note class is gated
            # (that is where the measured body false positives came from);
            # every other class keeps the original 0.3 behaviour so nothing
            # else regresses (e.g. the weakly-detected 200 note).
            if cls == 5:
                floor = _conf500_min_seen + 0.02 if _conf500_min_seen is not None else INITIAL_500_FLOOR
            else:
                floor = MIN_CONF_COINS

            # Hysteresis: a 500 detection just below the floor that appears
            # among a cluster of higher ones is treated as genuine and
            # lowers the learned floor (handles weak real-500 frames)
            if cls == 5 and floor - 0.1 <= conf < floor:
                near_high = sum(1 for c in _conf500_obs if c >= floor)
                if near_high >= 2:
                    _conf500_min_seen = min(_conf500_min_seen or conf, conf)
                    floor = _conf500_min_seen + 0.02

            if conf < floor:
                continue
            if cls == 5:
                _conf500_obs.append(conf)
            raw.append((cls, conf, (x1, y1, x2, y2)))
    _conf500_obs = [c for c in _conf500_obs][-30:]

    # Majority vote over the recent frame window (best box per class this frame)
    this_frame = {}
    for cls, conf, bbox in raw:
        if cls not in this_frame or conf > this_frame[cls][0]:
            this_frame[cls] = (conf, bbox)
    _frame_history.append(this_frame)
    for cls, val in this_frame.items():
        _last_box[cls] = val

    votes = Counter()
    for frame_map in _frame_history:
        for cls in frame_map:
            votes[cls] += 1

    # Prune classes no longer present anywhere in the voting window
    for c in list(_last_box.keys()):
        if c not in votes:
            _last_box.pop(c, None)

    # Plurality: report the class(es) that lead the vote. Classes seen in
    # fewer frames than the leader are dropped (kills 200/500/10 flips),
    # while genuinely multiple items in view tie at the top and both show.
    top_votes = max(votes.values(), default=0)
    for cls, n in votes.most_common():
        if n < MIN_VOTES or n < top_votes:
            continue
        conf, (x1, y1, x2, y2) = _last_box[cls]

        # Get currency name and value
        currency_name = CURRENCY_NAMES.get(cls, 'Unknown')
        currency_value = CURRENCY_VALUES.get(cls, 0)

        # Add to total value
        total_value += currency_value

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add label (denomination name only - no confidence shown)
        label = currency_name
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add to detections list
        detections.append({
            'currency': currency_name,
            'confidence': conf,
            'value': currency_value,
            'bbox': [x1, y1, x2, y2]
        })

    return frame, detections, total_value

    # Loads the main HTML page and provides the user interface for the currency recognition system.


@app.route('/')
def index():
    """Serve the main web page"""
    return render_template('index.html')


# Provides an API endpoint to load or reload the trained YOLOv8 model and returns the loading status.

@app.route('/debug_raw')
def debug_raw():
    """TEMP diagnostic: raw model output (no floors/voting) + saves frames."""
    global model
    if model is None:
        return jsonify({'error': 'model not loaded'})
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'error': 'camera busy - close the app tab and retry'})
    out = []
    for i in range(6):
        ok, frame = cap.read()
        if not ok:
            out.append({'frame': i, 'read_failed': True})
            continue
        cv2.imwrite(f'debug_frame_{i}.jpg', frame)
        res = model(frame, verbose=False, conf=0.15, imgsz=1024)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            dets.append({'cls': int(b.cls[0]),
                         'name': CURRENCY_NAMES.get(int(b.cls[0]), '?'),
                         'conf': round(float(b.conf[0]), 3),
                         'box': f'{x2-x1}x{y2-y1}'})
        out.append({'frame': i, 'shape': list(frame.shape), 'raw': dets})
    cap.release()
    return jsonify(out)

@app.route('/api/load-model', methods=['POST'])
def api_load_model():
    """API endpoint to load/reload the model"""
    if load_model():
        return jsonify({'status': 'success', 'message': 'Model loaded successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Model not found'}), 404
# The api_detect() function is responsible for receiving a single image from the frontend, processing that image, 
# sending it to the YOLOv8 currency detection model, and returning the detection result to the frontend.

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for single image detection"""
    global model
    
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    try:
        # Get image from request
        data = request.json
        image_data = data.get('image')
        
        # Decode base64 image
        header, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image'}), 400
        
        # Detect currency
        result_frame, detections, total_value = detect_currency(frame)
        
        if detections:
            return jsonify({
                'status': 'success',
                'detections': detections,
                'total_value': total_value,
                'message': f"Detected: {', '.join(d['currency'] for d in detections)}"
            })
        else:
            return jsonify({
                'status': 'success',
                'detections': [],
                'total_value': 0,
                'message': 'No currency detected'
            })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500



# Creates a live video endpoint that continuously sends processed camera frames to the web browser.
@app.route('/video_feed')
def video_feed():
    """Video streaming route for real-time detection"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Continuously captures frames from the webcam, sends them to the detection function, processes the results 
# and streams the output video to the browser.
def generate_frames():
    """Generate frames from camera for real-time detection"""
    global model
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Detect currency in frame
        if model is not None:
            frame, detections, total_value = detect_currency(frame)
            # Store latest detections and total value globally for API access
            global latest_detections, latest_total_value
            latest_detections = detections
            latest_total_value = total_value
        
        # Add status indicator
        status = "Model Loaded" if model is not None else "Model Not Loaded"
        color = (0, 255, 0) if model is not None else (0, 0, 255)
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

    
# Returns the latest detected denominations, number of detected currencies and their total value in JSON format.

@app.route('/api/detections')
def api_detections():
    """API endpoint to get latest detection results with total value"""
    global latest_detections, latest_total_value
    return jsonify({
        'status': 'success',
        'detections': latest_detections,
        'count': len(latest_detections),
        'total_value': latest_total_value
    })

# Checks and returns the current status of the YOLOv8 model and detection system.
@app.route('/api/status')
def api_status():
    """Get system status"""
    return jsonify({
        'model_loaded': model is not None,
        'detection_active': detection_active
    })


# Starts the Flask application, loads the trained YOLOv8 model and makes the system available through the local web server.

if __name__ == '__main__':
    print("="*60)
    print("Real-Time Indian Currency Recognition System")
    print("="*60)
    print("\nStarting server...")
    
    # Try to load model on startup
    load_model()
    
    print("\nServer running at: http://localhost:5000")
    print("Open browser and allow camera permissions")
    print("\nPress CTRL+C to quit")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
