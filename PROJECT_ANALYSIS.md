# Real-Time Indian Currency Recognition System
## Complete Project Analysis and Technical Documentation

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Technical Stack](#technical-stack)
5. [System Components](#system-components)
6. [Dataset and Training](#dataset-and-training)
7. [Model Performance](#model-performance)
8. [Implementation Details](#implementation-details)
9. [Current Status and Limitations](#current-status-and-limitations)
10. [Future Enhancements](#future-enhancements)
11. [Team Information](#team-information)

---

## Project Overview

**Project Name:** Real-Time Indian Currency Recognition System  
**Type:** Web-based AI Application  
**Purpose:** Assist visually impaired individuals in identifying Indian currency denominations  
**Development Period:** Academic Project - BCE, Shravanabelagola  

### Key Objectives
- Develop real-time currency detection using computer vision
- Provide voice assistance for accessibility
- Create web-based interface for ease of use
- Ensure privacy through local processing
- Achieve high detection accuracy

---

## Problem Statement

Visually impaired individuals face significant challenges in:
- Identifying currency denominations independently
- Distinguishing between similar-looking notes
- Verifying currency authenticity
- Managing financial transactions without assistance

**Impact:** Loss of independence, reliance on others, vulnerability to fraud

---

## Solution Architecture

### High-Level Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │───▶│   OpenCV    │───▶│  YOLO Model │───▶│ Detection   │
│   Input     │    │  Processing │    │  Inference  │    │   Results   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Voice     │◀───│   Flask     │◀───│  Class      │◀───│   JSON API  │
│   Output    │    │   Backend   │    │  Mapping    │    │   Endpoint  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────┐
│   Web       │
│  Interface  │
└─────────────┘
```

### Data Flow
1. **Capture**: Camera captures video frames at 30 FPS
2. **Process**: OpenCV preprocesses frames for YOLO input
3. **Detect**: YOLOv8 model performs object detection
4. **Map**: Class IDs mapped to currency denominations
5. **Serve**: Flask API serves results via REST endpoints
6. **Display**: Web interface shows detections with bounding boxes
7. **Announce**: Web Speech API provides voice feedback

---

## Technical Stack

### Frontend Technologies
- **HTML5**: Structure and semantic markup
- **CSS3**: Styling with gradients and responsive design
- **JavaScript**: Client-side logic and API communication
- **Web Speech API**: Text-to-speech for voice output
- **Fetch API**: Asynchronous HTTP requests

### Backend Technologies
- **Python 3.8+**: Core programming language
- **Flask 3.0.0**: Web framework and API server
- **Flask-CORS 4.0.0**: Cross-origin resource sharing

### AI/ML Technologies
- **Ultralytics YOLOv8 8.0.196**: Object detection model
- **PyTorch 2.1.0**: Deep learning framework
- **TorchVision 0.16.0**: Computer vision utilities
- **OpenCV 4.9.0.80**: Image processing
- **NumPy 1.24.3**: Numerical computations

### Dependencies
```
ultralytics==8.0.196
opencv-python==4.9.0.80
flask==3.0.0
flask-cors==4.0.0
numpy==1.24.3
torch==2.1.0
torchvision==0.16.0
```

---

## System Components

### 1. Backend Application (app.py)

**Key Functions:**
- `load_model()`: Loads trained YOLO model from disk
- `detect_currency()`: Performs detection on video frames
- `generate_frames()`: Streams video with detection overlays
- API endpoints for status, detections, and model loading

**API Endpoints:**
- `GET /`: Serves main web interface
- `GET /video_feed`: Real-time video stream with detections
- `GET /api/status`: System status and model loading state
- `GET /api/detections`: Latest detection results
- `POST /api/load-model`: Manual model reload

**Currency Mappings:**
```python
CURRENCY_NAMES = {
    3: '10 Rupees Note', 4: '20 Rupees Note', 5: '50 Rupees Note',
    6: '100 Rupees Note', 7: '200 Rupees Note', 8: '500 Rupees Note',
    9: '2000 Rupees Note', 10: '1 Rupee Coin', 11: '2 Rupees Coin',
    12: '5 Rupees Coin', 13: '10 Rupees Coin', 14: '20 Rupee Coin'
}
```

### 2. Training Script (train_model.py)

**Training Configuration:**
- Base Model: YOLOv8n (nano version)
- Epochs: 50
- Image Size: 640x640 pixels
- Batch Size: 16
- Optimizer: SGD
- Learning Rate: 0.01 (with warmup)
- Device: CPU (configurable for GPU)

**Training Parameters:**
```python
results = model.train(
    data='dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='cpu',
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    patience=50
)
```

### 3. Web Interface (templates/index.html)

**Features:**
- Auto-loading model on startup
- Auto-activating camera
- Real-time video feed display
- Detection information overlay
- Voice announcements with cooldown
- Responsive design for mobile/desktop

**JavaScript Functions:**
- `startCamera()`: Activates camera feed
- `fetchDetections()`: Polls API for detection results
- `processDetections()`: Handles detection data
- `speak()`: Web Speech API for voice output
- `updateStatus()`: Updates UI status indicators

---

## Dataset and Training

### Dataset Structure
```
combined_dataset/
├── images/
│   ├── train/ (empty - images not present)
│   └── val/ (empty - images not present)
└── labels/
    ├── train/ (3,571 label files)
    └── val/ (2,112 label files)
```

### Dataset Configuration (dataset.yaml)
```yaml
names: [1_rupee_note, 2_rupee_note, 5_rupee_note, 10_rupee_note, 
        20_rupee_note, 50_rupee_note, 100_rupee_note, 200_rupee_note,
        500_rupee_note, 2000_rupee_note, 1_rupee_coin, 2_rupee_coin,
        5_rupee_coin, 10_rupee_coin, 20_rupee_coin]
nc: 15
path: ./dataset
train: images/train
val: images/val
```

### Actual Training Data Distribution
**Classes Present in Training:**
- Class 3: 649 samples (10₹ Note)
- Class 4: 543 samples (20₹ Note)
- Class 5: 498 samples (50₹ Note)
- Class 6: 331 samples (100₹ Note)
- Class 7: 271 samples (200₹ Note)
- Class 8: 279 samples (500₹ Note)
- Class 9: 102 samples (2000₹ Note)
- Class 10: 208 samples (1₹ Coin)
- Class 11: 120 samples (2₹ Coin)
- Class 12: 270 samples (5₹ Coin)
- Class 13: 180 samples (10₹ Coin)
- Class 14: 120 samples (20₹ Coin)

**Missing Classes:**
- Class 0: 1₹ Note (0 samples)
- Class 1: 2₹ Note (0 samples)
- Class 2: 5₹ Note (0 samples)

**Total Training Samples:** 3,571  
**Total Validation Samples:** 2,112

---

## Model Performance

### Training Results (50 Epochs)

**Final Metrics:**
- **Precision:** 99.92%
- **Recall:** 100%
- **mAP50:** 99.5%
- **mAP50-95:** 99.48%

**Training Progress:**
- Epoch 1: mAP50 = 18.7%
- Epoch 10: mAP50 = 93.9%
- Epoch 25: mAP50 = 99.5%
- Epoch 50: mAP50 = 99.5%

**Loss Reduction:**
- Box Loss: 1.80 → 0.36
- Class Loss: 3.70 → 0.26
- DFL Loss: 1.93 → 1.00

### Model Specifications
- **Architecture:** YOLOv8n (Nano)
- **Input Size:** 640x640 pixels
- **Parameters:** ~3.2M (lightweight)
- **Inference Speed:** ~30 FPS on CPU
- **Model Size:** ~6.5 MB

---

## Implementation Details

### Detection Pipeline

**Step 1: Frame Capture**
```python
cap = cv2.VideoCapture(0)
success, frame = cap.read()
```

**Step 2: Model Inference**
```python
results = model(frame, verbose=False, conf=0.3)
```

**Step 3: Detection Processing**
```python
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        currency_name = CURRENCY_NAMES.get(cls, 'Unknown')
```

**Step 4: Visualization**
```python
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(frame, label, (x1, y1 - 10), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

**Step 5: Frame Streaming**
```python
ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + 
       frame_bytes + b'\r\n')
```

### Voice Feedback System

**Web Speech API Implementation:**
```javascript
function speak(text) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-IN';
    utterance.rate = 1;
    utterance.volume = 1;
    window.speechSynthesis.speak(utterance);
}
```

**Cooldown Mechanism:**
- 3-second cooldown between announcements
- Prevents repetitive voice output
- Triggers on denomination change or timeout

---

## Current Status and Limitations

### Working Features
✅ Real-time currency detection (7 note denominations)  
✅ Voice announcements for detected currencies  
✅ Web-based interface with live video feed  
✅ High detection accuracy (99.5% mAP50)  
✅ Real-time performance (30 FPS)  
✅ Privacy-focused (local processing)  
✅ Responsive design  

### Known Issues
⚠️ **Class Mapping Inconsistencies**: Dataset labels may not match actual denominations  
⚠️ **Coin Detection**: Poor detection of coins due to small size  
⚠️ **Missing Classes**: 1₹, 2₹, 5₹ notes not in training data  
⚠️ **Dataset Images**: Original training images not accessible for verification  
⚠️ **Confidence Threshold**: Lowered to 0.3 to improve detection (may increase false positives)  

### Limitations
- Requires good lighting conditions
- Currency must be clearly visible to camera
- Works best with flat, unfolded notes
- Coin detection needs improvement
- No offline mobile deployment
- Browser-dependent (requires Web Speech API support)

---

## Future Enhancements

### Short-term Improvements
1. **Fix Class Mappings**: Verify and correct denomination mappings
2. **Improve Coin Detection**: Add specialized coin training data
3. **Add Missing Classes**: Include 1₹, 2₹, 5₹ notes in dataset
4. **Optimize Thresholds**: Fine-tune confidence thresholds
5. **Add Edge Cases**: Handle folded, damaged, or partially visible notes

### Long-term Goals
1. **Mobile App**: Native Android/iOS application
2. **Multi-currency Support**: Extend to other currencies
3. **Offline Deployment**: Mobile-optimized offline model
4. **Authentication**: Currency verification features
5. **Batch Processing**: Support for multiple currency detection
6. **Enhanced Accessibility**: Additional accessibility features

### Technical Improvements
- GPU acceleration for faster inference
- Model quantization for mobile deployment
- Data augmentation for robustness
- Ensemble methods for improved accuracy
- Real-time video processing optimization

---

## Team Information

**Project Team:**
- Aditya Patil (4BB23CS001)
- Apparay Janoji (4BB23CS005)
- Paris M Shirahatti (4BB23CS029)
- Spoorti K Kalagonda Korigeri (4BB23CS047)

**Department:** Computer Science and Engineering  
**Institution:** BCE, Shravanabelagola  
**Project Type:** Academic Curriculum Project  

---

## Conclusion

The Real-Time Indian Currency Recognition System successfully demonstrates the application of modern computer vision and deep learning techniques to solve a real-world accessibility problem. Despite some limitations in the current implementation, the system achieves high detection accuracy (99.5% mAP50) and provides real-time voice assistance for visually impaired individuals.

The project showcases:
- Successful integration of YOLOv8 with web technologies
- Practical application of AI for social good
- End-to-end system development from data collection to deployment
- Accessibility-focused design principles

With the identified improvements and future enhancements, this system has the potential to become a valuable tool for visually impaired individuals in India and could be extended to support other currencies and use cases.

---

## Appendix

### File Structure
```
major_project/
├── app.py                      # Flask backend (259 lines)
├── train_model.py              # Training script (119 lines)
├── requirements.txt            # Dependencies (19 lines)
├── dataset.yaml                # Dataset configuration
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── yolov8n.pt                  # Pre-trained YOLO model
├── templates/
│   └── index.html              # Web interface (526 lines)
├── combined_dataset/
│   ├── dataset.yaml
│   ├── labels/
│   │   ├── train/ (3,571 files)
│   │   └── val/ (2,112 files)
│   └── images/ (empty)
├── dataset/ (empty)
└── runs/
    └── detect/
        └── runs/
            └── detect/
                └── currency_detection/
                    ├── weights/
                    │   └── best.pt (trained model)
                    ├── results.csv (training metrics)
                    ├── confusion_matrix.png
                    └── results.png
```

### Key Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py --epochs 50 --img_size 640 --batch_size 16

# Run application
python app.py

# Access web interface
http://localhost:5000
```

### Performance Metrics Summary
| Metric | Value |
|--------|-------|
| Precision | 99.92% |
| Recall | 100% |
| mAP50 | 99.5% |
| mAP50-95 | 99.48% |
| Inference Speed | ~30 FPS |
| Model Size | ~6.5 MB |
| Training Epochs | 50 |
| Training Samples | 3,571 |
| Validation Samples | 2,112 |

---

*Document Version: 1.0*  
*Last Updated: September 6, 2026*  
*Generated for Team Presentation*
