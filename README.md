# Real-Time Indian Currency Recognition System

A web-based application that detects Indian currency notes using YOLO and provides voice assistance for visually impaired users.

## 🎯 Features

- ✅ Real-time currency detection using camera
- ✅ Voice output for detected denominations
- ✅ Supports all Indian currency notes (₹1, ₹2, ₹5, ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000)
- ✅ Web-based interface - no installation required for users
- ✅ Accessible design for visually impaired individuals
- ✅ No image upload option - only real-time camera detection

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- Modern web browser (Chrome, Firefox, Edge)

## 🚀 Installation

### Step 1: Clone/Download the Project
```bash
cd d:\major_project
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

## 📸 Dataset Preparation

### Collect Currency Images
1. Gather images of Indian currency notes (all denominations)
2. Place them in a folder named `currency_images`

### Prepare Dataset Structure
```powershell
python prepare_dataset.py --input_path ./currency_images --output_path ./dataset
```

### Label Your Images
Use labeling tools to annotate your images:
- **LabelImg**: https://github.com/tzutalin/labelImg
- **CVAT**: https://cvat.ai/
- **Roboflow Annotate**: https://roboflow.com/

Save labels in YOLO format (.txt files) with the following class IDs:
- 0: 1_rupee
- 1: 2_rupee
- 2: 5_rupee
- 3: 10_rupee
- 4: 20_rupee
- 5: 50_rupee
- 6: 100_rupee
- 7: 200_rupee
- 8: 500_rupee
- 9: 2000_rupee

## 🤖 Training the Model

### Train YOLO Model
```powershell
python train_model.py --epochs 100 --img_size 640 --batch_size 16
```

### Custom Training Options
```powershell
# For faster training with fewer epochs
python train_model.py --epochs 50

# For higher resolution detection
python train_model.py --img_size 1280

# For larger batches (if you have GPU)
python train_model.py --batch_size 32
```

The trained model will be saved at: `runs/detect/currency_detection/weights/best.pt`

## 🌐 Running the Application

### Start the Flask Server
```powershell
python app.py
```

### Access the Web Application
1. Open your browser
2. Navigate to: `http://localhost:5000`
3. Click "Load Model" to load the trained currency detection model
4. Click "Start Camera" to activate your camera
5. Allow camera permissions when prompted
6. Show currency notes to the camera
7. Listen for voice announcements!

## 🎮 How to Use

1. **Load Model**: Click the "Load Model" button to load the trained YOLO model
2. **Start Camera**: Click "Start Camera" to activate your webcam
3. **Show Currency**: Hold Indian currency notes in front of the camera
4. **Listen**: The system will announce the detected denomination aloud
5. **Stop Camera**: Click "Stop Camera" when done

### Voice Feedback
- The system announces detected currency every 3 seconds
- Voice output includes the denomination (e.g., "Detected 100 Rupees note")
- Use the "Test Voice" button to verify audio is working

## 📁 Project Structure

```
major_project/
│
├── app.py                      # Flask backend application
├── train_model.py              # Model training script
├── prepare_dataset.py          # Dataset preparation utility
├── requirements.txt            # Python dependencies
├── dataset.yaml                # Dataset configuration (auto-generated)
│
├── templates/
│   └── index.html              # Web interface
│
├── dataset/                    # Dataset directory (created after preparation)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
└── runs/
    └── detect/
        └── currency_detection/
            └── weights/
                └── best.pt     # Trained model
```

## ⚙️ Configuration

### Detection Settings
In `app.py`, you can modify:
- `DETECTION_COOLDOWN`: Time between voice announcements (default: 3 seconds)
- `conf`: Confidence threshold for detection (default: 0.5)

### Voice Settings
In `app.py`, you can adjust:
- `rate`: Speech speed (default: 150)
- `volume`: Voice volume (default: 0.9)

## 🔧 Troubleshooting

### Camera Not Working
- Ensure camera permissions are granted in your browser
- Check if another application is using the camera
- Try using Chrome or Firefox for best compatibility

### Model Not Loading
- Verify the model exists at `runs/detect/currency_detection/weights/best.pt`
- Re-train the model if necessary: `python train_model.py`

### No Voice Output
- Check your system volume
- Ensure speakers/headphones are connected
- Test voice using the "Test Voice" button
- On Windows, pyttsx3 should work automatically

### Poor Detection Accuracy
- Increase training epochs
- Add more diverse currency images to dataset
- Ensure proper lighting when capturing images
- Make sure currency notes are clearly visible

## 🎓 Technical Details

### Model Architecture
- **Base Model**: YOLOv8n (nano version for fast inference)
- **Input Size**: 640x640 pixels
- **Classes**: 10 (one for each denomination)
- **Training**: Transfer learning from COCO pretrained weights

### Detection Pipeline
1. Camera captures video frames (30 FPS)
2. Each frame is processed through YOLO model
3. Detections above confidence threshold (0.5) are accepted
4. Voice announcement triggered for new detections
5. 3-second cooldown prevents repetitive announcements

## 📝 Notes

- This system is designed specifically for Indian currency notes
- Works best with good lighting conditions
- Keep currency notes flat and clearly visible to the camera
- The system works offline after initial setup
- No images are uploaded or stored - all processing is local

## 👥 Credits

Developed by:
- Aditya Patil (4BB23CS001)
- Apparay Janoji (4BB23CS005)
- Paris M Shirahatti (4BB23CS029)
- Spoorti K Kalagonda Korigeri (4BB23CS047)

Department of Computer Science and Engineering
BCE, Shravanabelagola

## 📄 License

This project is created for educational purposes as part of academic curriculum.
