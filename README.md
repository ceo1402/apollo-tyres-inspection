# Apollo Tyres Chennai - Tyre Paint Mark Inspection System

A production-ready Python application for real-time computer vision inspection of tyre paint marks using a Logitech C922 Pro HD webcam.

## Features

- **Real-time Capture Engine**: State machine-based capture system preventing duplicate captures
- **Mark Detection**: Detects red/yellow solid dots and donut-shaped marks
- **Comprehensive Measurements**: Circularity, solidity, eccentricity, diameter, and more
- **SQLite Database**: Local storage for all inspection data
- **Streamlit Dashboard**: Real-time monitoring, labeling studio, analytics, and data export

## Hardware Requirements

- **Camera**: Logitech C922 Pro HD Stream Webcam (or compatible USB webcam)
- **Resolution**: 1920x1080 @ 30fps
- **Setup**: Camera mounted 1000-1200mm above conveyor belt

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system by editing `config.yaml` as needed.

## Quick Start

### Run Full System (Capture + Dashboard)
```bash
python run.py --mode full
```

### Run Dashboard Only
```bash
python run.py --mode dashboard
```

### Run with Mock Camera (Testing)
```bash
python run.py --mode full --mock-camera
```

### Access Dashboard
Open http://localhost:8501 in your browser.

## Calibration

### 1. Camera Test
```bash
python scripts/test_camera.py
```

### 2. Color & Scale Calibration
```bash
python scripts/calibrate.py
```

This will help you:
- Adjust HSV color ranges for red and yellow detection
- Calculate pixels-per-mm for accurate size measurements

## Dashboard Pages

### 1. Live Monitor (📹)
- Real-time camera feed with detection overlays
- State machine status indicator
- Capture progress toward 3000 target
- Start/Stop capture controls
- Manual capture trigger

### 2. Labeling Studio (🏷️)
- Browse and label captured images
- Per-mark quality ratings
- Defect tagging
- Overall verdict assignment

### 3. Analytics (📊)
- Threshold explorer with adjustable sliders
- Distribution charts (circularity, solidity, diameter)
- Confusion matrix for labeled data
- Sample gallery

### 4. Export (📤)
- Export to CSV or JSON
- Filter by date range and labels
- Column selection for exports

## Project Structure

```
tyre-mark-inspection/
├── run.py                      # Main entry point
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── camera.py               # Camera interface
│   ├── state_machine.py        # Conveyor state management
│   ├── tyre_detector.py        # Tyre presence detection
│   ├── mark_detector.py        # Color segmentation
│   ├── mark_classifier.py      # Solid vs donut classification
│   ├── measurement.py          # Shape metrics calculation
│   ├── database.py             # SQLite operations
│   ├── storage.py              # Image file management
│   ├── config.py               # Configuration loader
│   └── models.py               # Data classes
│
├── dashboard/
│   ├── app.py                  # Main Streamlit app
│   ├── pages/                  # Dashboard pages
│   └── components/             # Reusable UI components
│
├── scripts/
│   ├── calibrate.py            # Calibration utility
│   └── test_camera.py          # Camera test
│
└── data/                       # Created at runtime
    ├── inspection.db           # SQLite database
    ├── captures/               # Captured images
    ├── marks/                  # Individual mark images
    ├── baselines/              # Empty conveyor references
    └── exports/                # Exported data files
```

## Configuration

Key configuration options in `config.yaml`:

```yaml
camera:
  device_id: 0              # Camera device index
  pixels_per_mm: 1.2        # Calibrate on first run

detection:
  red_lower1: [0, 100, 100]     # HSV range for red
  red_upper1: [10, 255, 255]
  yellow_lower: [20, 100, 100]  # HSV range for yellow
  yellow_upper: [35, 255, 255]
  min_circularity_filter: 0.5   # Minimum circularity to detect

capture:
  stability_frames: 3           # Frames tyre must be stable
  min_capture_interval_ms: 1500 # Minimum time between captures
```

## Usage Tips

1. **First Run**: Ensure conveyor is empty when starting - the system will capture a baseline image.

2. **Lighting**: Consistent, well-lit environment improves detection accuracy.

3. **Calibration**: Re-calibrate HSV ranges if paint colors vary or lighting changes.

4. **Labeling**: Label at least 100-200 samples to establish meaningful thresholds in Analytics.

5. **Thresholds**: Use the Analytics page to find optimal circularity/solidity thresholds based on labeled data.

## Troubleshooting

### Camera not detected
- Check USB connection
- Try different `device_id` in config (0, 1, 2)
- Run `python scripts/test_camera.py` to diagnose

### Poor mark detection
- Run color calibration: `python scripts/calibrate.py`
- Adjust HSV ranges in config
- Check lighting conditions

### Duplicate captures
- Increase `min_capture_interval_ms`
- Increase `stability_frames`
- Verify tyre pauses at inspection station

## Raspberry Pi 5 Deployment

### Hardware Requirements
- Raspberry Pi 5 (8GB RAM recommended)
- 256GB SD card or larger
- Logitech C922 Pro HD webcam
- Pi OS 64-bit

### RPi5 Setup Instructions

1. **Update system:**
```bash
sudo apt update && sudo apt upgrade -y
```

2. **Install system dependencies:**
```bash
sudo apt install -y python3-pip python3-venv libatlas-base-dev \
    libhdf5-dev libopenblas-dev libjpeg-dev libtiff5-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev
```

3. **Clone the repository:**
```bash
cd ~
git clone https://github.com/ceo1402/apollo-tyres-inspection.git
cd apollo-tyres-inspection
```

4. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

5. **Install Python dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

6. **Test camera connection:**
```bash
python scripts/test_camera.py
```

7. **Run the system:**
```bash
# Full system (capture + dashboard)
python run.py --mode full

# Or dashboard only
python run.py --mode dashboard
```

8. **Access dashboard:**
Open `http://<rpi-ip-address>:8501` from any device on your network.

### Running as a Service (Auto-start on boot)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/apollo-inspection.service
```

Add:
```ini
[Unit]
Description=Apollo Tyres Inspection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/apollo-tyres-inspection
Environment=PATH=/home/pi/apollo-tyres-inspection/venv/bin
ExecStart=/home/pi/apollo-tyres-inspection/venv/bin/python run.py --mode full
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable apollo-inspection
sudo systemctl start apollo-inspection
```

Check status:
```bash
sudo systemctl status apollo-inspection
```

### Performance Tips for RPi5

1. **Use USB 3.0 port** for the webcam
2. **Reduce resolution** if needed (edit config.yaml):
   ```yaml
   camera:
     width: 1280
     height: 720
   ```
3. **Increase swap** if running low on memory:
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

## License

Proprietary - Apollo Tyres Chennai

## Support

For technical support, contact the development team.
