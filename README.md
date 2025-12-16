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

## License

Proprietary - Apollo Tyres Chennai

## Support

For technical support, contact the development team.
