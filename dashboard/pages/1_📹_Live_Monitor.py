"""Live Monitor page - Real-time camera feed and capture control."""

import streamlit as st
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.database import Database
from src.camera import Camera, MockCamera
from src.state_machine import ConveyorStateMachine, StateAction, ConveyorState
from src.tyre_detector import TyreDetector
from src.mark_detector import MarkDetector
from src.measurement import MeasurementCalculator, count_marks_by_type
from src.storage import ImageStorage
from src.models import CaptureResult

st.set_page_config(page_title="Live Monitor", page_icon="📹", layout="wide")

st.title("📹 Live Monitor & Capture")

# Initialize resources
@st.cache_resource
def init_resources():
    config = load_config(str(project_root / "config.yaml"))
    db = Database(config.storage)
    storage = ImageStorage(config.storage, config.capture)
    return config, db, storage

config, db, storage = init_resources()

# Session state initialization
if 'capture_running' not in st.session_state:
    st.session_state.capture_running = False
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'state_machine' not in st.session_state:
    st.session_state.state_machine = ConveyorStateMachine(config.capture)
if 'tyre_detector' not in st.session_state:
    st.session_state.tyre_detector = TyreDetector(config.capture)
if 'mark_detector' not in st.session_state:
    st.session_state.mark_detector = MarkDetector(config.detection)
if 'measurement_calc' not in st.session_state:
    st.session_state.measurement_calc = MeasurementCalculator(
        config.measurement, config.detection, config.camera.pixels_per_mm
    )
if 'session_captures' not in st.session_state:
    st.session_state.session_captures = 0

# Sidebar controls
st.sidebar.header("Capture Controls")

use_mock = st.sidebar.checkbox("Use Mock Camera", value=False, 
                               help="Use simulated camera for testing")

col1, col2 = st.sidebar.columns(2)

with col1:
    start_btn = st.button("▶️ Start", use_container_width=True)
with col2:
    stop_btn = st.button("⏹️ Stop", use_container_width=True)

manual_capture_btn = st.sidebar.button("📸 Manual Capture", use_container_width=True)
update_baseline_btn = st.sidebar.button("🔄 Update Baseline", use_container_width=True)

st.sidebar.markdown("---")

# Status display
st.sidebar.subheader("System Status")

total_captures = db.get_capture_count()
progress = total_captures / config.project.target_samples

st.sidebar.metric("Total Captures", f"{total_captures}/{config.project.target_samples}")
st.sidebar.progress(min(progress, 1.0))

st.sidebar.metric("Session Captures", st.session_state.session_captures)

if st.session_state.capture_running:
    st.sidebar.success("🟢 Capture Running")
else:
    st.sidebar.warning("🔴 Capture Stopped")

st.sidebar.markdown("---")

# Recent captures in sidebar
st.sidebar.subheader("Recent Captures")
recent = db.get_captures(limit=5)
for cap in recent:
    st.sidebar.caption(f"{cap['capture_id'][-13:]}: {cap['total_marks_detected']} marks")

# Main content
main_col, info_col = st.columns([2, 1])

with main_col:
    st.subheader("Camera Feed")
    frame_placeholder = st.empty()
    
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        state_indicator = st.empty()
    with status_col2:
        marks_indicator = st.empty()
    with status_col3:
        fps_indicator = st.empty()

with info_col:
    st.subheader("Detection Info")
    detection_info = st.empty()
    
    st.subheader("Current Marks")
    marks_display = st.empty()

# Handle button actions
if start_btn and not st.session_state.capture_running:
    try:
        if use_mock:
            st.session_state.camera = MockCamera(config.camera)
        else:
            st.session_state.camera = Camera(config.camera)
        
        if st.session_state.camera.connect():
            st.session_state.capture_running = True
            
            # Load or capture baseline
            baseline = storage.load_latest_baseline()
            if baseline is not None:
                st.session_state.tyre_detector.set_baseline(baseline)
            else:
                time.sleep(0.5)
                frame = st.session_state.camera.read()
                if frame is not None:
                    st.session_state.tyre_detector.set_baseline(frame)
                    storage.save_baseline(frame)
            
            st.session_state.state_machine.reset()
            st.success("Camera connected!")
        else:
            st.error("Failed to connect to camera")
    except Exception as e:
        st.error(f"Error starting capture: {e}")

if stop_btn and st.session_state.capture_running:
    st.session_state.capture_running = False
    if st.session_state.camera:
        st.session_state.camera.disconnect()
        st.session_state.camera = None
    st.info("Capture stopped")

if update_baseline_btn and st.session_state.camera:
    frame = st.session_state.camera.read()
    if frame is not None:
        st.session_state.tyre_detector.set_baseline(frame)
        storage.save_baseline(frame)
        st.success("Baseline updated!")

# Main capture loop
if st.session_state.capture_running and st.session_state.camera:
    force_capture = manual_capture_btn
    
    # Process frames
    frame_count = 0
    start_time = time.time()
    
    while st.session_state.capture_running:
        frame = st.session_state.camera.read()
        if frame is None:
            st.error("Lost camera connection")
            st.session_state.capture_running = False
            break
        
        # Detect tyre
        tyre_result = st.session_state.tyre_detector.detect(frame)
        
        # Handle forced capture
        if force_capture and tyre_result.is_present:
            st.session_state.state_machine.force_capture()
            force_capture = False
        
        # Update state machine
        action = st.session_state.state_machine.update(tyre_result)
        
        # Detect marks
        marks = []
        if tyre_result.is_present:
            tyre_mask = st.session_state.tyre_detector.get_tyre_mask(frame, tyre_result)
            detected_blobs = st.session_state.mark_detector.detect(frame, tyre_mask)
            marks = [
                st.session_state.measurement_calc.measure(frame, blob, i)
                for i, blob in enumerate(detected_blobs)
            ]
        
        # Handle capture action
        if action == StateAction.CAPTURE:
            # Process capture
            tyre_crop = st.session_state.tyre_detector.crop_tyre(frame, tyre_result)
            counts = count_marks_by_type(marks)
            
            sequence_number = db.get_next_sequence_number()
            capture_id = storage.generate_capture_id(sequence_number)
            
            annotated = storage.create_annotated_frame(
                frame, tyre_result, marks, "CAPTURED"
            )
            
            full_path, crop_path, annotated_path = storage.save_capture_images(
                capture_id, frame, tyre_crop, annotated
            )
            
            storage.save_mark_images(capture_id, frame, marks)
            
            capture = CaptureResult(
                capture_id=capture_id,
                timestamp=datetime.now(),
                sequence_number=sequence_number,
                camera_height_mm=config.camera.mounting_height_mm,
                frame_brightness_avg=float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))),
                tyre_detection_confidence=tyre_result.confidence,
                processing_duration_ms=0,
                total_marks_detected=counts['total'],
                red_solids=counts['red_solids'],
                red_donuts=counts['red_donuts'],
                yellow_solids=counts['yellow_solids'],
                yellow_donuts=counts['yellow_donuts'],
                is_empty_tyre=counts['total'] == 0,
                full_image_path=full_path,
                tyre_crop_path=crop_path,
                annotated_image_path=annotated_path,
                marks=marks
            )
            
            db.save_capture(capture)
            st.session_state.session_captures += 1
            st.toast(f"Captured: {capture_id}")
        
        elif action == StateAction.UPDATE_BASELINE:
            st.session_state.tyre_detector.set_baseline(frame)
            storage.save_baseline(frame)
        
        # Create annotated frame for display
        annotated = storage.create_annotated_frame(
            frame, tyre_result, marks, st.session_state.state_machine.state_name
        )
        
        # Convert BGR to RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Update display
        frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
        
        # Update status indicators
        state_indicator.metric("State", st.session_state.state_machine.state_name)
        marks_indicator.metric("Marks in View", len(marks))
        
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps_indicator.metric("FPS", f"{frame_count / elapsed:.1f}")
        
        # Update detection info
        with detection_info.container():
            if tyre_result.is_present:
                st.success("Tyre Detected")
                st.write(f"Confidence: {tyre_result.confidence:.2f}")
                st.write(f"Fully Visible: {'✅' if tyre_result.is_fully_visible else '❌'}")
                st.write(f"Stable: {'✅' if tyre_result.is_stable else '❌'}")
            else:
                st.info("No tyre in view")
        
        # Update marks display
        with marks_display.container():
            if marks:
                for mark in marks:
                    color_emoji = "🔴" if mark.auto_color == 'red' else "🟡"
                    type_emoji = "⭕" if mark.is_donut else "⚫"
                    st.write(f"{color_emoji} {type_emoji} Mark {mark.mark_index}: "
                            f"C={mark.circularity:.2f}, D={mark.estimated_diameter_mm:.1f}mm")
            else:
                st.info("No marks detected")
        
        # Small delay
        time.sleep(0.033)
        
        # Check if should stop (using a container for the button check)
        # Note: In Streamlit, we need to rerun to check button state
        # For continuous operation, consider using st.experimental_rerun() sparingly
        
else:
    # Show placeholder when not running
    frame_placeholder.info("Click 'Start' to begin capture")
    
    # Show last captured frame if available
    recent = db.get_captures(limit=1)
    if recent and recent[0].get('annotated_image_path'):
        path = recent[0]['annotated_image_path']
        if Path(path).exists():
            st.image(path, caption="Last capture")
