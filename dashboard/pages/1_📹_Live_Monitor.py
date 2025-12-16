"""Live Monitor page - Real-time camera feed and capture control.

This page displays the live feed from the main capture system.
The camera is managed by the main run.py process which writes frames to a shared file.
"""

import streamlit as st
import sys
import time
import cv2
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.database import Database
from src.storage import ImageStorage

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

# Path to live frame written by main capture process
live_frame_path = Path(config.storage.captures_path).parent / "live_frame.jpg"

# Sidebar controls
st.sidebar.header("System Status")

total_captures = db.get_capture_count()
progress = total_captures / config.project.target_samples

st.sidebar.metric("Total Captures", f"{total_captures}/{config.project.target_samples}")
st.sidebar.progress(min(progress, 1.0))

# Check if capture system is running by checking if live frame is recent
system_running = False
if live_frame_path.exists():
    frame_age = time.time() - live_frame_path.stat().st_mtime
    system_running = frame_age < 5  # Frame updated in last 5 seconds

if system_running:
    st.sidebar.success("🟢 Capture System Running")
else:
    st.sidebar.warning("🔴 Capture System Stopped")
    st.sidebar.info("Start with: sudo systemctl start apollo-inspection")

st.sidebar.markdown("---")

# Recent captures in sidebar
st.sidebar.subheader("Recent Captures")
recent = db.get_captures(limit=5)
for cap in recent:
    st.sidebar.caption(f"{cap['capture_id'][-13:]}: {cap['total_marks_detected']} marks")

# Main content
main_col, info_col = st.columns([2, 1])

with main_col:
    st.subheader("Live Camera Feed")
    
    # Live feed placeholder
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
    with col2:
        refresh_rate = st.slider("Refresh rate (sec)", 0.3, 3.0, 0.5, 0.1)
    
    # Display live frame from capture system
    if live_frame_path.exists():
        try:
            # Check how old the frame is
            frame_age = time.time() - live_frame_path.stat().st_mtime
            
            if frame_age < 10:  # Frame is recent
                # Read and display the frame
                frame_placeholder.image(str(live_frame_path), 
                                       caption=f"Live Feed - Updated {frame_age:.1f}s ago",
                                       use_container_width=True)
                status_placeholder.success(f"✅ Live feed active | Frame age: {frame_age:.1f}s | Captures: {total_captures}")
            else:
                frame_placeholder.image(str(live_frame_path), 
                                       caption=f"Last frame ({frame_age:.0f}s ago - system may be stopped)",
                                       use_container_width=True)
                status_placeholder.warning(f"⚠️ Frame is {frame_age:.0f}s old - capture system may be stopped")
        except Exception as e:
            frame_placeholder.error(f"Error loading frame: {e}")
    else:
        frame_placeholder.info("📷 Waiting for capture system to start...")
        status_placeholder.info("The capture system will write live frames when running.")
        
        # Show baseline as placeholder
        baseline_dir = Path(config.storage.baselines_path)
        if baseline_dir.exists():
            baselines = sorted(baseline_dir.glob("baseline_*.jpg"), reverse=True)
            if baselines:
                st.image(str(baselines[0]), caption="Last baseline (camera view when empty)")
    
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

with info_col:
    st.subheader("Capture Statistics")
    
    stats = db.get_statistics()
    
    st.metric("Total Captures", stats['total_captures'])
    st.metric("Total Marks", stats['total_marks'])
    st.metric("Labeled", stats['labeled_captures'])
    st.metric("Empty Tyres", stats['empty_tyres'])
    
    if stats['avg_circularity']:
        st.metric("Avg Circularity", f"{stats['avg_circularity']:.3f}")
    
    st.markdown("---")
    
    st.subheader("Mark Distribution")
    if stats['marks_by_type']:
        for mark_type, count in stats['marks_by_type'].items():
            st.write(f"**{mark_type}**: {count}")
    else:
        st.info("No marks detected yet")

st.markdown("---")

# Recent captures gallery
st.subheader("Recent Captures Gallery")

recent_captures = db.get_captures(limit=12)

if recent_captures:
    cols = st.columns(4)
    for i, capture in enumerate(recent_captures):
        with cols[i % 4]:
            if capture.get('annotated_image_path') and Path(capture['annotated_image_path']).exists():
                st.image(capture['annotated_image_path'], caption=capture['capture_id'][-13:], use_container_width=True)
            else:
                st.info(capture['capture_id'][-13:])
            st.caption(f"{capture['total_marks_detected']} marks | {capture['timestamp'][:19]}")
else:
    st.info("No captures yet. Position a tyre in front of the camera to begin.")

# Baseline info
st.markdown("---")
st.subheader("System Info")

baseline_dir = Path(config.storage.baselines_path)
if baseline_dir.exists():
    baselines = sorted(baseline_dir.glob("baseline_*.jpg"), reverse=True)
    if baselines:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Latest Baseline**: {baselines[0].name}")
            st.image(str(baselines[0]), caption="Current baseline (empty conveyor)", use_container_width=True)
        with col2:
            st.write(f"**Total Baselines**: {len(baselines)}")
            st.write(f"**Data Directory**: {config.storage.database_path}")
            st.write(f"**Target Samples**: {config.project.target_samples}")
