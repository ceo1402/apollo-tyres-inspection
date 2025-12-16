"""
Apollo Tyres Chennai - Tyre Paint Mark Inspection Dashboard
Main Streamlit application.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.database import Database

# Page config
st.set_page_config(
    page_title="Apollo Tyres - Paint Mark Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize config and database
@st.cache_resource
def init_resources():
    config = load_config(str(project_root / "config.yaml"))
    db = Database(config.storage)
    return config, db

config, db = init_resources()

# Main page content
st.title("🔍 Apollo Tyres Chennai")
st.subheader("Tyre Paint Mark Inspection System")

st.markdown("---")

# Show overall statistics
col1, col2, col3, col4 = st.columns(4)

stats = db.get_statistics()

with col1:
    st.metric(
        "Total Captures",
        stats['total_captures'],
        delta=f"{stats['total_captures']}/{config.project.target_samples} target"
    )

with col2:
    st.metric(
        "Labeled Captures",
        stats['labeled_captures'],
        delta=f"{stats['labeled_captures']}/{stats['total_captures']} total" if stats['total_captures'] > 0 else "0"
    )

with col3:
    st.metric(
        "Total Marks Detected",
        stats['total_marks']
    )

with col4:
    st.metric(
        "Avg Circularity",
        f"{stats['avg_circularity']:.3f}" if stats['avg_circularity'] else "N/A"
    )

st.markdown("---")

# Mark distribution
st.subheader("Mark Distribution")

if stats['marks_by_type']:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Red Solids", stats['marks_by_type'].get('red_solid', 0))
    with col2:
        st.metric("Red Donuts", stats['marks_by_type'].get('red_donut', 0))
    with col3:
        st.metric("Yellow Solids", stats['marks_by_type'].get('yellow_solid', 0))
    with col4:
        st.metric("Yellow Donuts", stats['marks_by_type'].get('yellow_donut', 0))
else:
    st.info("No marks detected yet")

st.markdown("---")

# Navigation
st.subheader("Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_📹_Live_Monitor.py", label="📹 Live Monitor", use_container_width=True)
    st.caption("Real-time camera feed and capture control")

with col2:
    st.page_link("pages/2_🏷️_Labeling_Studio.py", label="🏷️ Labeling Studio", use_container_width=True)
    st.caption("Manual annotation of captured images")

with col3:
    st.page_link("pages/3_📊_Analytics.py", label="📊 Analytics", use_container_width=True)
    st.caption("Threshold explorer and statistics")

with col4:
    st.page_link("pages/4_📤_Export.py", label="📤 Export", use_container_width=True)
    st.caption("Export data to CSV/JSON")

st.markdown("---")

# Recent captures preview
st.subheader("Recent Captures")

recent_captures = db.get_captures(limit=6)

if recent_captures:
    cols = st.columns(6)
    for i, capture in enumerate(recent_captures[:6]):
        with cols[i]:
            if capture.get('annotated_image_path') and Path(capture['annotated_image_path']).exists():
                st.image(capture['annotated_image_path'], caption=capture['capture_id'][-13:])
            else:
                st.info(capture['capture_id'][-13:])
            st.caption(f"{capture['total_marks_detected']} marks")
else:
    st.info("No captures yet. Start the capture system to begin inspection.")

# Footer
st.markdown("---")
st.caption("Apollo Tyres Chennai - Paint Mark Inspection POC | Version 1.0")
