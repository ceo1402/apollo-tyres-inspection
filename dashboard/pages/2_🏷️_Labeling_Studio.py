"""Labeling Studio page - Manual annotation of captured images."""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.database import Database
from src.storage import ImageStorage
from src.models import CaptureLabel, MarkLabel, DEFECT_TAGS, QUALITY_RATINGS, OVERALL_VERDICTS

st.set_page_config(page_title="Labeling Studio", page_icon="🏷️", layout="wide")

st.title("🏷️ Labeling Studio")

# Initialize resources
@st.cache_resource
def init_resources():
    config = load_config(str(project_root / "config.yaml"))
    db = Database(config.storage)
    storage = ImageStorage(config.storage, config.capture)
    return config, db, storage

config, db, storage = init_resources()

# Session state
if 'current_capture_idx' not in st.session_state:
    st.session_state.current_capture_idx = 0
if 'labeling_session_count' not in st.session_state:
    st.session_state.labeling_session_count = 0
if 'selected_mark' not in st.session_state:
    st.session_state.selected_mark = None

# Sidebar filters
st.sidebar.header("Filters")

show_labeled = st.sidebar.checkbox("Show Labeled", value=False)
show_unlabeled = st.sidebar.checkbox("Show Unlabeled", value=True)

date_filter = st.sidebar.date_input(
    "Date Range",
    value=[],
    help="Filter captures by date"
)

# Get captures based on filters
if show_labeled and show_unlabeled:
    captures = db.get_captures(limit=1000)
elif show_labeled:
    captures = db.get_captures(limit=1000, labeled_only=True)
elif show_unlabeled:
    captures = db.get_captures(limit=1000, unlabeled_only=True)
else:
    captures = []

# Apply date filter
if date_filter and len(date_filter) == 2:
    start_date = date_filter[0].isoformat()
    end_date = date_filter[1].isoformat() + "T23:59:59"
    captures = [c for c in captures if start_date <= c['timestamp'] <= end_date]

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.subheader("Session Stats")
st.sidebar.metric("Labeled this session", st.session_state.labeling_session_count)
st.sidebar.metric("Total labeled", db.get_capture_count(labeled_only=True))
st.sidebar.metric("Remaining unlabeled", len([c for c in captures if not c.get('is_labeled')]))

# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")

if captures:
    capture_options = [f"{c['capture_id'][-13:]} ({c['total_marks_detected']} marks)" 
                      for c in captures]
    selected_idx = st.sidebar.selectbox(
        "Select Capture",
        range(len(captures)),
        format_func=lambda i: capture_options[i],
        index=min(st.session_state.current_capture_idx, len(captures)-1)
    )
    st.session_state.current_capture_idx = selected_idx

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("◀ Prev"):
        st.session_state.current_capture_idx = max(0, st.session_state.current_capture_idx - 1)
        st.rerun()
with col2:
    if st.button("Skip"):
        st.session_state.current_capture_idx = min(len(captures)-1, st.session_state.current_capture_idx + 1)
        st.rerun()
with col3:
    if st.button("Next ▶"):
        st.session_state.current_capture_idx = min(len(captures)-1, st.session_state.current_capture_idx + 1)
        st.rerun()

# Main content
if not captures:
    st.info("No captures found matching the current filters.")
else:
    current_capture = captures[st.session_state.current_capture_idx]
    capture_details = db.get_capture(current_capture['capture_id'])
    
    if capture_details is None:
        st.error("Failed to load capture details")
    else:
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader(f"Capture: {capture_details['capture_id']}")
        with col2:
            st.write(f"📅 {capture_details['timestamp'][:19]}")
        with col3:
            if capture_details.get('is_labeled'):
                st.success("✅ Labeled")
            else:
                st.warning("⏳ Unlabeled")
        
        # Image and marks section
        img_col, marks_col = st.columns([2, 1])
        
        with img_col:
            # Display annotated image
            if capture_details.get('annotated_image_path') and Path(capture_details['annotated_image_path']).exists():
                st.image(capture_details['annotated_image_path'], 
                        caption="Annotated Image", use_container_width=True)
            elif capture_details.get('full_image_path') and Path(capture_details['full_image_path']).exists():
                st.image(capture_details['full_image_path'],
                        caption="Full Image", use_container_width=True)
            else:
                st.warning("Image not found")
            
            # Image tabs
            tab1, tab2, tab3 = st.tabs(["Annotated", "Full Frame", "Tyre Crop"])
            with tab1:
                if capture_details.get('annotated_image_path') and Path(capture_details['annotated_image_path']).exists():
                    st.image(capture_details['annotated_image_path'], use_container_width=True)
            with tab2:
                if capture_details.get('full_image_path') and Path(capture_details['full_image_path']).exists():
                    st.image(capture_details['full_image_path'], use_container_width=True)
            with tab3:
                if capture_details.get('tyre_crop_path') and Path(capture_details['tyre_crop_path']).exists():
                    st.image(capture_details['tyre_crop_path'], use_container_width=True)
        
        with marks_col:
            st.subheader("Detected Marks")
            
            marks = capture_details.get('marks', [])
            
            if not marks:
                st.info("No marks detected")
            else:
                for mark in marks:
                    with st.expander(f"Mark {mark['mark_index']}: {mark['auto_color']} {mark['auto_type']}", 
                                    expanded=st.session_state.selected_mark == mark['mark_index']):
                        
                        # Auto-detected values
                        st.write("**Auto-Detected:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Type: {mark['auto_type']}")
                            st.write(f"Color: {mark['auto_color']}")
                        with col2:
                            st.write(f"Circularity: {mark['circularity']:.3f}")
                            st.write(f"Diameter: {mark['estimated_diameter_mm']:.2f}mm")
                        
                        st.write(f"Solidity: {mark['solidity']:.3f}")
                        st.write(f"Position: ({mark['centroid_x']}, {mark['centroid_y']})")
                        
                        if mark.get('is_donut'):
                            st.write(f"Inner/Outer Ratio: {mark.get('inner_outer_ratio', 'N/A')}")
                        
                        # Mark labeling form
                        st.markdown("---")
                        st.write("**Manual Label:**")
                        
                        mark_key = f"mark_{capture_details['capture_id']}_{mark['mark_index']}"
                        
                        actual_type = st.selectbox(
                            "Actual Type",
                            options=['', 'solid', 'donut'],
                            index=0 if not mark.get('label_actual_type') else 
                                  (['', 'solid', 'donut'].index(mark.get('label_actual_type', ''))),
                            key=f"{mark_key}_type"
                        )
                        
                        actual_color = st.selectbox(
                            "Actual Color",
                            options=['', 'red', 'yellow'],
                            index=0 if not mark.get('label_actual_color') else
                                  (['', 'red', 'yellow'].index(mark.get('label_actual_color', ''))),
                            key=f"{mark_key}_color"
                        )
                        
                        quality = st.selectbox(
                            "Quality Rating",
                            options=[''] + QUALITY_RATINGS,
                            index=0 if not mark.get('label_quality_rating') else
                                  ([''] + QUALITY_RATINGS).index(mark.get('label_quality_rating', '')),
                            key=f"{mark_key}_quality"
                        )
                        
                        # Parse existing defects
                        existing_defects = []
                        if mark.get('label_defects'):
                            try:
                                existing_defects = json.loads(mark['label_defects'])
                            except:
                                existing_defects = []
                        
                        defects = st.multiselect(
                            "Defects",
                            options=DEFECT_TAGS,
                            default=existing_defects,
                            key=f"{mark_key}_defects"
                        )
                        
                        if st.button("Save Mark Label", key=f"{mark_key}_save"):
                            mark_label = MarkLabel(
                                label_actual_type=actual_type if actual_type else None,
                                label_actual_color=actual_color if actual_color else None,
                                label_quality_rating=quality if quality else None,
                                label_defects=defects,
                                label_matches_auto=(actual_type == mark['auto_type'] and 
                                                   actual_color == mark['auto_color']) if actual_type and actual_color else None
                            )
                            if db.update_mark_label(capture_details['capture_id'], mark['mark_index'], mark_label):
                                st.success("Mark label saved!")
                            else:
                                st.error("Failed to save mark label")
        
        # Overall capture labeling section
        st.markdown("---")
        st.subheader("Overall Capture Label")
        
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            overall_verdict = st.selectbox(
                "Overall Verdict",
                options=[''] + OVERALL_VERDICTS,
                index=0 if not capture_details.get('overall_verdict') else
                      ([''] + OVERALL_VERDICTS).index(capture_details.get('overall_verdict', '')),
                help="Overall quality assessment of this capture"
            )
            
            labeled_by = st.text_input(
                "Labeled By",
                value=capture_details.get('labeled_by', ''),
                placeholder="Your name"
            )
        
        with form_col2:
            label_notes = st.text_area(
                "Notes",
                value=capture_details.get('label_notes', ''),
                placeholder="Any additional notes about this capture...",
                height=100
            )
        
        # Auto-detected summary
        st.write("**Auto-Detected Summary:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Marks", capture_details['total_marks_detected'])
        with col2:
            st.metric("Red Solids", capture_details['red_solids'])
        with col3:
            st.metric("Red Donuts", capture_details['red_donuts'])
        with col4:
            st.metric("Yellow Solids", capture_details['yellow_solids'])
        with col5:
            st.metric("Yellow Donuts", capture_details['yellow_donuts'])
        
        # Save button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 Save & Next", type="primary", use_container_width=True):
                if not overall_verdict:
                    st.warning("Please select an overall verdict")
                elif not labeled_by:
                    st.warning("Please enter your name")
                else:
                    label = CaptureLabel(
                        is_labeled=True,
                        labeled_by=labeled_by,
                        labeled_at=datetime.now(),
                        overall_verdict=overall_verdict,
                        label_notes=label_notes
                    )
                    
                    if db.update_capture_label(capture_details['capture_id'], label):
                        st.success("Capture labeled successfully!")
                        st.session_state.labeling_session_count += 1
                        st.session_state.current_capture_idx = min(
                            len(captures)-1, 
                            st.session_state.current_capture_idx + 1
                        )
                        st.rerun()
                    else:
                        st.error("Failed to save label")
        
        with col2:
            if st.button("💾 Save Only", use_container_width=True):
                if not overall_verdict:
                    st.warning("Please select an overall verdict")
                elif not labeled_by:
                    st.warning("Please enter your name")
                else:
                    label = CaptureLabel(
                        is_labeled=True,
                        labeled_by=labeled_by,
                        labeled_at=datetime.now(),
                        overall_verdict=overall_verdict,
                        label_notes=label_notes
                    )
                    
                    if db.update_capture_label(capture_details['capture_id'], label):
                        st.success("Capture labeled successfully!")
                        st.session_state.labeling_session_count += 1
                        st.rerun()
                    else:
                        st.error("Failed to save label")
        
        with col3:
            st.write("")  # Spacer
