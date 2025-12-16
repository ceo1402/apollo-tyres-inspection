"""Export page - Export data to CSV/JSON."""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.database import Database
from src.storage import ImageStorage

st.set_page_config(page_title="Export", page_icon="📤", layout="wide")

st.title("📤 Data Export")

# Initialize resources
@st.cache_resource
def init_resources():
    config = load_config(str(project_root / "config.yaml"))
    db = Database(config.storage)
    storage = ImageStorage(config.storage, config.capture)
    return config, db, storage

config, db, storage = init_resources()

# Export options
st.subheader("Export Options")

col1, col2 = st.columns(2)

with col1:
    export_type = st.selectbox(
        "Export Type",
        options=[
            "All Captures",
            "Labeled Captures Only",
            "All Marks",
            "Full Dataset (JSON)"
        ]
    )
    
    labeled_only = st.checkbox(
        "Include Only Labeled Data",
        value=False,
        disabled=export_type == "Labeled Captures Only"
    )

with col2:
    date_range = st.date_input(
        "Date Range (Optional)",
        value=[],
        help="Filter exports by date range"
    )
    
    file_format = st.selectbox(
        "File Format",
        options=["CSV", "JSON"],
        disabled=export_type == "Full Dataset (JSON)"
    )

# Column selection for CSV exports
if export_type in ["All Captures", "Labeled Captures Only"] and file_format == "CSV":
    st.subheader("Select Columns")
    
    capture_columns = [
        'capture_id', 'timestamp', 'sequence_number',
        'camera_height_mm', 'frame_brightness_avg',
        'tyre_detection_confidence', 'processing_duration_ms',
        'total_marks_detected', 'red_solids', 'red_donuts',
        'yellow_solids', 'yellow_donuts', 'is_empty_tyre',
        'full_image_path', 'tyre_crop_path', 'annotated_image_path',
        'is_labeled', 'labeled_by', 'labeled_at',
        'overall_verdict', 'label_notes'
    ]
    
    selected_columns = st.multiselect(
        "Columns to Export",
        options=capture_columns,
        default=['capture_id', 'timestamp', 'total_marks_detected',
                'red_solids', 'red_donuts', 'yellow_solids', 'yellow_donuts',
                'is_labeled', 'overall_verdict']
    )

elif export_type == "All Marks" and file_format == "CSV":
    st.subheader("Select Columns")
    
    mark_columns = [
        'capture_id', 'mark_index', 'timestamp',
        'auto_type', 'auto_color',
        'centroid_x', 'centroid_y',
        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
        'area_pixels', 'perimeter_pixels', 'estimated_diameter_mm',
        'circularity', 'solidity', 'eccentricity', 'convexity', 'edge_roughness',
        'color_h_mean', 'color_s_mean', 'color_v_mean',
        'color_h_std', 'color_s_std', 'color_v_std',
        'is_donut', 'inner_outer_ratio',
        'label_actual_type', 'label_actual_color',
        'label_quality_rating', 'label_defects'
    ]
    
    selected_columns = st.multiselect(
        "Columns to Export",
        options=mark_columns,
        default=['capture_id', 'mark_index', 'auto_type', 'auto_color',
                'circularity', 'solidity', 'estimated_diameter_mm',
                'label_quality_rating']
    )

# Preview and export
st.markdown("---")

# Get data for preview
start_date = date_range[0].isoformat() if len(date_range) >= 1 else None
end_date = date_range[1].isoformat() + "T23:59:59" if len(date_range) >= 2 else None

if export_type == "All Captures":
    data = db.get_captures(limit=100000, labeled_only=labeled_only,
                          start_date=start_date, end_date=end_date)
elif export_type == "Labeled Captures Only":
    data = db.get_captures(limit=100000, labeled_only=True,
                          start_date=start_date, end_date=end_date)
elif export_type == "All Marks":
    data = db.get_marks_for_analytics(labeled_only=labeled_only,
                                      start_date=start_date, end_date=end_date)
else:
    data = None

if data:
    df = pd.DataFrame(data)
    
    # Apply column selection
    if export_type in ["All Captures", "Labeled Captures Only", "All Marks"] and file_format == "CSV":
        available_columns = [c for c in selected_columns if c in df.columns]
        df = df[available_columns]
    
    st.subheader("Data Preview")
    st.write(f"Total records: {len(df)}")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Export button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        if file_format == "CSV" and export_type != "Full Dataset (JSON)":
            # CSV export
            csv_data = df.to_csv(index=False)
            filename = f"export_{export_type.lower().replace(' ', '_')}_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        
        if file_format == "JSON" or export_type == "Full Dataset (JSON)":
            # JSON export
            if export_type == "Full Dataset (JSON)":
                # Full dataset with captures and marks
                captures = db.get_captures(limit=100000, labeled_only=labeled_only,
                                          start_date=start_date, end_date=end_date)
                full_data = []
                for capture in captures:
                    capture_detail = db.get_capture(capture['capture_id'])
                    if capture_detail:
                        full_data.append(capture_detail)
                json_data = json.dumps(full_data, indent=2, default=str)
            else:
                json_data = df.to_json(orient='records', indent=2)
            
            filename = f"export_{export_type.lower().replace(' ', '_')}_{timestamp}.json"
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=filename,
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        # Save to exports folder
        if st.button("💾 Save to Exports Folder", use_container_width=True):
            exports_path = Path(config.storage.exports_path)
            exports_path.mkdir(parents=True, exist_ok=True)
            
            if file_format == "CSV" and export_type != "Full Dataset (JSON)":
                filepath = exports_path / f"export_{export_type.lower().replace(' ', '_')}_{timestamp}.csv"
                df.to_csv(filepath, index=False)
            else:
                filepath = exports_path / f"export_{export_type.lower().replace(' ', '_')}_{timestamp}.json"
                if export_type == "Full Dataset (JSON)":
                    captures = db.get_captures(limit=100000, labeled_only=labeled_only)
                    full_data = []
                    for capture in captures:
                        capture_detail = db.get_capture(capture['capture_id'])
                        if capture_detail:
                            full_data.append(capture_detail)
                    with open(filepath, 'w') as f:
                        json.dump(full_data, f, indent=2, default=str)
                else:
                    df.to_json(filepath, orient='records', indent=2)
            
            st.success(f"Saved to {filepath}")

else:
    if export_type == "Full Dataset (JSON)":
        # Handle full dataset export
        captures = db.get_captures(limit=100000, labeled_only=labeled_only,
                                  start_date=start_date, end_date=end_date)
        
        if captures:
            st.subheader("Data Preview")
            st.write(f"Total captures: {len(captures)}")
            
            # Show sample
            if captures:
                sample = db.get_capture(captures[0]['capture_id'])
                if sample:
                    st.json(sample)
            
            st.markdown("---")
            
            # Export button
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            full_data = []
            with st.spinner("Preparing export..."):
                for capture in captures:
                    capture_detail = db.get_capture(capture['capture_id'])
                    if capture_detail:
                        full_data.append(capture_detail)
            
            json_data = json.dumps(full_data, indent=2, default=str)
            filename = f"export_full_dataset_{timestamp}.json"
            
            st.download_button(
                label="📥 Download Full Dataset (JSON)",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
        else:
            st.info("No data available for export")
    else:
        st.info("No data available for export")

# Export history
st.markdown("---")
st.subheader("Export History")

exports_path = Path(config.storage.exports_path)
if exports_path.exists():
    export_files = list(exports_path.glob("export_*"))
    export_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if export_files:
        for f in export_files[:10]:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f.name)
            with col2:
                st.write(f"{f.stat().st_size / 1024:.1f} KB")
            with col3:
                modified = datetime.fromtimestamp(f.stat().st_mtime)
                st.write(modified.strftime("%Y-%m-%d %H:%M"))
    else:
        st.info("No previous exports found")
else:
    st.info("Exports folder not yet created")
