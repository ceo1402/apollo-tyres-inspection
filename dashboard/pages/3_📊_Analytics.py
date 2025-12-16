"""Analytics page - Threshold explorer and statistics."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.database import Database

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Analytics & Threshold Explorer")

# Initialize resources
@st.cache_resource
def init_resources():
    config = load_config(str(project_root / "config.yaml"))
    db = Database(config.storage)
    return config, db

config, db = init_resources()

# Sidebar filters
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    value=[],
    help="Filter data by date range"
)

labeled_only = st.sidebar.checkbox("Labeled Data Only", value=False)

mark_types = st.sidebar.multiselect(
    "Mark Types",
    options=['red_solid', 'red_donut', 'yellow_solid', 'yellow_donut'],
    default=['red_solid', 'red_donut', 'yellow_solid', 'yellow_donut']
)

# Threshold sliders
st.sidebar.markdown("---")
st.sidebar.header("Threshold Settings")

circularity_range = st.sidebar.slider(
    "Circularity Range",
    min_value=0.0, max_value=1.0,
    value=(0.5, 1.0),
    step=0.01,
    help="Filter marks by circularity"
)

solidity_range = st.sidebar.slider(
    "Solidity Range",
    min_value=0.0, max_value=1.0,
    value=(0.7, 1.0),
    step=0.01,
    help="Filter marks by solidity"
)

eccentricity_max = st.sidebar.slider(
    "Max Eccentricity",
    min_value=0.0, max_value=1.0,
    value=0.5,
    step=0.01,
    help="Maximum allowed eccentricity"
)

diameter_range = st.sidebar.slider(
    "Diameter Range (mm)",
    min_value=0.0, max_value=30.0,
    value=(8.0, 16.0),
    step=0.5,
    help="Filter marks by estimated diameter"
)

# Get data
start_date = date_range[0].isoformat() if len(date_range) >= 1 else None
end_date = date_range[1].isoformat() + "T23:59:59" if len(date_range) >= 2 else None

marks_data = db.get_marks_for_analytics(
    labeled_only=labeled_only,
    start_date=start_date,
    end_date=end_date
)

if not marks_data:
    st.warning("No data available. Start capturing tyres to see analytics.")
else:
    # Convert to DataFrame
    df = pd.DataFrame(marks_data)
    
    # Add combined type column
    df['mark_type'] = df['auto_color'] + '_' + df['auto_type']
    
    # Filter by mark types
    df = df[df['mark_type'].isin(mark_types)]
    
    # Apply threshold filters
    df['passes_thresholds'] = (
        (df['circularity'] >= circularity_range[0]) &
        (df['circularity'] <= circularity_range[1]) &
        (df['solidity'] >= solidity_range[0]) &
        (df['solidity'] <= solidity_range[1]) &
        (df['eccentricity'] <= eccentricity_max) &
        (df['estimated_diameter_mm'] >= diameter_range[0]) &
        (df['estimated_diameter_mm'] <= diameter_range[1])
    )
    
    # Overview metrics
    st.subheader("Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Marks", len(df))
    with col2:
        passing = df['passes_thresholds'].sum()
        st.metric("Passing Thresholds", passing, 
                 delta=f"{passing/len(df)*100:.1f}%" if len(df) > 0 else "0%")
    with col3:
        failing = len(df) - passing
        st.metric("Failing Thresholds", failing)
    with col4:
        avg_circ = df['circularity'].mean()
        st.metric("Avg Circularity", f"{avg_circ:.3f}" if pd.notna(avg_circ) else "N/A")
    
    # Threshold impact analysis
    if 'label_quality_rating' in df.columns and df['label_quality_rating'].notna().any():
        st.markdown("---")
        st.subheader("Threshold Impact Analysis")
        
        labeled_df = df[df['label_quality_rating'].notna()].copy()
        
        if len(labeled_df) > 0:
            # Define "good" as excellent or good quality
            labeled_df['is_good'] = labeled_df['label_quality_rating'].isin(['excellent', 'good'])
            
            # Calculate confusion matrix
            tp = len(labeled_df[(labeled_df['passes_thresholds']) & (labeled_df['is_good'])])
            fp = len(labeled_df[(labeled_df['passes_thresholds']) & (~labeled_df['is_good'])])
            tn = len(labeled_df[(~labeled_df['passes_thresholds']) & (~labeled_df['is_good'])])
            fn = len(labeled_df[(~labeled_df['passes_thresholds']) & (labeled_df['is_good'])])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = tp + fp + tn + fn
                accuracy = (tp + tn) / total if total > 0 else 0
                st.metric("Accuracy", f"{accuracy:.1%}")
            
            with col2:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                st.metric("Precision", f"{precision:.1%}")
            
            with col3:
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                st.metric("Recall", f"{recall:.1%}")
            
            with col4:
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                st.metric("False Positive Rate", f"{fpr:.1%}")
            
            # Confusion matrix visualization
            fig = go.Figure(data=go.Heatmap(
                z=[[tp, fp], [fn, tn]],
                x=['Labeled Good', 'Labeled Bad'],
                y=['Pass Filter', 'Fail Filter'],
                text=[[str(tp), str(fp)], [str(fn), str(tn)]],
                texttemplate="%{text}",
                colorscale='Blues',
                showscale=False
            ))
            fig.update_layout(
                title="Confusion Matrix: Filter vs Labels",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Distribution charts
    st.markdown("---")
    st.subheader("Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Circularity histogram
        fig = px.histogram(
            df, x='circularity',
            nbins=50,
            color='passes_thresholds',
            color_discrete_map={True: 'green', False: 'red'},
            title="Circularity Distribution"
        )
        fig.add_vline(x=circularity_range[0], line_dash="dash", line_color="orange")
        fig.add_vline(x=circularity_range[1], line_dash="dash", line_color="orange")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Solidity histogram
        fig = px.histogram(
            df, x='solidity',
            nbins=50,
            color='passes_thresholds',
            color_discrete_map={True: 'green', False: 'red'},
            title="Solidity Distribution"
        )
        fig.add_vline(x=solidity_range[0], line_dash="dash", line_color="orange")
        fig.add_vline(x=solidity_range[1], line_dash="dash", line_color="orange")
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Circularity vs Solidity scatter
        fig = px.scatter(
            df, x='circularity', y='solidity',
            color='mark_type',
            title="Circularity vs Solidity",
            opacity=0.6
        )
        # Add threshold lines
        fig.add_hline(y=solidity_range[0], line_dash="dash", line_color="gray")
        fig.add_hline(y=solidity_range[1], line_dash="dash", line_color="gray")
        fig.add_vline(x=circularity_range[0], line_dash="dash", line_color="gray")
        fig.add_vline(x=circularity_range[1], line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mark type distribution
        type_counts = df['mark_type'].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Mark Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Diameter histogram
        fig = px.histogram(
            df, x='estimated_diameter_mm',
            nbins=30,
            color='mark_type',
            title="Diameter Distribution (mm)"
        )
        fig.add_vline(x=diameter_range[0], line_dash="dash", line_color="orange")
        fig.add_vline(x=diameter_range[1], line_dash="dash", line_color="orange")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Marks per capture distribution
        marks_per_capture = df.groupby('capture_id').size()
        fig = px.histogram(
            x=marks_per_capture,
            nbins=10,
            title="Marks per Capture Distribution",
            labels={'x': 'Number of Marks', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample gallery
    st.markdown("---")
    st.subheader("Sample Gallery")
    
    gallery_mode = st.radio(
        "View Mode",
        options=['All', 'Passing', 'Failing'],
        horizontal=True
    )
    
    if gallery_mode == 'Passing':
        gallery_df = df[df['passes_thresholds']]
    elif gallery_mode == 'Failing':
        gallery_df = df[~df['passes_thresholds']]
    else:
        gallery_df = df
    
    if len(gallery_df) == 0:
        st.info("No marks match the current criteria")
    else:
        # Get unique captures
        capture_ids = gallery_df['capture_id'].unique()[:12]
        
        cols = st.columns(4)
        for i, capture_id in enumerate(capture_ids):
            with cols[i % 4]:
                capture = db.get_capture(capture_id)
                if capture and capture.get('annotated_image_path') and Path(capture['annotated_image_path']).exists():
                    st.image(capture['annotated_image_path'], 
                            caption=f"{capture_id[-13:]}", use_container_width=True)
                    
                    # Show mark metrics
                    cap_marks = gallery_df[gallery_df['capture_id'] == capture_id]
                    for _, mark in cap_marks.iterrows():
                        st.caption(f"M{mark['mark_index']}: C={mark['circularity']:.2f}")
    
    # Data table
    st.markdown("---")
    st.subheader("Data Table")
    
    show_columns = st.multiselect(
        "Columns to Display",
        options=df.columns.tolist(),
        default=['capture_id', 'mark_index', 'auto_type', 'auto_color', 
                'circularity', 'solidity', 'estimated_diameter_mm', 'passes_thresholds']
    )
    
    st.dataframe(
        df[show_columns].head(100),
        use_container_width=True
    )
