"""Analytics page - Threshold explorer, statistical distributions, and quality scoring."""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.database import Database
from src.quality import QualityAnalyzer, compute_batch_quality_scores

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Analytics & Quality Assessment")

# Initialize resources
@st.cache_resource
def init_resources():
    config = load_config(str(project_root / "config.yaml"))
    db = Database(config.storage)
    return config, db

config, db = init_resources()
analyzer = QualityAnalyzer(config.quality)

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
    # Add quality scores to data
    marks_data = compute_batch_quality_scores(marks_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(marks_data)
    
    # Add combined type column
    df['mark_type'] = df['auto_color'] + '_' + df['auto_type']
    
    # Filter by mark types
    df = df[df['mark_type'].isin(mark_types)]
    
    if len(df) == 0:
        st.warning("No marks match the selected filters.")
    else:
        # Tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Quality Overview", 
            "🔔 Statistical Distributions",
            "🎯 Threshold Explorer",
            "📋 Data Table"
        ])
        
        with tab1:
            st.subheader("Quality Score Overview")
            st.markdown("""
            Quality scores combine multiple geometric measurements to objectively assess paint mark quality.
            Each mark receives a score from 0-100 based on circularity, solidity, eccentricity, and edge roughness.
            """)
            
            # Overall quality metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_score = df['quality_score'].mean()
                st.metric("Average Quality Score", f"{avg_score:.1f}/100")
            with col2:
                excellent_pct = (df['quality_grade'] == 'excellent').sum() / len(df) * 100
                st.metric("Excellent Grade", f"{excellent_pct:.1f}%")
            with col3:
                good_pct = (df['quality_grade'].isin(['excellent', 'good'])).sum() / len(df) * 100
                st.metric("Good or Better", f"{good_pct:.1f}%")
            with col4:
                poor_pct = (df['quality_grade'] == 'poor').sum() / len(df) * 100
                st.metric("Poor Grade", f"{poor_pct:.1f}%", delta_color="inverse")
            
            # Grade distribution
            col1, col2 = st.columns(2)
            
            with col1:
                grade_counts = df['quality_grade'].value_counts()
                grade_order = ['excellent', 'good', 'marginal', 'poor']
                grade_counts = grade_counts.reindex(grade_order).fillna(0)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=grade_counts.index,
                        y=grade_counts.values,
                        marker_color=['#28a745', '#17a2b8', '#ffc107', '#dc3545']
                    )
                ])
                fig.update_layout(
                    title="Quality Grade Distribution",
                    xaxis_title="Grade",
                    yaxis_title="Count",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Quality score histogram
                fig = px.histogram(
                    df, x='quality_score',
                    nbins=20,
                    color='quality_grade',
                    color_discrete_map={
                        'excellent': '#28a745',
                        'good': '#17a2b8', 
                        'marginal': '#ffc107',
                        'poor': '#dc3545'
                    },
                    title="Quality Score Distribution"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Component scores breakdown
            st.subheader("Component Score Breakdown")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Circularity Score", f"{df['circularity_score'].mean():.1f}")
            with col2:
                st.metric("Avg Solidity Score", f"{df['solidity_score'].mean():.1f}")
            with col3:
                st.metric("Avg Eccentricity Score", f"{df['eccentricity_score'].mean():.1f}")
            with col4:
                st.metric("Avg Edge Roughness Score", f"{df['edge_roughness_score'].mean():.1f}")
            
            # Radar chart for average component scores
            avg_scores = [
                df['circularity_score'].mean(),
                df['solidity_score'].mean(),
                df['eccentricity_score'].mean(),
                df['edge_roughness_score'].mean()
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=avg_scores + [avg_scores[0]],  # Close the shape
                theta=['Circularity', 'Solidity', 'Eccentricity', 'Edge Roughness', 'Circularity'],
                fill='toself',
                name='Average Scores'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Average Component Scores (Radar Chart)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Quality by mark type
            st.subheader("Quality by Mark Type")
            
            type_quality = df.groupby('mark_type').agg({
                'quality_score': 'mean',
                'circularity': 'mean',
                'solidity': 'mean',
                'eccentricity': 'mean'
            }).round(3)
            
            st.dataframe(type_quality, use_container_width=True)
        
        with tab2:
            st.subheader("Statistical Distributions (Bell Curves)")
            st.markdown("""
            These distributions show the spread of measurements across all captured marks.
            The bell curve (normal distribution) overlay helps identify:
            - **Central tendency**: Where most marks cluster
            - **Spread**: How consistent the marking process is
            - **Outliers**: Marks that deviate significantly from the norm
            
            **Sigma (σ) Reference Lines:**
            - ±1σ (green): ~68% of marks fall here - **Excellent**
            - ±2σ (yellow): ~95% of marks fall here - **Good**
            - ±3σ (red): ~99.7% of marks fall here - **Marginal**
            """)
            
            # Metrics for bell curves
            metrics = [
                ('circularity', 'Circularity', 'Higher is better (1.0 = perfect circle)'),
                ('solidity', 'Solidity', 'Higher is better (1.0 = no concavities)'),
                ('eccentricity', 'Eccentricity', 'Lower is better (0.0 = perfect circle)'),
                ('edge_roughness', 'Edge Roughness', 'Lower is better (0.0 = smooth edge)'),
                ('estimated_diameter_mm', 'Diameter (mm)', 'Should be consistent')
            ]
            
            for metric, title, description in metrics:
                if metric not in df.columns:
                    continue
                    
                values = df[metric].dropna()
                if len(values) == 0:
                    continue
                
                stats = analyzer.compute_distribution_stats(values.tolist())
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create histogram with normal curve overlay
                    fig = go.Figure()
                    
                    # Histogram
                    fig.add_trace(go.Histogram(
                        x=values,
                        nbinsx=30,
                        name='Observed',
                        opacity=0.7,
                        histnorm='probability density'
                    ))
                    
                    # Normal distribution curve
                    if stats.std > 0:
                        x_curve, y_curve = analyzer.generate_normal_curve(
                            stats.mean, stats.std, 
                            stats.min - stats.std, 
                            stats.max + stats.std
                        )
                        fig.add_trace(go.Scatter(
                            x=x_curve, y=y_curve,
                            mode='lines',
                            name='Normal Distribution',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Add sigma reference lines
                        for sigma, color, label in [
                            (1, 'green', '±1σ'),
                            (2, 'orange', '±2σ'),
                            (3, 'red', '±3σ')
                        ]:
                            fig.add_vline(x=stats.mean - sigma * stats.std, 
                                         line_dash="dash", line_color=color, 
                                         annotation_text=f"-{sigma}σ" if sigma == 1 else None)
                            fig.add_vline(x=stats.mean + sigma * stats.std, 
                                         line_dash="dash", line_color=color,
                                         annotation_text=f"+{sigma}σ" if sigma == 1 else None)
                        
                        # Mean line
                        fig.add_vline(x=stats.mean, line_dash="solid", line_color="blue",
                                     annotation_text="Mean")
                    
                    fig.update_layout(
                        title=f"{title} Distribution",
                        xaxis_title=title,
                        yaxis_title="Density",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"**{title}**")
                    st.caption(description)
                    st.write(f"**Mean:** {stats.mean:.4f}")
                    st.write(f"**Std Dev:** {stats.std:.4f}")
                    st.write(f"**Median:** {stats.median:.4f}")
                    st.write(f"**Range:** {stats.min:.4f} - {stats.max:.4f}")
                    st.write(f"**IQR:** {stats.q1:.4f} - {stats.q3:.4f}")
                    st.write(f"**Count:** {stats.count}")
                
                st.markdown("---")
        
        with tab3:
            st.subheader("Threshold Explorer")
            st.markdown("""
            Adjust thresholds to find optimal values for classifying marks as pass/fail.
            Use the labeled data (if available) to validate threshold effectiveness.
            """)
            
            # Threshold sliders
            col1, col2 = st.columns(2)
            
            with col1:
                circularity_range = st.slider(
                    "Circularity Range",
                    min_value=0.0, max_value=1.0,
                    value=(0.75, 1.0),
                    step=0.01,
                    help="Filter marks by circularity"
                )
                
                solidity_range = st.slider(
                    "Solidity Range",
                    min_value=0.0, max_value=1.0,
                    value=(0.85, 1.0),
                    step=0.01,
                    help="Filter marks by solidity"
                )
            
            with col2:
                eccentricity_max = st.slider(
                    "Max Eccentricity",
                    min_value=0.0, max_value=1.0,
                    value=0.3,
                    step=0.01,
                    help="Maximum allowed eccentricity"
                )
                
                diameter_range = st.slider(
                    "Diameter Range (mm)",
                    min_value=0.0, max_value=30.0,
                    value=(8.0, 16.0),
                    step=0.5,
                    help="Filter marks by estimated diameter"
                )
            
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
            
            # Threshold impact analysis with labels
            if 'label_quality_rating' in df.columns and df['label_quality_rating'].notna().any():
                st.markdown("---")
                st.subheader("Threshold vs Manual Labels")
                
                labeled_df = df[df['label_quality_rating'].notna()].copy()
                
                if len(labeled_df) > 0:
                    labeled_df['is_good'] = labeled_df['label_quality_rating'].isin(['excellent', 'good'])
                    
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
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        st.metric("F1 Score", f"{f1:.1%}")
                    
                    # Confusion matrix
                    fig = go.Figure(data=go.Heatmap(
                        z=[[tp, fp], [fn, tn]],
                        x=['Labeled Good', 'Labeled Bad'],
                        y=['Pass Filter', 'Fail Filter'],
                        text=[[str(tp), str(fp)], [str(fn), str(tn)]],
                        texttemplate="%{text}",
                        colorscale='Blues',
                        showscale=False
                    ))
                    fig.update_layout(title="Confusion Matrix", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Distribution charts with thresholds
            st.markdown("---")
            st.subheader("Metric Distributions with Thresholds")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df, x='circularity', nbins=50,
                    color='passes_thresholds',
                    color_discrete_map={True: 'green', False: 'red'},
                    title="Circularity Distribution"
                )
                fig.add_vline(x=circularity_range[0], line_dash="dash", line_color="orange")
                fig.add_vline(x=circularity_range[1], line_dash="dash", line_color="orange")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    df, x='solidity', nbins=50,
                    color='passes_thresholds',
                    color_discrete_map={True: 'green', False: 'red'},
                    title="Solidity Distribution"
                )
                fig.add_vline(x=solidity_range[0], line_dash="dash", line_color="orange")
                fig.add_vline(x=solidity_range[1], line_dash="dash", line_color="orange")
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            fig = px.scatter(
                df, x='circularity', y='solidity',
                color='passes_thresholds',
                color_discrete_map={True: 'green', False: 'red'},
                title="Circularity vs Solidity",
                opacity=0.6,
                hover_data=['quality_score', 'quality_grade']
            )
            fig.add_hline(y=solidity_range[0], line_dash="dash", line_color="gray")
            fig.add_vline(x=circularity_range[0], line_dash="dash", line_color="gray")
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
                capture_ids = gallery_df['capture_id'].unique()[:8]
                cols = st.columns(4)
                for i, capture_id in enumerate(capture_ids):
                    with cols[i % 4]:
                        capture = db.get_capture(capture_id)
                        if capture and capture.get('annotated_image_path') and Path(capture['annotated_image_path']).exists():
                            st.image(capture['annotated_image_path'], 
                                    caption=f"{capture_id[-13:]}", use_container_width=True)
                            cap_marks = gallery_df[gallery_df['capture_id'] == capture_id]
                            for _, mark in cap_marks.iterrows():
                                st.caption(f"M{mark['mark_index']}: Q={mark['quality_score']:.0f}, C={mark['circularity']:.2f}")
        
        with tab4:
            st.subheader("Data Table")
            
            # Column selection
            default_cols = ['capture_id', 'mark_index', 'auto_type', 'auto_color', 
                           'quality_score', 'quality_grade', 'circularity', 'solidity', 
                           'eccentricity', 'estimated_diameter_mm']
            
            available_cols = [c for c in default_cols if c in df.columns]
            
            show_columns = st.multiselect(
                "Columns to Display",
                options=df.columns.tolist(),
                default=available_cols
            )
            
            if show_columns:
                st.dataframe(
                    df[show_columns].sort_values('quality_score', ascending=False).head(100),
                    use_container_width=True
                )
            
            # Export option
            st.markdown("---")
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Data as CSV",
                csv_data,
                "marks_with_quality_scores.csv",
                "text/csv"
            )
