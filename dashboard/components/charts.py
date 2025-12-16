"""Plotly chart builders for analytics."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


def circularity_histogram(df: pd.DataFrame, 
                          threshold_min: float = 0.5,
                          threshold_max: float = 1.0,
                          color_by: Optional[str] = None) -> go.Figure:
    """
    Create circularity distribution histogram.
    """
    fig = px.histogram(
        df, x='circularity',
        nbins=50,
        color=color_by,
        title="Circularity Distribution",
        labels={'circularity': 'Circularity', 'count': 'Count'}
    )
    
    fig.add_vline(x=threshold_min, line_dash="dash", line_color="red",
                  annotation_text=f"Min: {threshold_min}")
    fig.add_vline(x=threshold_max, line_dash="dash", line_color="red",
                  annotation_text=f"Max: {threshold_max}")
    
    fig.update_layout(bargap=0.1)
    return fig


def solidity_histogram(df: pd.DataFrame,
                       threshold_min: float = 0.7,
                       threshold_max: float = 1.0,
                       color_by: Optional[str] = None) -> go.Figure:
    """
    Create solidity distribution histogram.
    """
    fig = px.histogram(
        df, x='solidity',
        nbins=50,
        color=color_by,
        title="Solidity Distribution",
        labels={'solidity': 'Solidity', 'count': 'Count'}
    )
    
    fig.add_vline(x=threshold_min, line_dash="dash", line_color="red",
                  annotation_text=f"Min: {threshold_min}")
    fig.add_vline(x=threshold_max, line_dash="dash", line_color="red",
                  annotation_text=f"Max: {threshold_max}")
    
    fig.update_layout(bargap=0.1)
    return fig


def circularity_solidity_scatter(df: pd.DataFrame,
                                  color_by: str = 'mark_type',
                                  threshold_circ: Tuple[float, float] = (0.5, 1.0),
                                  threshold_solid: Tuple[float, float] = (0.7, 1.0)) -> go.Figure:
    """
    Create circularity vs solidity scatter plot.
    """
    fig = px.scatter(
        df, x='circularity', y='solidity',
        color=color_by,
        title="Circularity vs Solidity",
        labels={'circularity': 'Circularity', 'solidity': 'Solidity'},
        opacity=0.6
    )
    
    # Add threshold lines
    fig.add_hline(y=threshold_solid[0], line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=threshold_solid[1], line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=threshold_circ[0], line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=threshold_circ[1], line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add threshold region
    fig.add_shape(
        type="rect",
        x0=threshold_circ[0], x1=threshold_circ[1],
        y0=threshold_solid[0], y1=threshold_solid[1],
        fillcolor="green", opacity=0.1,
        line=dict(width=0)
    )
    
    return fig


def mark_type_pie(df: pd.DataFrame) -> go.Figure:
    """
    Create mark type distribution pie chart.
    """
    if 'mark_type' not in df.columns:
        df = df.copy()
        df['mark_type'] = df['auto_color'] + '_' + df['auto_type']
    
    type_counts = df['mark_type'].value_counts()
    
    fig = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Mark Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    return fig


def diameter_histogram(df: pd.DataFrame,
                       threshold_min: float = 8.0,
                       threshold_max: float = 16.0) -> go.Figure:
    """
    Create diameter distribution histogram.
    """
    fig = px.histogram(
        df, x='estimated_diameter_mm',
        nbins=30,
        title="Diameter Distribution",
        labels={'estimated_diameter_mm': 'Diameter (mm)', 'count': 'Count'}
    )
    
    fig.add_vline(x=threshold_min, line_dash="dash", line_color="red")
    fig.add_vline(x=threshold_max, line_dash="dash", line_color="red")
    
    return fig


def marks_per_capture_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Create marks per capture distribution histogram.
    """
    marks_per_capture = df.groupby('capture_id').size()
    
    fig = px.histogram(
        x=marks_per_capture,
        nbins=10,
        title="Marks per Capture Distribution",
        labels={'x': 'Number of Marks', 'y': 'Count'}
    )
    
    return fig


def confusion_matrix_heatmap(tp: int, fp: int, fn: int, tn: int) -> go.Figure:
    """
    Create confusion matrix heatmap.
    """
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
        xaxis_title="Actual Label",
        yaxis_title="Filter Result"
    )
    
    return fig


def time_series_captures(df: pd.DataFrame) -> go.Figure:
    """
    Create time series of captures.
    """
    if 'timestamp' not in df.columns:
        return None
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    
    fig = px.line(
        daily_counts, x='date', y='count',
        title="Captures Over Time",
        labels={'date': 'Date', 'count': 'Captures'}
    )
    
    return fig


def quality_distribution_bar(df: pd.DataFrame) -> go.Figure:
    """
    Create quality rating distribution bar chart.
    """
    if 'label_quality_rating' not in df.columns:
        return None
    
    quality_counts = df['label_quality_rating'].value_counts()
    
    colors = {
        'excellent': 'green',
        'good': 'lightgreen',
        'marginal': 'orange',
        'poor': 'red'
    }
    
    fig = px.bar(
        x=quality_counts.index,
        y=quality_counts.values,
        title="Quality Rating Distribution",
        labels={'x': 'Quality Rating', 'y': 'Count'},
        color=quality_counts.index,
        color_discrete_map=colors
    )
    
    return fig


def multi_metric_subplot(df: pd.DataFrame) -> go.Figure:
    """
    Create subplot with multiple metric distributions.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Circularity', 'Solidity', 'Eccentricity', 'Diameter (mm)')
    )
    
    fig.add_trace(
        go.Histogram(x=df['circularity'], nbinsx=30, name='Circularity'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=df['solidity'], nbinsx=30, name='Solidity'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Histogram(x=df['eccentricity'], nbinsx=30, name='Eccentricity'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=df['estimated_diameter_mm'], nbinsx=30, name='Diameter'),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Metric Distributions",
        showlegend=False,
        height=600
    )
    
    return fig
