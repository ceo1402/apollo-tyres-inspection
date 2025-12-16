"""Labeling form UI components."""

import streamlit as st
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.models import DEFECT_TAGS, QUALITY_RATINGS, OVERALL_VERDICTS, CaptureLabel, MarkLabel


def capture_label_form(capture: Dict[str, Any], 
                       form_key: str = "capture_label") -> Optional[CaptureLabel]:
    """
    Display capture labeling form.
    
    Args:
        capture: Capture dictionary from database
        form_key: Unique key for the form
        
    Returns:
        CaptureLabel if form was submitted, None otherwise
    """
    with st.form(key=form_key):
        st.subheader("Capture Label")
        
        col1, col2 = st.columns(2)
        
        with col1:
            verdict_options = [''] + OVERALL_VERDICTS
            current_verdict = capture.get('overall_verdict', '')
            verdict_idx = verdict_options.index(current_verdict) if current_verdict in verdict_options else 0
            
            overall_verdict = st.selectbox(
                "Overall Verdict *",
                options=verdict_options,
                index=verdict_idx,
                help="Overall quality assessment"
            )
            
            labeled_by = st.text_input(
                "Labeled By *",
                value=capture.get('labeled_by', ''),
                placeholder="Your name"
            )
        
        with col2:
            label_notes = st.text_area(
                "Notes",
                value=capture.get('label_notes', ''),
                placeholder="Additional observations...",
                height=100
            )
        
        submitted = st.form_submit_button("Save Label", type="primary")
        
        if submitted:
            if not overall_verdict:
                st.error("Please select an overall verdict")
                return None
            if not labeled_by:
                st.error("Please enter your name")
                return None
            
            return CaptureLabel(
                is_labeled=True,
                labeled_by=labeled_by,
                labeled_at=datetime.now(),
                overall_verdict=overall_verdict,
                label_notes=label_notes
            )
    
    return None


def mark_label_form(mark: Dict[str, Any], 
                    form_key: str = "mark_label") -> Optional[MarkLabel]:
    """
    Display mark labeling form.
    
    Args:
        mark: Mark dictionary from database
        form_key: Unique key for the form
        
    Returns:
        MarkLabel if form was submitted, None otherwise
    """
    with st.expander(f"Label Mark {mark.get('mark_index', 'N/A')}", expanded=False):
        # Auto-detected values display
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Auto Type:** {mark.get('auto_type', 'N/A')}")
        with col2:
            st.write(f"**Auto Color:** {mark.get('auto_color', 'N/A')}")
        
        st.markdown("---")
        
        # Manual labels
        col1, col2 = st.columns(2)
        
        with col1:
            type_options = ['', 'solid', 'donut']
            current_type = mark.get('label_actual_type', '')
            type_idx = type_options.index(current_type) if current_type in type_options else 0
            
            actual_type = st.selectbox(
                "Actual Type",
                options=type_options,
                index=type_idx,
                key=f"{form_key}_type"
            )
            
            quality_options = [''] + QUALITY_RATINGS
            current_quality = mark.get('label_quality_rating', '')
            quality_idx = quality_options.index(current_quality) if current_quality in quality_options else 0
            
            quality_rating = st.selectbox(
                "Quality Rating",
                options=quality_options,
                index=quality_idx,
                key=f"{form_key}_quality"
            )
        
        with col2:
            color_options = ['', 'red', 'yellow']
            current_color = mark.get('label_actual_color', '')
            color_idx = color_options.index(current_color) if current_color in color_options else 0
            
            actual_color = st.selectbox(
                "Actual Color",
                options=color_options,
                index=color_idx,
                key=f"{form_key}_color"
            )
        
        # Defects multi-select
        import json
        current_defects = []
        if mark.get('label_defects'):
            try:
                current_defects = json.loads(mark['label_defects'])
            except:
                current_defects = []
        
        defects = st.multiselect(
            "Defects",
            options=DEFECT_TAGS,
            default=current_defects,
            key=f"{form_key}_defects"
        )
        
        if st.button("Save Mark Label", key=f"{form_key}_save"):
            return MarkLabel(
                label_actual_type=actual_type if actual_type else None,
                label_actual_color=actual_color if actual_color else None,
                label_quality_rating=quality_rating if quality_rating else None,
                label_defects=defects,
                label_matches_auto=(actual_type == mark.get('auto_type') and 
                                   actual_color == mark.get('auto_color')) if actual_type and actual_color else None
            )
    
    return None


def quick_verdict_buttons(capture_id: str) -> Optional[str]:
    """
    Display quick verdict buttons for fast labeling.
    
    Returns:
        Selected verdict or None
    """
    st.write("Quick Verdict:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Accept", key=f"quick_accept_{capture_id}", use_container_width=True):
            return 'acceptable'
    
    with col2:
        if st.button("⚠️ Marginal", key=f"quick_marginal_{capture_id}", use_container_width=True):
            return 'marginal'
    
    with col3:
        if st.button("❌ Reject", key=f"quick_reject_{capture_id}", use_container_width=True):
            return 'reject'
    
    with col4:
        if st.button("❓ Review", key=f"quick_review_{capture_id}", use_container_width=True):
            return 'needs_review'
    
    return None
