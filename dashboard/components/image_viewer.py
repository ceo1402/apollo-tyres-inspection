"""Zoomable image viewer component."""

import streamlit as st
from pathlib import Path
from typing import Optional, List, Tuple
import base64


def image_viewer(image_path: str, marks: Optional[List[dict]] = None,
                 show_controls: bool = True) -> Optional[int]:
    """
    Display an image with optional mark overlays and zoom controls.
    
    Args:
        image_path: Path to the image file
        marks: List of mark dictionaries with position info
        show_controls: Whether to show zoom controls
        
    Returns:
        Selected mark index if a mark was clicked, None otherwise
    """
    if not Path(image_path).exists():
        st.warning(f"Image not found: {image_path}")
        return None
    
    # Display image
    st.image(image_path, use_container_width=True)
    
    # Show mark info if available
    if marks:
        selected_mark = None
        cols = st.columns(min(len(marks), 4))
        
        for i, mark in enumerate(marks):
            with cols[i % 4]:
                color_emoji = "🔴" if mark.get('auto_color') == 'red' else "🟡"
                type_emoji = "⭕" if mark.get('is_donut') else "⚫"
                
                if st.button(f"{color_emoji}{type_emoji} Mark {mark.get('mark_index', i)}", 
                           key=f"mark_btn_{i}"):
                    selected_mark = i
        
        return selected_mark
    
    return None


def mark_detail_viewer(mark: dict, image_path: Optional[str] = None):
    """
    Display detailed view of a single mark.
    
    Args:
        mark: Mark dictionary with all measurements
        image_path: Optional path to mark crop image
    """
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if image_path and Path(image_path).exists():
            st.image(image_path, caption=f"Mark {mark.get('mark_index', 'N/A')}")
        else:
            st.info("Mark image not available")
    
    with col2:
        st.write("**Detection Results:**")
        st.write(f"- Type: {mark.get('auto_type', 'N/A')}")
        st.write(f"- Color: {mark.get('auto_color', 'N/A')}")
        
        st.write("**Measurements:**")
        st.write(f"- Circularity: {mark.get('circularity', 0):.3f}")
        st.write(f"- Solidity: {mark.get('solidity', 0):.3f}")
        st.write(f"- Diameter: {mark.get('estimated_diameter_mm', 0):.2f} mm")
        st.write(f"- Eccentricity: {mark.get('eccentricity', 0):.3f}")
        
        if mark.get('is_donut'):
            st.write(f"- Inner/Outer Ratio: {mark.get('inner_outer_ratio', 'N/A')}")


def comparison_viewer(image1_path: str, image2_path: str,
                     label1: str = "Image 1", label2: str = "Image 2"):
    """
    Display two images side by side for comparison.
    """
    col1, col2 = st.columns(2)
    
    with col1:
        if Path(image1_path).exists():
            st.image(image1_path, caption=label1, use_container_width=True)
        else:
            st.warning(f"Image not found: {image1_path}")
    
    with col2:
        if Path(image2_path).exists():
            st.image(image2_path, caption=label2, use_container_width=True)
        else:
            st.warning(f"Image not found: {image2_path}")
