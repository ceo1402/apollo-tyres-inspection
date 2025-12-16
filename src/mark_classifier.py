"""Mark classifier for solid vs donut classification."""

import cv2
import numpy as np
from typing import Optional, Tuple

from .mark_detector import DetectedBlob
from .config import get_config, DetectionConfig


class MarkClassifier:
    """Classifies marks as solid dots or donuts."""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or get_config().detection
    
    def classify(self, frame: np.ndarray, blob: DetectedBlob) -> Tuple[str, bool, Optional[float]]:
        """
        Classify a detected blob as solid or donut.
        
        Returns:
            Tuple of (type, is_donut, inner_outer_ratio)
            type: 'solid' or 'donut'
            is_donut: Boolean
            inner_outer_ratio: Ratio of inner to outer diameter (None for solid)
        """
        # Get the region of interest
        bbox = blob.bbox
        padding = 5
        x1 = max(0, bbox.x - padding)
        y1 = max(0, bbox.y - padding)
        x2 = min(frame.shape[1], bbox.x + bbox.w + padding)
        y2 = min(frame.shape[0], bbox.y + bbox.h + padding)
        
        roi = frame[y1:y2, x1:x2]
        roi_mask = blob.mask[y1:y2, x1:x2]
        
        if roi.size == 0 or roi_mask.size == 0:
            return 'solid', False, None
        
        # Method 1: Check for inner contours (holes)
        has_hole, inner_ratio = self._check_inner_contour(roi_mask)
        
        if has_hole:
            # Verify the ratio is within expected range for donuts
            if (self.config.donut_inner_ratio_min <= inner_ratio <= 
                self.config.donut_inner_ratio_max):
                return 'donut', True, inner_ratio
        
        # Method 2: Check solidity
        solidity = self._calculate_solidity(blob.contour)
        
        # Donuts typically have lower solidity due to the hole
        if solidity < 0.85:
            # Additional check using intensity profile
            is_donut, ratio = self._intensity_profile_check(roi, roi_mask, blob.color)
            if is_donut:
                return 'donut', True, ratio
        
        return 'solid', False, None
    
    def _check_inner_contour(self, mask: np.ndarray) -> Tuple[bool, float]:
        """Check for inner contours (holes) in the mask."""
        # Find contours with hierarchy
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if hierarchy is None or len(contours) < 2:
            return False, 0.0
        
        hierarchy = hierarchy[0]
        
        # Look for contours with a parent (inner contours)
        outer_idx = None
        inner_areas = []
        outer_area = 0
        
        for i, (cnt, h) in enumerate(zip(contours, hierarchy)):
            # h = [next, prev, child, parent]
            if h[3] == -1:  # No parent = outer contour
                area = cv2.contourArea(cnt)
                if area > outer_area:
                    outer_area = area
                    outer_idx = i
        
        if outer_idx is None or outer_area == 0:
            return False, 0.0
        
        # Find inner contours (children of outer)
        for i, h in enumerate(hierarchy):
            if h[3] == outer_idx:  # Parent is the outer contour
                inner_areas.append(cv2.contourArea(contours[i]))
        
        if not inner_areas:
            return False, 0.0
        
        total_inner_area = sum(inner_areas)
        
        # Calculate ratio of inner to outer diameter (using area)
        # diameter ratio = sqrt(area ratio)
        area_ratio = total_inner_area / outer_area
        diameter_ratio = np.sqrt(area_ratio)
        
        return True, diameter_ratio
    
    def _calculate_solidity(self, contour: np.ndarray) -> float:
        """Calculate solidity of a contour."""
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 1.0
        
        return area / hull_area
    
    def _intensity_profile_check(self, roi: np.ndarray, mask: np.ndarray, 
                                  color: str) -> Tuple[bool, float]:
        """Check intensity profile for donut pattern."""
        if roi.size == 0 or mask.size == 0:
            return False, 0.0
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Get saturation channel (colors are saturated, background is not)
        saturation = hsv[:, :, 1]
        
        # Find center
        h, w = mask.shape
        center = (w // 2, h // 2)
        
        # Sample points in concentric circles
        max_radius = min(h, w) // 2
        if max_radius < 5:
            return False, 0.0
        
        radii = np.linspace(0, max_radius, 10)
        intensities = []
        
        for r in radii:
            if r < 2:
                # Center point
                if mask[center[1], center[0]] > 0:
                    intensities.append(saturation[center[1], center[0]])
                else:
                    intensities.append(0)
            else:
                # Sample points on circle
                angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                ring_values = []
                for angle in angles:
                    x = int(center[0] + r * np.cos(angle))
                    y = int(center[1] + r * np.sin(angle))
                    if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                        ring_values.append(saturation[y, x])
                
                if ring_values:
                    intensities.append(np.mean(ring_values))
                else:
                    intensities.append(0)
        
        intensities = np.array(intensities)
        
        # Donut pattern: low center, high ring, low outside
        if len(intensities) < 5:
            return False, 0.0
        
        center_intensity = np.mean(intensities[:2])
        ring_intensity = np.mean(intensities[2:6])
        
        # Check for donut pattern (center significantly lower than ring)
        if ring_intensity > 0 and center_intensity / ring_intensity < 0.6:
            # Estimate inner/outer ratio based on where intensity drops
            threshold = ring_intensity * 0.5
            inner_idx = 0
            for i, val in enumerate(intensities):
                if val > threshold:
                    inner_idx = i
                    break
            
            if inner_idx > 0:
                inner_ratio = inner_idx / len(intensities) * 2
                return True, min(0.6, max(0.2, inner_ratio))
        
        return False, 0.0
