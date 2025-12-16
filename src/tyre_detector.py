"""Tyre presence and position detection."""

import cv2
import numpy as np
from typing import Optional, Tuple

from .models import TyreDetectionResult, BoundingBox
from .config import get_config, CaptureConfig


class TyreDetector:
    """Detects tyre presence and position in frames."""
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or get_config().capture
        self._baseline: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_center: Optional[Tuple[int, int]] = None
        self._motion_threshold = 10  # pixels
        
    def set_baseline(self, frame: np.ndarray):
        """Set the baseline (empty conveyor) image."""
        self._baseline = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._baseline = cv2.GaussianBlur(self._baseline, (21, 21), 0)
    
    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None
    
    def detect(self, frame: np.ndarray) -> TyreDetectionResult:
        """
        Detect tyre presence and position in the frame.
        
        Returns TyreDetectionResult with detection information.
        """
        if self._baseline is None:
            # No baseline, assume empty
            return TyreDetectionResult(
                is_present=False,
                is_fully_visible=False,
                is_stable=False,
                confidence=0.0
            )
        
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Compute difference from baseline
        diff = cv2.absdiff(self._baseline, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Calculate presence score
        total_pixels = thresh.shape[0] * thresh.shape[1]
        changed_pixels = cv2.countNonZero(thresh)
        diff_score = changed_pixels / total_pixels
        
        # Check if tyre is present
        is_present = diff_score > self.config.tyre_presence_threshold
        
        if not is_present:
            self._prev_frame = gray
            self._prev_center = None
            return TyreDetectionResult(
                is_present=False,
                is_fully_visible=False,
                is_stable=False,
                confidence=0.0,
                diff_score=diff_score
            )
        
        # Find the largest contour (tyre)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return TyreDetectionResult(
                is_present=True,
                is_fully_visible=False,
                is_stable=False,
                confidence=diff_score,
                diff_score=diff_score
            )
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = BoundingBox(x=x, y=y, w=w, h=h)
        
        # Calculate center
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = (x + w // 2, y + h // 2)
        
        # Try to fit a circle (tyre is circular)
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(largest_contour)
        circle_center = (int(circle_x), int(circle_y))
        circle_radius = int(radius)
        
        # Check if fully visible (margins from edges)
        margin = self.config.tyre_fully_visible_margin
        frame_h, frame_w = frame.shape[:2]
        is_fully_visible = (
            x >= margin and
            y >= margin and
            x + w <= frame_w - margin and
            y + h <= frame_h - margin
        )
        
        # Check stability (compare with previous frame)
        is_stable = False
        if self._prev_center is not None:
            motion = np.sqrt(
                (center[0] - self._prev_center[0]) ** 2 +
                (center[1] - self._prev_center[1]) ** 2
            )
            is_stable = motion < self._motion_threshold
        
        # Update previous state
        self._prev_frame = gray
        self._prev_center = center
        
        # Calculate confidence based on circularity
        contour_area = cv2.contourArea(largest_contour)
        circle_area = np.pi * radius ** 2
        circularity = contour_area / circle_area if circle_area > 0 else 0
        confidence = min(1.0, circularity * diff_score * 2)
        
        return TyreDetectionResult(
            is_present=True,
            is_fully_visible=is_fully_visible,
            is_stable=is_stable,
            confidence=confidence,
            bounding_box=bbox,
            center=center,
            radius=circle_radius,
            diff_score=diff_score
        )
    
    def get_tyre_mask(self, frame: np.ndarray, result: TyreDetectionResult) -> Optional[np.ndarray]:
        """Create a mask for the tyre region."""
        if not result.is_present or result.center is None or result.radius is None:
            return None
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, result.center, result.radius, 255, -1)
        return mask
    
    def crop_tyre(self, frame: np.ndarray, result: TyreDetectionResult, 
                  padding: int = 20) -> Optional[np.ndarray]:
        """Crop the frame to the tyre region."""
        if not result.is_present or result.bounding_box is None:
            return None
        
        bbox = result.bounding_box
        h, w = frame.shape[:2]
        
        x1 = max(0, bbox.x - padding)
        y1 = max(0, bbox.y - padding)
        x2 = min(w, bbox.x + bbox.w + padding)
        y2 = min(h, bbox.y + bbox.h + padding)
        
        return frame[y1:y2, x1:x2].copy()
