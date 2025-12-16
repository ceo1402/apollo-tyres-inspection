"""Mark detection using color segmentation and blob detection."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .config import get_config, DetectionConfig
from .models import BoundingBox


@dataclass
class DetectedBlob:
    """Represents a detected color blob before classification."""
    contour: np.ndarray
    color: str  # 'red' or 'yellow'
    center: Tuple[int, int]
    area: float
    bbox: BoundingBox
    mask: np.ndarray


class MarkDetector:
    """Detects paint marks using color segmentation."""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or get_config().detection
        
        # HSV ranges
        self.red_lower1 = np.array(self.config.red_lower1)
        self.red_upper1 = np.array(self.config.red_upper1)
        self.red_lower2 = np.array(self.config.red_lower2)
        self.red_upper2 = np.array(self.config.red_upper2)
        self.yellow_lower = np.array(self.config.yellow_lower)
        self.yellow_upper = np.array(self.config.yellow_upper)
        
    def detect(self, frame: np.ndarray, 
               tyre_mask: Optional[np.ndarray] = None) -> List[DetectedBlob]:
        """
        Detect paint marks in the frame.
        
        Args:
            frame: BGR image
            tyre_mask: Optional mask to restrict detection to tyre area
            
        Returns:
            List of detected blobs
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply tyre mask if provided
        if tyre_mask is not None:
            hsv = cv2.bitwise_and(hsv, hsv, mask=tyre_mask)
        
        # Detect red marks
        red_blobs = self._detect_color(frame, hsv, 'red', tyre_mask)
        
        # Detect yellow marks
        yellow_blobs = self._detect_color(frame, hsv, 'yellow', tyre_mask)
        
        return red_blobs + yellow_blobs
    
    def _detect_color(self, frame: np.ndarray, hsv: np.ndarray, 
                      color: str, tyre_mask: Optional[np.ndarray]) -> List[DetectedBlob]:
        """Detect marks of a specific color."""
        # Create color mask
        if color == 'red':
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            color_mask = cv2.bitwise_or(mask1, mask2)
        else:  # yellow
            color_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        # Apply tyre mask
        if tyre_mask is not None:
            color_mask = cv2.bitwise_and(color_mask, tyre_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_CLOSE, kernel, 
            iterations=self.config.morph_iterations
        )
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_OPEN, kernel, 
            iterations=1
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.config.min_mark_area or area > self.config.max_mark_area:
                continue
            
            # Filter by circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.config.min_circularity_filter:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(x=x, y=y, w=w, h=h)
            
            # Get center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            # Create individual mask for this blob
            blob_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(blob_mask, [contour], -1, 255, -1)
            
            blobs.append(DetectedBlob(
                contour=contour,
                color=color,
                center=(cx, cy),
                area=area,
                bbox=bbox,
                mask=blob_mask
            ))
        
        return blobs
    
    def update_hsv_ranges(self, color: str, lower: List[int], upper: List[int], 
                          lower2: Optional[List[int]] = None, 
                          upper2: Optional[List[int]] = None):
        """Update HSV ranges for a color (for calibration)."""
        if color == 'red':
            self.red_lower1 = np.array(lower)
            self.red_upper1 = np.array(upper)
            if lower2 is not None:
                self.red_lower2 = np.array(lower2)
            if upper2 is not None:
                self.red_upper2 = np.array(upper2)
        elif color == 'yellow':
            self.yellow_lower = np.array(lower)
            self.yellow_upper = np.array(upper)
    
    def get_color_mask_preview(self, frame: np.ndarray, color: str) -> np.ndarray:
        """Get a preview of the color mask for calibration."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if color == 'red':
            mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            return cv2.bitwise_or(mask1, mask2)
        else:
            return cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
