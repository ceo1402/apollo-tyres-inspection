"""Measurement calculations for detected marks."""

import cv2
import numpy as np
from typing import Optional, Tuple

from .mark_detector import DetectedBlob
from .mark_classifier import MarkClassifier
from .models import MarkMeasurement, BoundingBox, ColorMetrics
from .config import get_config, MeasurementConfig, DetectionConfig


class MeasurementCalculator:
    """Calculates all measurements for detected marks."""
    
    def __init__(self, 
                 measurement_config: Optional[MeasurementConfig] = None,
                 detection_config: Optional[DetectionConfig] = None,
                 pixels_per_mm: float = 1.2):
        self.measurement_config = measurement_config or get_config().measurement
        self.detection_config = detection_config or get_config().detection
        self.pixels_per_mm = pixels_per_mm
        self.classifier = MarkClassifier(self.detection_config)
    
    def measure(self, frame: np.ndarray, blob: DetectedBlob, 
                mark_index: int) -> MarkMeasurement:
        """
        Calculate all measurements for a detected blob.
        
        Args:
            frame: Original BGR frame
            blob: Detected blob
            mark_index: Index of the mark
            
        Returns:
            MarkMeasurement with all calculated values
        """
        contour = blob.contour
        
        # Basic measurements
        area_pixels = cv2.contourArea(contour)
        perimeter_pixels = cv2.arcLength(contour, True)
        
        # Estimated diameter in mm
        diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
        estimated_diameter_mm = diameter_pixels / self.pixels_per_mm
        
        # Circularity
        if perimeter_pixels > 0:
            circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2)
        else:
            circularity = 0.0
        
        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_pixels / hull_area if hull_area > 0 else 1.0
        
        # Eccentricity (from fitted ellipse)
        eccentricity = self._calculate_eccentricity(contour)
        
        # Convexity
        hull_perimeter = cv2.arcLength(hull, True)
        convexity = hull_perimeter / perimeter_pixels if perimeter_pixels > 0 else 1.0
        
        # Edge roughness
        edge_roughness = self._calculate_edge_roughness(contour)
        
        # Color metrics
        color_metrics = self._calculate_color_metrics(frame, blob.mask)
        
        # Classify as solid or donut
        mark_type, is_donut, inner_outer_ratio = self.classifier.classify(frame, blob)
        
        return MarkMeasurement(
            mark_index=mark_index,
            auto_type=mark_type,
            auto_color=blob.color,
            centroid_x=blob.center[0],
            centroid_y=blob.center[1],
            bbox=blob.bbox,
            area_pixels=area_pixels,
            perimeter_pixels=perimeter_pixels,
            estimated_diameter_mm=estimated_diameter_mm,
            circularity=circularity,
            solidity=solidity,
            eccentricity=eccentricity,
            convexity=convexity,
            edge_roughness=edge_roughness,
            color_metrics=color_metrics,
            is_donut=is_donut,
            inner_outer_ratio=inner_outer_ratio,
            contour=contour,
            mask=blob.mask
        )
    
    def _calculate_eccentricity(self, contour: np.ndarray) -> float:
        """Calculate eccentricity from fitted ellipse."""
        if len(contour) < 5:
            return 0.0
        
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (width, height), angle = ellipse
            
            if height == 0:
                return 0.0
            
            a = max(width, height) / 2
            b = min(width, height) / 2
            
            if a == 0:
                return 0.0
            
            # Eccentricity formula: sqrt(1 - (b/a)^2)
            eccentricity = np.sqrt(1 - (b / a) ** 2)
            return eccentricity
        except:
            return 0.0
    
    def _calculate_edge_roughness(self, contour: np.ndarray) -> float:
        """Calculate edge roughness using contour approximation."""
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        
        # Approximate contour with different epsilon values
        epsilon_base = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon_base, True)
        
        # More points in approximation = rougher edge
        # Normalize by expected points for a smooth circle
        expected_points = 20  # Approximate points for a smooth circle
        roughness = max(0, (len(approx) - expected_points) / expected_points)
        
        return min(1.0, roughness)
    
    def _calculate_color_metrics(self, frame: np.ndarray, 
                                  mask: np.ndarray) -> ColorMetrics:
        """Calculate color statistics within the mark region."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract pixels within mask
        h_values = hsv[:, :, 0][mask > 0]
        s_values = hsv[:, :, 1][mask > 0]
        v_values = hsv[:, :, 2][mask > 0]
        
        if len(h_values) == 0:
            return ColorMetrics(0, 0, 0, 0, 0, 0)
        
        return ColorMetrics(
            h_mean=float(np.mean(h_values)),
            s_mean=float(np.mean(s_values)),
            v_mean=float(np.mean(v_values)),
            h_std=float(np.std(h_values)),
            s_std=float(np.std(s_values)),
            v_std=float(np.std(v_values))
        )
    
    def set_pixels_per_mm(self, pixels_per_mm: float):
        """Update the pixels per mm calibration value."""
        self.pixels_per_mm = pixels_per_mm


def count_marks_by_type(marks: list) -> dict:
    """Count marks by type and color."""
    counts = {
        'red_solids': 0,
        'red_donuts': 0,
        'yellow_solids': 0,
        'yellow_donuts': 0,
        'total': len(marks)
    }
    
    for mark in marks:
        if mark.auto_color == 'red':
            if mark.is_donut:
                counts['red_donuts'] += 1
            else:
                counts['red_solids'] += 1
        else:  # yellow
            if mark.is_donut:
                counts['yellow_donuts'] += 1
            else:
                counts['yellow_solids'] += 1
    
    return counts
