"""Data models for the tyre mark inspection system."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import json


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class ColorMetrics:
    h_mean: float
    s_mean: float
    v_mean: float
    h_std: float
    s_std: float
    v_std: float


@dataclass
class MarkMeasurement:
    """Represents a single detected paint mark with all measurements."""
    mark_index: int
    
    # Auto-detected values
    auto_type: str  # 'solid' or 'donut'
    auto_color: str  # 'red' or 'yellow'
    
    # Position
    centroid_x: int
    centroid_y: int
    bbox: BoundingBox
    
    # Measurements
    area_pixels: float
    perimeter_pixels: float
    estimated_diameter_mm: float
    circularity: float
    solidity: float
    eccentricity: float
    convexity: float
    edge_roughness: float
    
    # Color metrics
    color_metrics: ColorMetrics
    
    # Donut specific
    is_donut: bool
    inner_outer_ratio: Optional[float] = None
    
    # Contour data (not stored in DB)
    contour: Optional[any] = None
    mask: Optional[any] = None


@dataclass
class TyreDetectionResult:
    """Result of tyre detection in a frame."""
    is_present: bool
    is_fully_visible: bool
    is_stable: bool
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    center: Optional[Tuple[int, int]] = None
    radius: Optional[int] = None
    diff_score: float = 0.0


@dataclass
class CaptureResult:
    """Complete capture result with all data."""
    capture_id: str
    timestamp: datetime
    sequence_number: int
    
    # Capture metadata
    camera_height_mm: float
    frame_brightness_avg: float
    tyre_detection_confidence: float
    processing_duration_ms: float
    
    # Summary counts
    total_marks_detected: int
    red_solids: int
    red_donuts: int
    yellow_solids: int
    yellow_donuts: int
    is_empty_tyre: bool
    
    # File paths
    full_image_path: Optional[str] = None
    tyre_crop_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    
    # Marks
    marks: List[MarkMeasurement] = field(default_factory=list)


@dataclass
class MarkLabel:
    """Manual label for a mark."""
    label_actual_type: Optional[str] = None
    label_actual_color: Optional[str] = None
    label_quality_rating: Optional[str] = None  # 'excellent', 'good', 'marginal', 'poor'
    label_defects: List[str] = field(default_factory=list)
    label_matches_auto: Optional[bool] = None
    
    def defects_to_json(self) -> str:
        return json.dumps(self.label_defects)
    
    @staticmethod
    def defects_from_json(json_str: str) -> List[str]:
        if not json_str:
            return []
        return json.loads(json_str)


@dataclass
class CaptureLabel:
    """Manual label for a capture."""
    is_labeled: bool = False
    labeled_by: Optional[str] = None
    labeled_at: Optional[datetime] = None
    overall_verdict: Optional[str] = None  # 'acceptable', 'marginal', 'reject', 'needs_review'
    label_notes: Optional[str] = None


# Defect tags for labeling
DEFECT_TAGS = [
    'oval_shape',
    'irregular_edge',
    'incomplete',
    'double_stamp',
    'smeared',
    'off_center',
    'color_faded',
    'color_bleeding',
    'incomplete_ring',
    'filled_center',
    'missed_by_detection',
    'false_positive'
]

# Quality ratings
QUALITY_RATINGS = ['excellent', 'good', 'marginal', 'poor']

# Overall verdicts
OVERALL_VERDICTS = ['acceptable', 'marginal', 'reject', 'needs_review']
