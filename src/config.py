"""Configuration loader for the tyre mark inspection system."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProjectConfig:
    name: str = "Apollo Tyres Chennai - Paint Mark Inspection POC"
    target_samples: int = 3000


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    mounting_height_mm: float = 1100
    pixels_per_mm: float = 1.2


@dataclass
class CaptureConfig:
    tyre_presence_threshold: float = 0.15
    tyre_fully_visible_margin: int = 50
    empty_conveyor_threshold: float = 0.05
    stability_frames: int = 3
    min_capture_interval_ms: int = 1500
    save_full_frame: bool = True
    save_tyre_crop: bool = True
    save_annotated: bool = True
    save_individual_marks: bool = True
    jpeg_quality: int = 95


@dataclass
class DetectionConfig:
    red_lower1: List[int] = field(default_factory=lambda: [0, 100, 100])
    red_upper1: List[int] = field(default_factory=lambda: [10, 255, 255])
    red_lower2: List[int] = field(default_factory=lambda: [170, 100, 100])
    red_upper2: List[int] = field(default_factory=lambda: [180, 255, 255])
    yellow_lower: List[int] = field(default_factory=lambda: [20, 100, 100])
    yellow_upper: List[int] = field(default_factory=lambda: [35, 255, 255])
    min_mark_area: int = 100
    max_mark_area: int = 2000
    min_circularity_filter: float = 0.5
    donut_inner_ratio_min: float = 0.2
    donut_inner_ratio_max: float = 0.6
    morph_kernel_size: int = 5
    morph_iterations: int = 2


@dataclass
class MeasurementConfig:
    expected_diameter_mm: float = 12.0
    diameter_tolerance_mm: float = 3.0


@dataclass
class QualityConfig:
    """Configuration for quality assessment thresholds.
    
    These thresholds define what constitutes a "good" paint mark.
    Marks are scored based on how close they are to ideal values.
    """
    # Ideal circularity is 1.0 (perfect circle)
    circularity_ideal: float = 1.0
    circularity_min_acceptable: float = 0.75
    circularity_weight: float = 0.35  # Weight in composite score
    
    # Ideal solidity is 1.0 (no concavities)
    solidity_ideal: float = 1.0
    solidity_min_acceptable: float = 0.85
    solidity_weight: float = 0.25
    
    # Ideal eccentricity is 0.0 (perfect circle, not ellipse)
    eccentricity_ideal: float = 0.0
    eccentricity_max_acceptable: float = 0.3
    eccentricity_weight: float = 0.20
    
    # Edge roughness ideal is 0.0 (smooth edges)
    edge_roughness_ideal: float = 0.0
    edge_roughness_max_acceptable: float = 0.3
    edge_roughness_weight: float = 0.20
    
    # Statistical thresholds (sigma multipliers)
    sigma_excellent: float = 1.0  # Within 1 std dev
    sigma_good: float = 2.0       # Within 2 std devs
    sigma_marginal: float = 3.0   # Within 3 std devs


@dataclass
class StorageConfig:
    database_path: str = "./data/inspection.db"
    captures_path: str = "./data/captures"
    marks_path: str = "./data/marks"
    baselines_path: str = "./data/baselines"
    exports_path: str = "./data/exports"


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8501


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    
    if not path.exists():
        print(f"Config file not found at {config_path}, using defaults")
        return Config()
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    config = Config()
    
    if 'project' in data:
        config.project = ProjectConfig(**data['project'])
    
    if 'camera' in data:
        config.camera = CameraConfig(**data['camera'])
    
    if 'capture' in data:
        config.capture = CaptureConfig(**data['capture'])
    
    if 'detection' in data:
        config.detection = DetectionConfig(**data['detection'])
    
    if 'measurement' in data:
        config.measurement = MeasurementConfig(**data['measurement'])
    
    if 'quality' in data:
        config.quality = QualityConfig(**data['quality'])
    
    if 'storage' in data:
        config.storage = StorageConfig(**data['storage'])
    
    if 'dashboard' in data:
        config.dashboard = DashboardConfig(**data['dashboard'])
    
    return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config):
    """Set the global configuration instance."""
    global _config
    _config = config
