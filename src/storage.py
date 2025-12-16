"""Image file storage management."""

import cv2
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import logging

from .models import MarkMeasurement, TyreDetectionResult
from .config import get_config, StorageConfig, CaptureConfig

# Set up logging
logger = logging.getLogger(__name__)


class ImageStorage:
    """Manages saving and loading of inspection images."""
    
    def __init__(self, 
                 storage_config: Optional[StorageConfig] = None,
                 capture_config: Optional[CaptureConfig] = None):
        self.storage_config = storage_config or get_config().storage
        self.capture_config = capture_config or get_config().capture
        
        # Create directories
        self.captures_path = Path(self.storage_config.captures_path)
        self.marks_path = Path(self.storage_config.marks_path)
        self.baselines_path = Path(self.storage_config.baselines_path)
        self.exports_path = Path(self.storage_config.exports_path)
        
        for path in [self.captures_path, self.marks_path, 
                     self.baselines_path, self.exports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Cleanup tracking
        self._last_cleanup_time: float = 0
        self._cleanup_interval_seconds: int = 3600  # Default: 1 hour
    
    def generate_capture_id(self, sequence_number: int) -> str:
        """Generate a unique capture ID."""
        now = datetime.now()
        return f"CAP-{now.strftime('%Y%m%d-%H%M%S')}-{sequence_number:04d}"
    
    def _get_date_folder(self, base_path: Path) -> Path:
        """Get or create date-based subfolder."""
        date_folder = base_path / datetime.now().strftime('%Y-%m-%d')
        date_folder.mkdir(parents=True, exist_ok=True)
        return date_folder
    
    def save_capture_images(self, 
                            capture_id: str,
                            full_frame: np.ndarray,
                            tyre_crop: Optional[np.ndarray] = None,
                            annotated_frame: Optional[np.ndarray] = None) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Save capture images.
        
        Returns:
            Tuple of (full_image_path, tyre_crop_path, annotated_path)
        """
        date_folder = self._get_date_folder(self.captures_path)
        quality = self.capture_config.jpeg_quality
        
        # Save full frame
        full_path = str(date_folder / f"{capture_id}_full.jpg")
        if self.capture_config.save_full_frame:
            cv2.imwrite(full_path, full_frame, 
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # Save tyre crop
        crop_path = None
        if tyre_crop is not None and self.capture_config.save_tyre_crop:
            crop_path = str(date_folder / f"{capture_id}_crop.jpg")
            cv2.imwrite(crop_path, tyre_crop,
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # Save annotated frame
        annotated_path = None
        if annotated_frame is not None and self.capture_config.save_annotated:
            annotated_path = str(date_folder / f"{capture_id}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_frame,
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        return full_path, crop_path, annotated_path
    
    def save_mark_images(self, 
                         capture_id: str,
                         frame: np.ndarray,
                         marks: List[MarkMeasurement],
                         padding: int = 10) -> List[str]:
        """
        Save individual mark images.
        
        Returns:
            List of mark image paths
        """
        if not self.capture_config.save_individual_marks:
            return []
        
        date_folder = self._get_date_folder(self.marks_path)
        quality = self.capture_config.jpeg_quality
        paths = []
        
        h, w = frame.shape[:2]
        
        for mark in marks:
            bbox = mark.bbox
            x1 = max(0, bbox.x - padding)
            y1 = max(0, bbox.y - padding)
            x2 = min(w, bbox.x + bbox.w + padding)
            y2 = min(h, bbox.y + bbox.h + padding)
            
            mark_img = frame[y1:y2, x1:x2].copy()
            
            mark_path = str(date_folder / f"{capture_id}_mark_{mark.mark_index}.jpg")
            cv2.imwrite(mark_path, mark_img,
                       [cv2.IMWRITE_JPEG_QUALITY, quality])
            paths.append(mark_path)
        
        return paths
    
    def save_baseline(self, frame: np.ndarray) -> str:
        """Save baseline (empty conveyor) image."""
        now = datetime.now()
        filename = f"baseline_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = str(self.baselines_path / filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path
    
    def load_latest_baseline(self) -> Optional[np.ndarray]:
        """Load the most recent baseline image."""
        baseline_files = list(self.baselines_path.glob('baseline_*.jpg'))
        if not baseline_files:
            return None
        
        latest = max(baseline_files, key=lambda p: p.stat().st_mtime)
        return cv2.imread(str(latest))
    
    def load_image(self, path: str) -> Optional[np.ndarray]:
        """Load an image from path."""
        if not Path(path).exists():
            return None
        return cv2.imread(path)
    
    def create_annotated_frame(self,
                               frame: np.ndarray,
                               tyre_result: Optional[TyreDetectionResult],
                               marks: List[MarkMeasurement],
                               state_name: str = "") -> np.ndarray:
        """Create an annotated frame with detection overlays."""
        annotated = frame.copy()
        
        # Draw tyre boundary
        if tyre_result and tyre_result.is_present:
            if tyre_result.center and tyre_result.radius:
                cv2.circle(annotated, tyre_result.center, tyre_result.radius,
                          (0, 255, 0), 2)
            if tyre_result.bounding_box:
                bbox = tyre_result.bounding_box
                cv2.rectangle(annotated, 
                             (bbox.x, bbox.y),
                             (bbox.x + bbox.w, bbox.y + bbox.h),
                             (0, 255, 0), 1)
        
        # Draw marks
        for mark in marks:
            color = (0, 0, 255) if mark.auto_color == 'red' else (0, 255, 255)
            
            # Draw contour
            if mark.contour is not None:
                cv2.drawContours(annotated, [mark.contour], -1, color, 2)
            
            # Draw bounding box
            bbox = mark.bbox
            cv2.rectangle(annotated,
                         (bbox.x, bbox.y),
                         (bbox.x + bbox.w, bbox.y + bbox.h),
                         color, 1)
            
            # Draw mark index and type
            label = f"{mark.mark_index}: {mark.auto_type[0].upper()}"
            cv2.putText(annotated, label,
                       (bbox.x, bbox.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw circularity value
            circ_text = f"C:{mark.circularity:.2f}"
            cv2.putText(annotated, circ_text,
                       (bbox.x, bbox.y + bbox.h + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw state indicator
        if state_name:
            cv2.putText(annotated, f"State: {state_name}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw mark count
        cv2.putText(annotated, f"Marks: {len(marks)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def get_export_path(self, filename: str) -> str:
        """Get path for export file."""
        return str(self.exports_path / filename)
    
    # =====================================================
    # Cache/Storage Cleanup Methods for Production
    # =====================================================
    
    def cleanup_old_baselines(self, keep_count: int = 10) -> int:
        """
        Remove old baseline images, keeping only the most recent ones.
        
        Args:
            keep_count: Number of most recent baselines to keep
            
        Returns:
            Number of files deleted
        """
        baseline_files = sorted(
            self.baselines_path.glob('baseline_*.jpg'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        deleted = 0
        for old_file in baseline_files[keep_count:]:
            try:
                old_file.unlink()
                deleted += 1
                logger.debug(f"Deleted old baseline: {old_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete baseline {old_file}: {e}")
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old baseline images")
        
        return deleted
    
    def cleanup_live_frame(self) -> bool:
        """
        Remove stale live frame file if it's too old.
        Prevents dashboard from showing very old frames.
        
        Returns:
            True if file was deleted
        """
        live_frame_path = self.captures_path.parent / "live_frame.jpg"
        
        if live_frame_path.exists():
            try:
                import time
                age = time.time() - live_frame_path.stat().st_mtime
                # Remove if older than 1 hour (system likely stopped)
                if age > 3600:
                    live_frame_path.unlink()
                    logger.info(f"Removed stale live frame (age: {age:.0f}s)")
                    return True
            except Exception as e:
                logger.warning(f"Failed to cleanup live frame: {e}")
        
        return False
    
    def cleanup_empty_date_folders(self) -> int:
        """
        Remove empty date folders from captures and marks directories.
        
        Returns:
            Number of folders deleted
        """
        deleted = 0
        
        for base_path in [self.captures_path, self.marks_path]:
            if not base_path.exists():
                continue
                
            for folder in base_path.iterdir():
                if folder.is_dir():
                    try:
                        # Check if folder is empty
                        if not any(folder.iterdir()):
                            folder.rmdir()
                            deleted += 1
                            logger.debug(f"Deleted empty folder: {folder}")
                    except Exception as e:
                        logger.warning(f"Failed to remove folder {folder}: {e}")
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} empty date folders")
        
        return deleted
    
    def get_storage_stats(self) -> dict:
        """
        Get current storage usage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'captures_size_mb': 0,
            'captures_count': 0,
            'marks_size_mb': 0,
            'marks_count': 0,
            'baselines_size_mb': 0,
            'baselines_count': 0,
            'total_size_mb': 0,
        }
        
        def get_dir_stats(path: Path) -> Tuple[int, float]:
            """Get file count and size in MB for a directory."""
            count = 0
            size_bytes = 0
            if path.exists():
                for f in path.rglob('*.jpg'):
                    count += 1
                    size_bytes += f.stat().st_size
            return count, size_bytes / (1024 * 1024)
        
        stats['captures_count'], stats['captures_size_mb'] = get_dir_stats(self.captures_path)
        stats['marks_count'], stats['marks_size_mb'] = get_dir_stats(self.marks_path)
        stats['baselines_count'], stats['baselines_size_mb'] = get_dir_stats(self.baselines_path)
        
        stats['total_size_mb'] = (
            stats['captures_size_mb'] + 
            stats['marks_size_mb'] + 
            stats['baselines_size_mb']
        )
        
        return stats
    
    def periodic_cleanup(self, force: bool = False) -> dict:
        """
        Perform periodic cleanup operations.
        Called automatically during idle periods.
        
        Args:
            force: If True, run cleanup regardless of time since last cleanup
            
        Returns:
            Dictionary with cleanup results
        """
        import time
        current_time = time.time()
        
        # Check if cleanup is needed (default: every hour)
        if not force and (current_time - self._last_cleanup_time) < self._cleanup_interval_seconds:
            return {'skipped': True, 'reason': 'Too soon since last cleanup'}
        
        self._last_cleanup_time = current_time
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'baselines_deleted': 0,
            'empty_folders_deleted': 0,
            'live_frame_cleaned': False,
            'storage_stats': {},
        }
        
        try:
            # Cleanup old baselines (keep last 10)
            results['baselines_deleted'] = self.cleanup_old_baselines(keep_count=10)
            
            # Cleanup empty date folders
            results['empty_folders_deleted'] = self.cleanup_empty_date_folders()
            
            # Cleanup stale live frame
            results['live_frame_cleaned'] = self.cleanup_live_frame()
            
            # Get current storage stats
            results['storage_stats'] = self.get_storage_stats()
            
            logger.info(f"Periodic cleanup completed: {results}")
            
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")
            results['error'] = str(e)
        
        return results
    
    def set_cleanup_interval(self, seconds: int):
        """Set the interval between automatic cleanups."""
        self._cleanup_interval_seconds = max(60, seconds)  # Minimum 1 minute
