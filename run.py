#!/usr/bin/env python3
"""
Apollo Tyres Chennai - Tyre Paint Mark Inspection System
Main entry point for running capture and dashboard.

Production-ready implementation with:
- Periodic cache cleanup
- Improved error handling
- Optimized frame rates for conveyor capture
- Logging for monitoring
"""

import sys
import os
import time
import signal
import threading
import queue
import logging
import cv2
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import load_config, get_config
from src.camera import Camera, MockCamera
from src.state_machine import ConveyorStateMachine, StateAction, ConveyorState
from src.tyre_detector import TyreDetector
from src.mark_detector import MarkDetector
from src.measurement import MeasurementCalculator, count_marks_by_type
from src.database import Database
from src.storage import ImageStorage
from src.models import CaptureResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inspection.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class InspectionSystem:
    """Main inspection system coordinating all components.
    
    Production-ready implementation with:
    - Automatic periodic cleanup
    - Configurable frame rates
    - Error recovery and logging
    - Edge case handling
    """
    
    def __init__(self, config_path: str = "config.yaml", use_mock_camera: bool = False):
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        if use_mock_camera:
            self.camera = MockCamera(self.config.camera)
        else:
            self.camera = Camera(self.config.camera)
        
        self.state_machine = ConveyorStateMachine(self.config.capture)
        self.tyre_detector = TyreDetector(self.config.capture)
        self.mark_detector = MarkDetector(self.config.detection)
        self.measurement_calc = MeasurementCalculator(
            self.config.measurement,
            self.config.detection,
            self.config.camera.pixels_per_mm
        )
        self.database = Database(self.config.storage)
        self.storage = ImageStorage(self.config.storage, self.config.capture)
        
        # State
        self._running = False
        self._capture_thread = None
        self._cleanup_thread = None
        self._frame_queue = queue.Queue(maxsize=5)
        self._latest_frame = None
        self._latest_annotated = None
        self._latest_state = None
        self._latest_marks = []
        self._capture_count = 0
        self._session_captures = 0
        self._lock = threading.Lock()
        
        # Control flags
        self._force_capture = False
        self._update_baseline = False
        
        # Production monitoring
        self._last_tyre_time = time.time()
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10
        
        # Frame rate control - adaptive based on state
        self._frame_delay_idle = 0.1  # 10 FPS when idle (no tyre)
        self._frame_delay_active = 0.033  # 30 FPS when tyre present
        
        # Cleanup configuration
        self._cleanup_interval_seconds = 3600  # Run cleanup every hour
        self._last_cleanup_time = 0
        
        logger.info("InspectionSystem initialized")
    
    def start(self) -> bool:
        """Start the inspection system."""
        if self._running:
            return True
        
        # Connect camera
        if not self.camera.connect():
            logger.error("Failed to connect to camera")
            return False
        
        # Load or capture baseline
        baseline = self.storage.load_latest_baseline()
        if baseline is not None:
            self.tyre_detector.set_baseline(baseline)
            logger.info("Loaded existing baseline")
        else:
            logger.info("No baseline found. Capturing new baseline...")
            time.sleep(0.5)
            frame = self.camera.read()
            if frame is not None:
                self.tyre_detector.set_baseline(frame)
                self.storage.save_baseline(frame)
                logger.info("Baseline captured and saved")
        
        # Get sequence number
        self._capture_count = self.database.get_capture_count()
        
        # Start capture thread
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureLoop")
        self._capture_thread.start()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="CleanupLoop")
        self._cleanup_thread.start()
        
        logger.info(f"Inspection system started. Total captures: {self._capture_count}")
        print(f"Inspection system started. Total captures: {self._capture_count}")
        return True
    
    def stop(self):
        """Stop the inspection system."""
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)
        self.camera.disconnect()
        
        # Final cleanup
        self.storage.periodic_cleanup(force=True)
        
        logger.info("Inspection system stopped")
        print("Inspection system stopped")
    
    def _cleanup_loop(self):
        """Background thread for periodic cleanup operations."""
        logger.info("Cleanup loop started")
        
        while self._running:
            try:
                current_time = time.time()
                
                # Check if cleanup is needed
                if current_time - self._last_cleanup_time >= self._cleanup_interval_seconds:
                    logger.info("Running periodic cleanup...")
                    results = self.storage.periodic_cleanup(force=True)
                    self._last_cleanup_time = current_time
                    
                    if not results.get('skipped'):
                        stats = results.get('storage_stats', {})
                        logger.info(f"Cleanup completed. Storage: {stats.get('total_size_mb', 0):.1f}MB, "
                                   f"Captures: {stats.get('captures_count', 0)}")
                
                # Sleep for a minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)
    
    def _capture_loop(self):
        """Main capture loop running in background thread.
        
        Optimized for production with:
        - Adaptive frame rate based on state
        - Error recovery
        - Periodic status logging
        """
        logger.info("Capture loop started")
        frame_count = 0
        last_log_time = time.time()
        
        while self._running:
            try:
                # Read frame
                frame = self.camera.read_with_retry()
                if frame is None:
                    self._consecutive_errors += 1
                    if self._consecutive_errors >= self._max_consecutive_errors:
                        logger.error(f"Too many consecutive read errors ({self._consecutive_errors}), attempting reconnect")
                        self.camera.disconnect()
                        time.sleep(1)
                        if not self.camera.connect():
                            logger.error("Failed to reconnect camera")
                        self._consecutive_errors = 0
                    time.sleep(0.1)
                    continue
                
                self._consecutive_errors = 0  # Reset error counter on success
                frame_count += 1
                
                # Check for forced baseline update
                if self._update_baseline:
                    self.tyre_detector.set_baseline(frame)
                    self.storage.save_baseline(frame)
                    self._update_baseline = False
                    logger.info("Baseline updated")
                    continue
                
                # Detect tyre
                tyre_result = self.tyre_detector.detect(frame)
                
                # Update last tyre time for idle detection
                if tyre_result.is_present:
                    self._last_tyre_time = time.time()
                
                # Check for forced capture
                if self._force_capture and tyre_result.is_present:
                    self.state_machine.force_capture()
                    self._force_capture = False
                
                # Update state machine
                action = self.state_machine.update(tyre_result)
                
                # Handle actions
                marks = []
                if action == StateAction.CAPTURE:
                    marks = self._process_capture(frame, tyre_result)
                elif action == StateAction.UPDATE_BASELINE:
                    self.tyre_detector.set_baseline(frame)
                    self.storage.save_baseline(frame)
                    logger.debug("Automatic baseline update")
                
                # Detect marks for preview (even if not capturing)
                if tyre_result.is_present and not marks:
                    tyre_mask = self.tyre_detector.get_tyre_mask(frame, tyre_result)
                    detected_blobs = self.mark_detector.detect(frame, tyre_mask)
                    marks = [
                        self.measurement_calc.measure(frame, blob, i)
                        for i, blob in enumerate(detected_blobs)
                    ]
                
                # Create annotated frame
                annotated = self.storage.create_annotated_frame(
                    frame, tyre_result, marks, self.state_machine.state_name
                )
                
                # Update shared state
                with self._lock:
                    self._latest_frame = frame
                    self._latest_annotated = annotated
                    self._latest_state = self.state_machine.state
                    self._latest_marks = marks
                
                # Save latest frame to file for dashboard live view
                try:
                    live_frame_path = Path(self.config.storage.captures_path).parent / "live_frame.jpg"
                    cv2.imwrite(str(live_frame_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                except Exception as e:
                    logger.debug(f"Failed to save live frame: {e}")
                
                # Adaptive frame rate - faster when tyre present, slower when idle
                if tyre_result.is_present:
                    time.sleep(self._frame_delay_active)  # 30 FPS for active capture
                else:
                    time.sleep(self._frame_delay_idle)  # 10 FPS when idle to save CPU
                
                # Periodic status logging (every 60 seconds)
                current_time = time.time()
                if current_time - last_log_time >= 60:
                    idle_time = current_time - self._last_tyre_time
                    logger.info(f"Status: Captures={self._capture_count}, "
                               f"Session={self._session_captures}, "
                               f"State={self.state_machine.state_name}, "
                               f"Idle={idle_time:.0f}s")
                    last_log_time = current_time
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def _process_capture(self, frame, tyre_result) -> list:
        """Process a capture when triggered."""
        start_time = time.time()
        
        try:
            # Get tyre mask and crop
            tyre_mask = self.tyre_detector.get_tyre_mask(frame, tyre_result)
            tyre_crop = self.tyre_detector.crop_tyre(frame, tyre_result)
            
            # Detect marks
            detected_blobs = self.mark_detector.detect(frame, tyre_mask)
            
            # Measure marks
            marks = [
                self.measurement_calc.measure(frame, blob, i)
                for i, blob in enumerate(detected_blobs)
            ]
            
            # Count marks by type
            counts = count_marks_by_type(marks)
            
            # Generate capture ID
            sequence_number = self.database.get_next_sequence_number()
            capture_id = self.storage.generate_capture_id(sequence_number)
            
            # Create annotated frame
            annotated = self.storage.create_annotated_frame(
                frame, tyre_result, marks, "CAPTURED"
            )
            
            # Save images
            full_path, crop_path, annotated_path = self.storage.save_capture_images(
                capture_id, frame, tyre_crop, annotated
            )
            
            # Save individual mark images
            mark_paths = self.storage.save_mark_images(capture_id, frame, marks)
            
            # Calculate processing time
            processing_ms = (time.time() - start_time) * 1000
            
            # Create capture result
            capture = CaptureResult(
                capture_id=capture_id,
                timestamp=datetime.now(),
                sequence_number=sequence_number,
                camera_height_mm=self.config.camera.mounting_height_mm,
                frame_brightness_avg=self.camera.get_brightness(frame),
                tyre_detection_confidence=tyre_result.confidence,
                processing_duration_ms=processing_ms,
                total_marks_detected=counts['total'],
                red_solids=counts['red_solids'],
                red_donuts=counts['red_donuts'],
                yellow_solids=counts['yellow_solids'],
                yellow_donuts=counts['yellow_donuts'],
                is_empty_tyre=counts['total'] == 0,
                full_image_path=full_path,
                tyre_crop_path=crop_path,
                annotated_image_path=annotated_path,
                marks=marks
            )
            
            # Save to database
            self.database.save_capture(capture)
            
            # Update counters
            self._capture_count += 1
            self._session_captures += 1
            
            logger.info(f"Captured {capture_id}: {counts['total']} marks, "
                       f"{processing_ms:.1f}ms processing time")
            print(f"Captured {capture_id}: {counts['total']} marks, "
                  f"{processing_ms:.1f}ms processing time")
            
            return marks
            
        except Exception as e:
            logger.error(f"Error processing capture: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def trigger_capture(self):
        """Manually trigger a capture."""
        self._force_capture = True
    
    def update_baseline(self):
        """Request baseline update."""
        self._update_baseline = True
    
    def get_latest_frame(self):
        """Get the latest captured frame."""
        with self._lock:
            return self._latest_frame
    
    def get_latest_annotated(self):
        """Get the latest annotated frame."""
        with self._lock:
            return self._latest_annotated
    
    def get_status(self) -> dict:
        """Get current system status."""
        with self._lock:
            state_info = self.state_machine.get_status_info()
            return {
                'running': self._running,
                'state': state_info['state'],
                'total_captures': self._capture_count,
                'session_captures': self._session_captures,
                'target_samples': self.config.project.target_samples,
                'marks_in_view': len(self._latest_marks),
                'has_baseline': self.tyre_detector.has_baseline,
                'idle_time': time.time() - self._last_tyre_time,
            }
    
    def get_statistics(self) -> dict:
        """Get database statistics."""
        return self.database.get_statistics()
    
    def get_storage_stats(self) -> dict:
        """Get storage usage statistics."""
        return self.storage.get_storage_stats()
    
    def run_cleanup(self) -> dict:
        """Manually trigger cleanup."""
        return self.storage.periodic_cleanup(force=True)


# Global instance for dashboard access
_system_instance = None


def get_system() -> InspectionSystem:
    """Get or create the global system instance."""
    global _system_instance
    if _system_instance is None:
        _system_instance = InspectionSystem()
    return _system_instance


def run_dashboard():
    """Run the Streamlit dashboard."""
    import subprocess
    dashboard_path = project_root / "dashboard" / "app.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.address", "0.0.0.0",
        "--server.port", "8501"
    ])


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Apollo Tyres Chennai - Tyre Paint Mark Inspection System"
    )
    parser.add_argument(
        "--mode", choices=["full", "capture", "dashboard"],
        default="full",
        help="Run mode: full (capture + dashboard), capture only, or dashboard only"
    )
    parser.add_argument(
        "--mock-camera", action="store_true",
        help="Use mock camera for testing"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        run_dashboard()
    else:
        # Start system
        global _system_instance
        _system_instance = InspectionSystem(
            config_path=args.config,
            use_mock_camera=args.mock_camera
        )
        
        if not _system_instance.start():
            logger.error("Failed to start inspection system")
            print("Failed to start inspection system")
            sys.exit(1)
        
        # Handle shutdown
        def signal_handler(sig, frame):
            print("\nShutting down...")
            logger.info("Shutdown signal received")
            _system_instance.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if args.mode == "full":
            # Run dashboard in main thread
            run_dashboard()
        else:
            # Capture only mode - run until interrupted
            print("Running in capture-only mode. Press Ctrl+C to stop.")
            logger.info("Running in capture-only mode")
            while True:
                time.sleep(1)
                status = _system_instance.get_status()
                print(f"\rCaptures: {status['total_captures']}/{status['target_samples']} "
                      f"| State: {status['state']} "
                      f"| Idle: {status['idle_time']:.0f}s", end="")


if __name__ == "__main__":
    main()
