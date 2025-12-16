"""Camera interface for frame acquisition."""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from threading import Lock

from .config import get_config, CameraConfig


class Camera:
    """Camera interface for Logitech C922 Pro HD webcam."""
    
    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or get_config().camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = Lock()
        self._is_connected = False
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0
        
    def connect(self) -> bool:
        """Connect to the camera."""
        with self.lock:
            if self._is_connected:
                return True
            
            # Use V4L2 backend explicitly for Linux/RPi
            self.cap = cv2.VideoCapture(self.config.device_id, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                print(f"Failed to open camera device {self.config.device_id}")
                return False
            
            # Set FOURCC codec before resolution for better compatibility
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Warm up camera - discard initial frames
            for _ in range(5):
                self.cap.read()
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera connected: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self._is_connected = True
            return True
    
    def disconnect(self):
        """Disconnect from the camera."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._is_connected = False
            print("Camera disconnected")
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the camera."""
        with self.lock:
            if not self._is_connected or self.cap is None:
                return None
            
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to read frame from camera")
                self._is_connected = False
                return None
            
            self._last_frame = frame
            self._last_frame_time = time.time()
            
            return frame
    
    def read_with_retry(self, max_retries: int = 3, retry_delay: float = 0.5) -> Optional[np.ndarray]:
        """Read a frame with automatic reconnection on failure."""
        for attempt in range(max_retries):
            frame = self.read()
            if frame is not None:
                return frame
            
            print(f"Read attempt {attempt + 1} failed, reconnecting...")
            self.disconnect()
            time.sleep(retry_delay)
            
            if not self.connect():
                continue
        
        return None
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.config.width, self.config.height)
    
    def get_brightness(self, frame: Optional[np.ndarray] = None) -> float:
        """Calculate average brightness of a frame."""
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class MockCamera(Camera):
    """Mock camera for testing without hardware."""
    
    def __init__(self, config: Optional[CameraConfig] = None):
        super().__init__(config)
        self._frame_count = 0
        self._mock_tyre_present = False
        self._mock_marks = []
    
    def connect(self) -> bool:
        self._is_connected = True
        print("Mock camera connected")
        return True
    
    def disconnect(self):
        self._is_connected = False
        print("Mock camera disconnected")
    
    def read(self) -> Optional[np.ndarray]:
        if not self._is_connected:
            return None
        
        # Generate a test frame
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background (conveyor)
        
        if self._mock_tyre_present:
            # Draw a tyre (dark circle)
            center = (self.config.width // 2, self.config.height // 2)
            radius = min(self.config.width, self.config.height) // 3
            cv2.circle(frame, center, radius, (30, 30, 30), -1)
            
            # Draw some paint marks
            for i, (dx, dy, color, is_donut) in enumerate(self._mock_marks):
                mark_center = (center[0] + dx, center[1] + dy)
                if color == 'red':
                    bgr_color = (0, 0, 200)
                else:
                    bgr_color = (0, 200, 200)
                
                if is_donut:
                    cv2.circle(frame, mark_center, 15, bgr_color, 3)
                else:
                    cv2.circle(frame, mark_center, 12, bgr_color, -1)
        
        self._frame_count += 1
        self._last_frame = frame
        self._last_frame_time = time.time()
        
        return frame
    
    def set_tyre_present(self, present: bool, marks: list = None):
        """Set mock tyre presence and marks for testing."""
        self._mock_tyre_present = present
        if marks:
            self._mock_marks = marks
        elif present:
            # Default test marks
            self._mock_marks = [
                (-50, -50, 'red', False),
                (50, -50, 'yellow', False),
                (-50, 50, 'red', True),
                (50, 50, 'yellow', True),
            ]
