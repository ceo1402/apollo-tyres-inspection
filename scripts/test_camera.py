#!/usr/bin/env python3
"""
Camera connectivity test for the tyre mark inspection system.
"""

import sys
from pathlib import Path
import cv2
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.camera import Camera, MockCamera


def test_camera():
    print("=" * 60)
    print("Apollo Tyres - Camera Test Utility")
    print("=" * 60)
    
    config = load_config(str(project_root / "config.yaml"))
    
    print(f"\nCamera Configuration:")
    print(f"  Device ID: {config.camera.device_id}")
    print(f"  Resolution: {config.camera.width}x{config.camera.height}")
    print(f"  FPS: {config.camera.fps}")
    
    print("\nTesting camera connection...")
    camera = Camera(config.camera)
    
    if not camera.connect():
        print("\n❌ Failed to connect to camera!")
        print("\nTroubleshooting:")
        print("  1. Check if camera is connected via USB")
        print("  2. Try a different USB port")
        print("  3. Check if another application is using the camera")
        print("  4. Try changing device_id in config.yaml (0, 1, 2, etc.)")
        return False
    
    print("✅ Camera connected successfully!")
    
    # Test frame capture
    print("\nTesting frame capture...")
    frame = camera.read()
    
    if frame is None:
        print("❌ Failed to capture frame!")
        camera.disconnect()
        return False
    
    print(f"✅ Frame captured: {frame.shape[1]}x{frame.shape[0]}")
    print(f"   Brightness: {camera.get_brightness(frame):.1f}")
    
    # FPS test
    print("\nTesting frame rate (5 seconds)...")
    frame_count = 0
    start_time = time.time()
    
    while time.time() - start_time < 5:
        frame = camera.read()
        if frame is not None:
            frame_count += 1
    
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed
    
    print(f"✅ Captured {frame_count} frames in {elapsed:.1f}s")
    print(f"   Actual FPS: {actual_fps:.1f}")
    
    # Display test
    print("\nOpening live preview window...")
    print("Press 'q' to quit, 's' to save a test image")
    
    cv2.namedWindow('Camera Test')
    
    while True:
        frame = camera.read()
        if frame is None:
            continue
        
        # Add info overlay
        cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {actual_fps:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Brightness: {camera.get_brightness(frame):.1f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"test_capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    cv2.destroyAllWindows()
    camera.disconnect()
    
    print("\n✅ Camera test complete!")
    return True


def test_mock_camera():
    print("\n" + "=" * 60)
    print("Testing Mock Camera")
    print("=" * 60)
    
    config = load_config(str(project_root / "config.yaml"))
    camera = MockCamera(config.camera)
    
    camera.connect()
    camera.set_tyre_present(True)
    
    print("\nGenerating test frame with mock tyre...")
    frame = camera.read()
    
    if frame is not None:
        cv2.imwrite("mock_test.jpg", frame)
        print("✅ Saved mock_test.jpg")
    
    camera.disconnect()


def main():
    success = test_camera()
    
    if not success:
        print("\n" + "-" * 60)
        response = input("Would you like to test with mock camera instead? (y/n): ")
        if response.lower() == 'y':
            test_mock_camera()


if __name__ == "__main__":
    main()
