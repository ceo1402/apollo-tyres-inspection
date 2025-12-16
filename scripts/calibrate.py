#!/usr/bin/env python3
"""
Camera calibration utility for the tyre mark inspection system.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.camera import Camera


def calibrate_colors(camera: Camera, config):
    """Interactive HSV color calibration."""
    print("Color Calibration Mode")
    print("Press 'r' for red, 'y' for yellow, 's' to save, 'q' to quit")
    
    cv2.namedWindow('Calibration')
    cv2.namedWindow('Mask')
    
    # Default values
    h_low, s_low, v_low = 0, 100, 100
    h_high, s_high, v_high = 10, 255, 255
    
    cv2.createTrackbar('H Low', 'Calibration', h_low, 180, lambda x: None)
    cv2.createTrackbar('H High', 'Calibration', h_high, 180, lambda x: None)
    cv2.createTrackbar('S Low', 'Calibration', s_low, 255, lambda x: None)
    cv2.createTrackbar('S High', 'Calibration', s_high, 255, lambda x: None)
    cv2.createTrackbar('V Low', 'Calibration', v_low, 255, lambda x: None)
    cv2.createTrackbar('V High', 'Calibration', v_high, 255, lambda x: None)
    
    current_color = 'red'
    
    while True:
        frame = camera.read()
        if frame is None:
            continue
        
        # Get trackbar values
        h_low = cv2.getTrackbarPos('H Low', 'Calibration')
        h_high = cv2.getTrackbarPos('H High', 'Calibration')
        s_low = cv2.getTrackbarPos('S Low', 'Calibration')
        s_high = cv2.getTrackbarPos('S High', 'Calibration')
        v_low = cv2.getTrackbarPos('V Low', 'Calibration')
        v_high = cv2.getTrackbarPos('V High', 'Calibration')
        
        # Create mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply mask
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Add text overlay
        cv2.putText(frame, f"Calibrating: {current_color.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"HSV: [{h_low},{s_low},{v_low}] - [{h_high},{s_high},{v_high}]",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Calibration', frame)
        cv2.imshow('Mask', result)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_color = 'red'
            print("Calibrating RED")
        elif key == ord('y'):
            current_color = 'yellow'
            print("Calibrating YELLOW")
        elif key == ord('s'):
            print(f"\n{current_color.upper()} HSV Range:")
            print(f"  Lower: [{h_low}, {s_low}, {v_low}]")
            print(f"  Upper: [{h_high}, {s_high}, {v_high}]")
            print("\nAdd to config.yaml:")
            if current_color == 'red':
                print(f"  red_lower1: [{h_low}, {s_low}, {v_low}]")
                print(f"  red_upper1: [{h_high}, {s_high}, {v_high}]")
            else:
                print(f"  yellow_lower: [{h_low}, {s_low}, {v_low}]")
                print(f"  yellow_upper: [{h_high}, {s_high}, {v_high}]")
    
    cv2.destroyAllWindows()


def calibrate_scale(camera: Camera, config):
    """Calibrate pixels per mm using a reference object."""
    print("\nScale Calibration")
    print("Place a reference object of known size in view")
    print("Click two points to measure distance")
    print("Press 'q' to quit, 'c' to capture points")
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
    
    cv2.namedWindow('Scale Calibration')
    cv2.setMouseCallback('Scale Calibration', mouse_callback)
    
    while True:
        frame = camera.read()
        if frame is None:
            continue
        
        display = frame.copy()
        
        # Draw points
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (pt[0]+10, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw line between points
        if len(points) >= 2:
            cv2.line(display, points[-2], points[-1], (0, 255, 0), 2)
            pixel_dist = np.sqrt((points[-1][0] - points[-2][0])**2 + 
                                (points[-1][1] - points[-2][1])**2)
            cv2.putText(display, f"Distance: {pixel_dist:.1f} pixels",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display, "Click to place points, 'c' to calculate, 'q' to quit",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Scale Calibration', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c') and len(points) >= 2:
            pixel_dist = np.sqrt((points[-1][0] - points[-2][0])**2 + 
                                (points[-1][1] - points[-2][1])**2)
            
            known_mm = float(input("Enter the known distance in mm: "))
            pixels_per_mm = pixel_dist / known_mm
            
            print(f"\nCalibration Result:")
            print(f"  Pixel distance: {pixel_dist:.1f}")
            print(f"  Known distance: {known_mm} mm")
            print(f"  Pixels per mm: {pixels_per_mm:.2f}")
            print(f"\nAdd to config.yaml:")
            print(f"  pixels_per_mm: {pixels_per_mm:.2f}")
            
            points = []
        elif key == ord('r'):
            points = []
    
    cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("Apollo Tyres - Camera Calibration Utility")
    print("=" * 60)
    
    config = load_config(str(project_root / "config.yaml"))
    camera = Camera(config.camera)
    
    if not camera.connect():
        print("Failed to connect to camera")
        sys.exit(1)
    
    print("\nCalibration Options:")
    print("1. Color calibration (HSV ranges)")
    print("2. Scale calibration (pixels per mm)")
    print("3. Both")
    print("q. Quit")
    
    choice = input("\nSelect option: ").strip().lower()
    
    try:
        if choice == '1':
            calibrate_colors(camera, config)
        elif choice == '2':
            calibrate_scale(camera, config)
        elif choice == '3':
            calibrate_colors(camera, config)
            calibrate_scale(camera, config)
        elif choice == 'q':
            pass
        else:
            print("Invalid option")
    finally:
        camera.disconnect()
    
    print("\nCalibration complete")


if __name__ == "__main__":
    main()
