import os
import cv2
import numpy as np
import pytest
from src.object_tracker import ObjectTracker

def test_ccw_orientation():
    tracker = ObjectTracker()
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    
    # C is clockwise (right of vector AB)
    C_cw = (0.5, -1.0)
    # C is counterclockwise (left of vector AB)
    C_ccw = (0.5, 1.0)
    # C is collinear
    C_col = (0.5, 0.0)
    
    # NOTE: Our math implementation has:
    # (B[1] - A[1]) * (C[0] - B[0]) - (B[0] - A[0]) * (C[1] - B[1])
    # Let's verify the exact orientation values returned
    assert tracker._get_ccw_orientation(A, B, C_col) == 0
    assert tracker._get_ccw_orientation(A, B, C_ccw) == -1
    assert tracker._get_ccw_orientation(A, B, C_cw) == 1

def test_check_intersection():
    tracker = ObjectTracker()
    
    # Intersecting segments
    A, B = (0.0, 0.0), (2.0, 2.0)
    C, D = (0.0, 2.0), (2.0, 0.0)
    assert tracker._check_intersection(A, B, C, D) is True

    # Non-intersecting segments
    C2, D2 = (3.0, 3.0), (4.0, 4.0)
    assert tracker._check_intersection(A, B, C2, D2) is False

def test_get_point_side():
    tracker = ObjectTracker()
    A = (0.0, 0.0)
    B = (2.0, 0.0)
    
    # Point above line AB
    P_above = (1.0, 1.0)
    # Point below line AB
    P_below = (1.0, -1.0)
    # Collinear point
    P_on = (1.0, 0.0)
    
    # Formula: (B_x - A_x)*(P_y - A_y) - (B_y - A_y)*(P_x - A_x)
    # P_above: (2-0)*(1-0) - (0)*(1-0) = 2 > 0 -> +1 (Left/Clockwise side, "inside")
    assert tracker._get_point_side(A, B, P_above) == 1
    assert tracker._get_point_side(A, B, P_below) == -1
    assert tracker._get_point_side(A, B, P_on) == 0

def test_save_crop(temp_dir):
    tracker = ObjectTracker(snapshot_dir=temp_dir)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:50, 10:50, :] = 255 # Draw a square
    
    bbox = (10, 10, 50, 50)
    filepath = tracker.save_crop(frame, bbox, tracker_id=42, event_type="ENTER")
    
    assert filepath is not None
    assert os.path.exists(filepath)
    assert "enter_id42_" in os.path.basename(filepath)
