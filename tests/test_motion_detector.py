import numpy as np
import pytest
from src.motion_detector import MotionDetector

def test_motion_detector_init():
    detector = MotionDetector(threshold=0.01, min_contour_area=100, alpha=0.1)
    assert detector.threshold == 0.01
    assert detector.min_contour_area == 100
    assert detector.alpha == 0.1
    assert detector.background_accumulator is None

def test_motion_detector_first_frame():
    """First frame should initialize background accumulator and return False."""
    detector = MotionDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    assert detector.detect(frame) is False
    assert detector.background_accumulator is not None

def test_motion_detector_no_motion():
    """Subsequent frames with identical content should return False (no motion)."""
    detector = MotionDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Init
    detector.detect(frame)
    # Check
    assert detector.detect(frame) is False

def test_motion_detector_with_motion():
    """Frame with substantial change should register as motion."""
    # Use small threshold to make testing easy
    detector = MotionDetector(threshold=0.01, min_contour_area=10, alpha=0.5)
    
    # 1. Init background with black image
    frame_black = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.detect(frame_black)
    
    # 2. Introduce a large white block (50x50, 25% of 100x100 frame)
    frame_motion = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_motion[25:75, 25:75, :] = 255
    
    # detect should return True
    assert detector.detect(frame_motion) is True

def test_motion_detector_reset():
    detector = MotionDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    detector.detect(frame)
    assert detector.background_accumulator is not None
    
    detector.reset()
    assert detector.background_accumulator is None


def test_motion_detector_with_roi():
    # Set ROI to center 50%
    detector = MotionDetector(threshold=0.01, min_contour_area=10, alpha=0.5, roi=[(0.25, 0.25), (0.75, 0.75)])
    
    # 1. Init background with black frame
    frame_black = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.detect(frame_black)
    
    # 2. Introduce motion OUTSIDE of the ROI (at top-left 0:20, 0:20)
    frame_motion_outside = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_motion_outside[0:20, 0:20, :] = 255
    # Should NOT register motion because it's outside the ROI
    assert detector.detect(frame_motion_outside) is False
    
    # 3. Introduce motion INSIDE of the ROI (at center 45:55, 45:55)
    frame_motion_inside = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_motion_inside[45:55, 45:55, :] = 255
    # Should register motion
    assert detector.detect(frame_motion_inside) is True


def test_motion_detector_with_polygon_roi():
    # Set ROI to a triangular polygon: top-left corner region
    detector = MotionDetector(threshold=0.01, min_contour_area=10, alpha=0.5, roi=[(0.0, 0.0), (0.8, 0.0), (0.0, 0.8)])
    
    # 1. Init background with black frame
    frame_black = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.detect(frame_black)
    
    # 2. Introduce motion OUTSIDE of the triangular ROI (at bottom-right corner 80:100, 80:100)
    frame_motion_outside = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_motion_outside[80:100, 80:100, :] = 255
    # Should NOT register motion
    assert detector.detect(frame_motion_outside) is False
    
    # 3. Introduce motion INSIDE of the triangular ROI (at top-left 10:20, 10:20)
    frame_motion_inside = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_motion_inside[10:20, 10:20, :] = 255
    # Should register motion
    assert detector.detect(frame_motion_inside) is True


def test_motion_detector_invalid_roi():
    # Empty or short ROI list should fall back to entire frame
    detector = MotionDetector(threshold=0.01, min_contour_area=10, alpha=0.5, roi=[])
    frame_black = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.detect(frame_black)
    
    frame_motion = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_motion[45:55, 45:55, :] = 255
    assert detector.detect(frame_motion) is True
