import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
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

def test_save_crop_error(temp_dir):
    tracker = ObjectTracker(snapshot_dir=temp_dir)
    # Use empty frame or invalid bbox to trigger exception
    filepath = tracker.save_crop(None, (0, 0, 0, 0), tracker_id=42, event_type="ENTER")
    assert filepath is None

class MockTensor:
    def __init__(self, data):
        self.data = np.array(data)
    def cpu(self):
        return self
    def numpy(self):
        return self.data

class MockBoxes:
    def __init__(self, xyxy, ids, conf):
        self.xyxy = MockTensor(xyxy)
        self.id = MockTensor(ids) if ids is not None else None
        self.conf = MockTensor(conf)

class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes

def test_process_frame_no_results():
    with patch("src.object_tracker.YOLO") as mock_yolo:
        tracker = ObjectTracker()
        mock_yolo.return_value.track.return_value = []
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        events = tracker.process_frame(frame)
        assert events == []

def test_process_frame_no_boxes_id():
    with patch("src.object_tracker.YOLO") as mock_yolo:
        tracker = ObjectTracker()
        mock_yolo.return_value.track.return_value = [MockResult(boxes=MockBoxes([[10, 10, 20, 20]], None, [0.9]))]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        events = tracker.process_frame(frame)
        assert events == []

def test_process_frame_crossing_events(temp_dir):
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Tripwire at y = 0.5 (middle of frame) horizontal
        tracker = ObjectTracker(tripwire_line=[(0.0, 0.5), (1.0, 0.5)], snapshot_dir=temp_dir)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Frame 1: Person 1 starts at bottom (y = 80, outside/below tripwire)
        # Point A = (0, 50), Point B = (100, 50)
        # P_prev = (50, 80)
        # Side of (50, 80): (100 - 0)*(80 - 50) - (50 - 50)*(50 - 0) = 100 * 30 = 3000 > 0 -> 1 (inside/above or wait, let's verify)
        # Wait, A=(0,50), B=(100,50). P=(50, 80). (B_x - A_x)*(P_y - A_y) = (100-0)*(80-50) = 3000 -> +1.
        # So P=(50, 80) is side +1.
        # Frame 2: Person 1 moves to top (y = 20)
        # P_curr = (50, 20). Side: (100-0)*(20-50) = -3000 -> -1.
        # Transition from +1 to -1.
        # Wait! If prev_side = 1 and curr_side = -1: event_type = "LEAVE"
        
        # Track 1
        boxes_f1 = MockBoxes([[40, 70, 60, 90]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        events_f1 = tracker.process_frame(frame)
        assert events_f1 == []
        assert len(tracker.track_histories[1]) == 1
        assert tracker.track_sides[1] == 1
        
        # Frame 2: Track 1 crosses
        boxes_f2 = MockBoxes([[40, 10, 60, 30]], [1], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        events_f2 = tracker.process_frame(frame)
        assert len(events_f2) == 1
        assert events_f2[0]["event_type"] == "LEAVE"
        assert events_f2[0]["tracker_id"] == 1
        assert events_f2[0]["confidence"] == 0.96
        assert events_f2[0]["snapshot_path"] is not None
        assert os.path.exists(events_f2[0]["snapshot_path"])

        # Frame 3: Track 1 is still active but hasn't crossed again
        boxes_f3 = MockBoxes([[40, 5, 60, 25]], [1], [0.97])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f3)]
        events_f3 = tracker.process_frame(frame)
        assert events_f3 == []

        # Frame 4: Track 1 disappears (clean up dead tracks)
        boxes_f4 = MockBoxes([], [], [])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f4)]
        events_f4 = tracker.process_frame(frame)
        assert events_f4 == []
        assert 1 not in tracker.track_histories
