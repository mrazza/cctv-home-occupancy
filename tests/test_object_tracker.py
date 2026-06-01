import os
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.object_tracker import ObjectTracker

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
        assert tracker.track_confirmed_sides[1] == 1
        
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


def test_object_tracker_save_crop_zero_size(temp_dir):
    """Verifies save_crop returns None if the calculated crop bounding box is of zero size."""
    tracker = ObjectTracker(snapshot_dir=temp_dir)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Bounding box coordinates with size 0: x1=50, y1=50, x2=50, y2=50
    crop_path = tracker.save_crop(frame, (50, 50, 50, 50), 99, "ENTER")
    assert crop_path is None


def test_object_tracker_save_crop_exception(temp_dir):
    """Verifies save_crop handles unexpected exceptions gracefully and returns None."""
    tracker = ObjectTracker(snapshot_dir=temp_dir)
    # Passing None instead of image to raise exception
    crop_path = tracker.save_crop(None, (10, 10, 20, 20), 99, "ENTER")
    assert crop_path is None



def test_object_tracker_get_point_side_almost_collinear():
    """Verifies _get_point_side handles very small non-zero float values as collinear."""
    tracker = ObjectTracker()
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    # P is extremely close to the line segment AB
    P = (0.5, 1e-11)
    assert tracker._get_point_side(A, B, P) == 0


def test_object_tracker_enter_crossing_and_history_pop(temp_dir):
    """Verifies that transition from -1 to 1 triggers ENTER, and history is capped at 10 items."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Tripwire at y = 0.5 (middle of frame) horizontal
        tracker = ObjectTracker(tripwire_line=[(0.0, 0.5), (1.0, 0.5)], snapshot_dir=temp_dir)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Frame 1: Person starts above (y = 20, side = -1)
        boxes_f1 = MockBoxes([[40, 10, 60, 30]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        events_f1 = tracker.process_frame(frame)
        assert events_f1 == []
        assert tracker.track_confirmed_sides[1] == -1
        
        # Frame 2: Moves below (y = 80, side = 1) -> triggers ENTER
        boxes_f2 = MockBoxes([[40, 70, 60, 90]], [1], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        events_f2 = tracker.process_frame(frame)
        assert len(events_f2) == 1
        assert events_f2[0]["event_type"] == "ENTER"
        
        # Now let's push many more frames to trigger popping the history when len > 10
        for i in range(15):
            boxes_loop = MockBoxes([[40 + i, 70, 60 + i, 90]], [1], [0.95])
            mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_loop)]
            tracker.process_frame(frame)
            
        # The history length should be capped at 10
        assert len(tracker.track_histories[1]) == 10


def test_object_tracker_unreachable_else(temp_dir):
    """Verifies fallback when sides don't map to standard ENTER/LEAVE (though mathematically impossible with -1/1)."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        tracker = ObjectTracker(tripwire_line=[(0.0, 0.5), (1.0, 0.5)], snapshot_dir=temp_dir)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Prime the track
        boxes_f1 = MockBoxes([[40, 10, 60, 30]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        tracker.process_frame(frame)
        
        # Manually alter track_confirmed_sides to a non-standard value to hit 'else' branch
        tracker.track_confirmed_sides[1] = 42
        
        # Move across line
        boxes_f2 = MockBoxes([[40, 70, 60, 90]], [1], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        events = tracker.process_frame(frame)
        assert len(events) == 0  # event_type is None


def test_tripwire_jitter_does_not_produce_spurious_events(temp_dir):
    """Regression test reproducing tripwire jitter causing spurious ENTER/LEAVE events."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Tripwire at y = 0.5 (middle of frame) horizontal
        # Set dead_zone_width to 0.1 (10% of frame height -> 10 px total width on 100 px height, i.e., ±5 px)
        tracker = ObjectTracker(
            tripwire_line=[(0.0, 0.5), (1.0, 0.5)],
            snapshot_dir=temp_dir,
            dead_zone_width=0.1
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        all_events = []
        
        # Frame 1: Person starts outside/above (centroid y = 20)
        boxes_f1 = MockBoxes([[40, 10, 60, 30]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        all_events.extend(tracker.process_frame(frame))
        
        # Frame 2: Person crosses to inside/below (centroid y = 80) -> Should trigger ENTER
        boxes_f2 = MockBoxes([[40, 65, 60, 95]], [1], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        all_events.extend(tracker.process_frame(frame))
        
        # Frame 3: Person jitters inside dead zone (centroid y = 48) -> Should NOT trigger event
        boxes_f3 = MockBoxes([[40, 38, 60, 58]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f3)]
        all_events.extend(tracker.process_frame(frame))
        
        # Frame 4: Person jitters inside dead zone (centroid y = 52) -> Should NOT trigger event
        boxes_f4 = MockBoxes([[40, 42, 60, 62]], [1], [0.94])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f4)]
        all_events.extend(tracker.process_frame(frame))
        
        # Frame 5: Person jitters inside dead zone (centroid y = 46) -> Should NOT trigger event
        boxes_f5 = MockBoxes([[40, 36, 60, 56]], [1], [0.93])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f5)]
        all_events.extend(tracker.process_frame(frame))
        
        # Assert total events across all 5 frames is exactly 1 (only the first ENTER)
        assert len(all_events) == 1
        assert all_events[0]["event_type"] == "ENTER"


def test_dead_zone_zero_width_equivalent_to_old_behavior(temp_dir):
    """Verifies that dead_zone_width=0.0 matches old intersection/side crossing behavior."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Tripwire at y = 0.5 horizontal, dead_zone_width = 0.0
        tracker = ObjectTracker(
            tripwire_line=[(0.0, 0.5), (1.0, 0.5)],
            snapshot_dir=temp_dir,
            dead_zone_width=0.0
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Frame 1: starts outside/above (centroid y = 20)
        boxes_f1 = MockBoxes([[40, 10, 60, 30]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        events_f1 = tracker.process_frame(frame)
        assert events_f1 == []
        
        # Frame 2: crosses inside/below (centroid y = 80) -> ENTER
        boxes_f2 = MockBoxes([[40, 70, 60, 90]], [1], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        events_f2 = tracker.process_frame(frame)
        assert len(events_f2) == 1
        assert events_f2[0]["event_type"] == "ENTER"
        
        # Frame 3: crosses back outside/above (centroid y = 20) -> LEAVE
        boxes_f3 = MockBoxes([[40, 10, 60, 30]], [1], [0.97])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f3)]
        events_f3 = tracker.process_frame(frame)
        assert len(events_f3) == 1
        assert events_f3[0]["event_type"] == "LEAVE"


def test_genuine_crossing_with_dead_zone(temp_dir):
    """Verifies a genuine crossing event is triggered when the centroid fully clears the dead zone."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Tripwire at y = 0.5 horizontal, dead_zone_width = 0.2 (20 px total, ±10 px, so inside is y > 60, outside is y < 40)
        tracker = ObjectTracker(
            tripwire_line=[(0.0, 0.5), (1.0, 0.5)],
            snapshot_dir=temp_dir,
            dead_zone_width=0.2
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Frame 1: Person starts outside (centroid y = 20)
        boxes_f1 = MockBoxes([[40, 10, 60, 30]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        assert tracker.process_frame(frame) == []
        
        # Frame 2: Person enters dead zone but does not clear it (centroid y = 55, inside is y > 60)
        boxes_f2 = MockBoxes([[40, 45, 60, 65]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        assert tracker.process_frame(frame) == []
        
        # Frame 3: Person fully clears the dead zone (centroid y = 65, which is > 60) -> Should trigger ENTER
        boxes_f3 = MockBoxes([[40, 55, 60, 75]], [1], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f3)]
        events = tracker.process_frame(frame)
        assert len(events) == 1
        assert events[0]["event_type"] == "ENTER"


def test_signed_distance_method():
    """Unit tests for _get_signed_distance with known horizontal and vertical line orientations."""
    tracker = ObjectTracker()
    
    # Directed segment AB from (0, 0) to (10, 0) - Horizontal vector pointing right
    A = (0.0, 0.0)
    B = (10.0, 0.0)
    
    # Point on vector AB
    P_on = (5.0, 0.0)
    assert tracker._get_signed_distance(A, B, P_on) == pytest.approx(0.0)
    
    # Point above vector AB (Left hand side of vector -> Positive distance)
    # A to B is right, so left is up (+y)
    P_above = (5.0, 3.0)
    assert tracker._get_signed_distance(A, B, P_above) == pytest.approx(3.0)
    
    # Point below vector AB (Right hand side of vector -> Negative distance)
    P_below = (5.0, -4.0)
    assert tracker._get_signed_distance(A, B, P_below) == pytest.approx(-4.0)
    
    # Directed segment CD from (0, 0) to (0, 10) - Vertical vector pointing down/up?
    # In screen coords, positive y is down, but let's test general math: CD from (0, 0) to (0, 10)
    C = (0.0, 0.0)
    D = (0.0, 10.0)
    
    # C to D is along +y. Left hand side of CD vector is -x (dx = 0, dy = 10, nx = -10, ny = 0)
    # Formula: (dx * (P_y - A_y) - dy * (P_x - A_x)) / line_len = (0 - 10 * (P_x - 0)) / 10 = -P_x
    P_left = (-3.0, 5.0)
    assert tracker._get_signed_distance(C, D, P_left) == pytest.approx(3.0)
    
    P_right = (4.0, 5.0)
    assert tracker._get_signed_distance(C, D, P_right) == pytest.approx(-4.0)

