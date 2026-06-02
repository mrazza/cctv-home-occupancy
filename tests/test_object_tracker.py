import os
import numpy as np
import pytest
from unittest.mock import patch
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
        assert events_f2[0].event_type == "LEAVE"
        assert events_f2[0].tracker_id == 1
        assert events_f2[0].confidence == 0.96
        assert events_f2[0].snapshot_path is not None
        assert os.path.exists(events_f2[0].snapshot_path)

        # Frame 3: Track 1 is still active but hasn't crossed again
        boxes_f3 = MockBoxes([[40, 5, 60, 25]], [1], [0.97])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f3)]
        events_f3 = tracker.process_frame(frame)
        assert events_f3 == []

        # Frame 4: Track 1 disappears (but history is kept due to TTL buffer)
        boxes_f4 = MockBoxes([], [], [])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f4)]
        events_f4 = tracker.process_frame(frame)
        assert events_f4 == []
        assert 1 in tracker.track_histories

        # Fast forward frames to exceed track_buffer (30 by default)
        for _ in range(35):
            tracker.process_frame(frame)
        
        # Now it should be cleaned up
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
        assert events_f2[0].event_type == "ENTER"
        
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
        assert all_events[0].event_type == "ENTER"


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
        assert events_f2[0].event_type == "ENTER"
        
        # Frame 3: crosses back outside/above (centroid y = 20) -> LEAVE
        boxes_f3 = MockBoxes([[40, 10, 60, 30]], [1], [0.97])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f3)]
        events_f3 = tracker.process_frame(frame)
        assert len(events_f3) == 1
        assert events_f3[0].event_type == "LEAVE"


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
        assert events[0].event_type == "ENTER"


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

def test_track_recovery_prevents_duplicate_events(temp_dir):
    """Verifies that if a track drops for a frame and recovers, it retains its confirmed side and doesn't trigger duplicate events."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Tripwire at y = 0.5 horizontal, dead_zone_width = 0.2
        tracker = ObjectTracker(
            tripwire_line=[(0.0, 0.5), (1.0, 0.5)],
            snapshot_dir=temp_dir,
            dead_zone_width=0.2
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        all_events = []
        
        # Frame 1: Person starts inside (centroid y = 80, inside is +1, outside is -1)
        # Bounding box y centers: 80
        boxes_f1 = MockBoxes([[40, 70, 60, 90]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        all_events.extend(tracker.process_frame(frame))
        
        # Frame 2: Person crosses to outside (centroid y = 20) -> LEAVE
        boxes_f2 = MockBoxes([[40, 10, 60, 30]], [1], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        all_events.extend(tracker.process_frame(frame))
        
        # Frame 3: YOLO drops the track (e.g. occlusion)
        boxes_f3 = MockBoxes([], [], [])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f3)]
        all_events.extend(tracker.process_frame(frame))
        
        # Frame 4: YOLO recovers the track with same ID (track buffer), but jitter makes it centroid y=52 (inside dead zone)
        # Wait, if y=52, it's slightly "inside" the absolute center y=50, but within dead zone (40-60).
        boxes_f4 = MockBoxes([[40, 42, 60, 62]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f4)]
        all_events.extend(tracker.process_frame(frame))
        
        # Frame 5: Person moves fully outside again (y=20)
        boxes_f5 = MockBoxes([[40, 10, 60, 30]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f5)]
        all_events.extend(tracker.process_frame(frame))
        
        # Assert only 1 LEAVE event was triggered (the original one)
        assert len(all_events) == 1
        assert all_events[0].event_type == "LEAVE"

def test_strict_segment_ignores_out_of_bounds_crossing(temp_dir):
    """Verifies that with tripwire_strict_segment=True, crossings outside the segment bounds are ignored."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Segment from x=0.2 to x=0.8 (pixels 20 to 80).
        tracker_strict = ObjectTracker(
            tripwire_line=[(0.2, 0.5), (0.8, 0.5)],
            snapshot_dir=temp_dir,
            tripwire_strict_segment=True
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Person crosses at x=90 (outside the segment bounds 20-80).
        # Frame 1: inside
        boxes_f1 = MockBoxes([[85, 70, 95, 90]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        tracker_strict.process_frame(frame)
        
        # Frame 2: crosses outside
        boxes_f2 = MockBoxes([[85, 10, 95, 30]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        events = tracker_strict.process_frame(frame)
        
        # Should be ignored
        assert len(events) == 0

def test_infinite_line_allows_out_of_bounds_crossing(temp_dir):
    """Verifies that with tripwire_strict_segment=False, crossings outside the segment bounds trigger events."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Segment from x=0.2 to x=0.8 (pixels 20 to 80).
        tracker_inf = ObjectTracker(
            tripwire_line=[(0.2, 0.5), (0.8, 0.5)],
            snapshot_dir=temp_dir,
            tripwire_strict_segment=False
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Person crosses at x=90 (outside the segment bounds 20-80).
        # Frame 1: inside
        boxes_f1 = MockBoxes([[85, 70, 95, 90]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f1)]
        tracker_inf.process_frame(frame)
        
        # Frame 2: crosses outside
        boxes_f2 = MockBoxes([[85, 10, 95, 30]], [1], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_f2)]
        events = tracker_inf.process_frame(frame)
        
        # Should trigger event
        assert len(events) == 1
        assert events[0].event_type == "LEAVE"


def test_latest_boxes_tracking(temp_dir):
    """Verifies that ObjectTracker.latest_boxes tracks YOLO results correctly."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        tracker = ObjectTracker(snapshot_dir=temp_dir)
        
        # 1. Initially, latest_boxes must be None
        assert tracker.latest_boxes is None
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 2. Process frame with no boxes -> latest_boxes must remain None
        mock_yolo.return_value.track.return_value = []
        tracker.process_frame(frame)
        assert tracker.latest_boxes is None
        
        # 3. Process frame with boxes -> latest_boxes must store the active boxes
        boxes_f1 = MockBoxes([[40, 70, 60, 90]], [1], [0.95])
        mock_result = MockResult(boxes=boxes_f1)
        mock_yolo.return_value.track.return_value = [mock_result]
        
        tracker.process_frame(frame)
        assert tracker.latest_boxes is not None
        assert tracker.latest_boxes == boxes_f1
        
        # 4. Next frame, tracking returns no detections -> latest_boxes resets to None
        mock_yolo.return_value.track.return_value = []
        tracker.process_frame(frame)
        assert tracker.latest_boxes is None


def test_object_tracker_get_signed_distance_zero_length():
    """Verifies that _get_signed_distance returns 0.0 when the tripwire segment has zero length."""
    tracker = ObjectTracker()
    A = (1.0, 1.0)
    B = (1.0, 1.0)
    P = (2.0, 2.0)
    assert tracker._get_signed_distance(A, B, P) == 0.0


def test_object_tracker_is_point_in_segment_bounds_zero_length():
    """Verifies that _is_point_in_segment_bounds returns False when the segment has zero length."""
    tracker = ObjectTracker()
    A = (1.0, 1.0)
    B = (1.0, 1.0)
    P = (2.0, 2.0)
    assert tracker._is_point_in_segment_bounds(A, B, P) is False


def test_collinear_resolution_inside_and_outside(temp_dir):
    """Verifies that a collinear track is correctly resolved to either inside or outside side."""
    with patch("src.object_tracker.YOLO") as mock_yolo:
        # Tripwire at y = 0.5 horizontal (pixel y=50 on 100 height)
        # Set dead_zone_width to 0.1 (10 pixels total, ±5 px)
        tracker = ObjectTracker(
            tripwire_line=[(0.0, 0.5), (1.0, 0.5)],
            snapshot_dir=temp_dir,
            dead_zone_width=0.1
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # 1. SCENARIO A: Resolving collinear to INSIDE (+1)
        # Frame 1: Person starts exactly collinear at y=50
        # Centroid y is 50.
        boxes_a1 = MockBoxes([[40, 40, 60, 60]], [10], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_a1)]
        tracker.process_frame(frame)
        
        # Verify initial side is collinear (0)
        assert tracker.track_confirmed_sides[10] == 0

        # Frame 2: Person moves inside (centroid y=80, which is outside the dead zone y > 55)
        boxes_a2 = MockBoxes([[40, 70, 60, 90]], [10], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_a2)]
        tracker.process_frame(frame)

        # Verify side resolved to INSIDE (+1)
        assert tracker.track_confirmed_sides[10] == 1

        # 2. SCENARIO B: Resolving collinear to OUTSIDE (-1)
        # Frame 1: Person starts exactly collinear at y=50
        boxes_b1 = MockBoxes([[40, 40, 60, 60]], [20], [0.95])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_b1)]
        tracker.process_frame(frame)

        # Verify initial side is collinear (0)
        assert tracker.track_confirmed_sides[20] == 0

        # Frame 2: Person moves outside (centroid y=20, which is outside the dead zone y < 45)
        boxes_b2 = MockBoxes([[40, 10, 60, 30]], [20], [0.96])
        mock_yolo.return_value.track.return_value = [MockResult(boxes=boxes_b2)]
        tracker.process_frame(frame)

        # Verify side resolved to OUTSIDE (-1)
        assert tracker.track_confirmed_sides[20] == -1


