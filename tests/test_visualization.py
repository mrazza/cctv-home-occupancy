from src.visualization import calculate_arrow_endpoint, compute_dead_zone_lines

def test_calculate_arrow_endpoint_normal():
    A = (0, 0)
    B = (10, 0)
    # Midpoint should be (5, 0)
    # Vector AB is (10, 0). Perpendicular vector pointing to the left side: (-0, 10) = (0, 10).
    # Normalized is (0.0, 1.0).
    # Arrow destination should be (5 + 50 * 0, 0 + 50 * 1) = (5, 50).
    mid, dest = calculate_arrow_endpoint(A, B, length=50.0)
    assert mid == (5, 0)
    assert dest == (5, 50)

def test_calculate_arrow_endpoint_zero_length():
    A = (10, 10)
    B = (10, 10)
    mid, dest = calculate_arrow_endpoint(A, B)
    assert mid == (10, 10)
    assert dest == (10, 10)

def test_compute_dead_zone_lines_normal():
    pt_A = (0, 0)
    pt_B = (10, 0)
    # Vector AB is (10, 0).
    # Unit normal pointing to the inside (+1 / left-hand side): (-dy, dx) / mag = (0, 10) / 10 = (0, 1)
    # Inside offset line: offset by +5 in y direction.
    # Outside offset line: offset by -5 in y direction.
    (ins_A, ins_B), (out_A, out_B) = compute_dead_zone_lines(pt_A, pt_B, dead_zone_half_px=5.0)
    assert ins_A == (0, 5)
    assert ins_B == (10, 5)
    assert out_A == (0, -5)
    assert out_B == (10, -5)

def test_compute_dead_zone_lines_zero_length():
    pt_A = (10, 10)
    pt_B = (10, 10)
    (ins_A, ins_B), (out_A, out_B) = compute_dead_zone_lines(pt_A, pt_B, dead_zone_half_px=5.0)
    assert ins_A == (10, 10)
    assert ins_B == (10, 10)
    assert out_A == (10, 10)
    assert out_B == (10, 10)

def test_draw_hud_sidebar_resolutions():
    from unittest.mock import MagicMock
    import numpy as np
    from visualize_tracker import draw_hud_sidebar

    mock_tracker = MagicMock()
    mock_tracker.track_histories = {}
    mock_tracker.model = MagicMock()
    mock_tracker.model.model_name = "yolov8n.pt"
    mock_tracker.conf = 0.25
    mock_tracker.track_buffer = 30
    mock_tracker.tripwire_strict_segment = True
    mock_tracker.dead_zone_width = 0.1
    mock_tracker.tripwire_line = ((0.1, 0.2), (0.9, 0.8))

    # Test full mode (e.g. 1920x1080)
    frame_full = np.zeros((1080, 1920, 3), dtype=np.uint8)
    draw_hud_sidebar(
        frame_full, mock_tracker, is_paused=False, show_roi=True,
        show_tripwire=True, show_history=True, fps=30.0, frame_count=100,
        source_name="rtsp://test_stream"
    )
    assert not np.all(frame_full == 0)

    # Test compact mode (e.g. 640x360)
    frame_compact = np.zeros((360, 640, 3), dtype=np.uint8)
    draw_hud_sidebar(
        frame_compact, mock_tracker, is_paused=True, show_roi=False,
        show_tripwire=False, show_history=False, fps=15.0, frame_count=500,
        source_name="rtsp://test_stream_low_res"
    )
    assert not np.all(frame_compact == 0)

from src.visualization import draw_zonal_tripwire, draw_tracked_person
import numpy as np

def test_draw_zonal_tripwire():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pt_A = (10, 50)
    pt_B = (90, 50)
    res = draw_zonal_tripwire(frame, pt_A, pt_B, dead_zone_half_px=10.0)
    assert res is not None
    assert not np.all(res == 0)


def test_draw_tracked_person_bottom_center():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (20, 20, 60, 80)
    res = draw_tracked_person(frame, bbox, tracker_id=5, tracker_point="bottom_center", zone_label="INSIDE", conf=0.88)
    assert res is not None
    assert not np.all(res == 0)


def test_draw_tracked_person_centroid():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (20, 20, 60, 80)
    res = draw_tracked_person(frame, bbox, tracker_id=5, tracker_point="centroid", zone_label="OUTSIDE", conf=0.75)
    assert res is not None
    assert not np.all(res == 0)
