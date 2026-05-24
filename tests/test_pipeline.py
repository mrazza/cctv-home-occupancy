import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import cv2
import logging
from src.database import DatabaseManager
from src.pipeline import PipelineOrchestrator, ThreadedVideoReader, FrameRegistry
from src.motion_detector import MotionDetector
from src.object_tracker import ObjectTracker

def test_pipeline_states_idle_to_active(db_manager):
    """Verifies that the orchestrator switches states correctly when motion is detected."""
    mock_md = MagicMock(spec=MotionDetector)
    mock_ot = MagicMock(spec=ObjectTracker)
    
    # Configure mock object tracker to have empty tracks
    mock_ot.track_histories = {}
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md,
        object_tracker=mock_ot,
        cooldown_frames=5
    )
    
    assert orchestrator.state == "IDLE"
    
    # 1. No motion frame
    mock_md.detect.return_value = False
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    events = orchestrator.process_single_frame(frame)
    
    assert len(events) == 0
    assert orchestrator.state == "IDLE"
    assert mock_ot.process_frame.called is False
    
    # 2. Motion detected frame
    mock_md.detect.return_value = True
    mock_ot.process_frame.return_value = []
    
    events = orchestrator.process_single_frame(frame)
    assert len(events) == 0
    assert orchestrator.state == "ACTIVE"
    assert orchestrator.cooldown_counter == 5
    assert mock_ot.process_frame.called is True

def test_pipeline_active_cooldown(db_manager):
    """Verifies the active cooldown works and reverts back to IDLE after cooldown frames."""
    mock_md = MagicMock(spec=MotionDetector)
    mock_ot = MagicMock(spec=ObjectTracker)
    
    mock_ot.track_histories = {}
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md,
        object_tracker=mock_ot,
        cooldown_frames=2
    )
    
    # Prime pipeline into ACTIVE state
    orchestrator.state = "ACTIVE"
    orchestrator.cooldown_counter = 2
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Frame 1: No motion, no tracks -> Cooldown decrements
    mock_md.detect.return_value = False
    mock_ot.process_frame.return_value = []
    
    orchestrator.process_single_frame(frame)
    assert orchestrator.state == "ACTIVE"
    assert orchestrator.cooldown_counter == 1
    
    # Frame 2: Still no motion, no tracks -> Reverts to IDLE
    orchestrator.process_single_frame(frame)
    assert orchestrator.state == "IDLE"
    assert mock_md.reset.called is True

def test_pipeline_logs_events(db_manager):
    """Verifies that detected events are logged into the database and update presence."""
    mock_md = MagicMock(spec=MotionDetector)
    mock_ot = MagicMock(spec=ObjectTracker)
    
    mock_ot.track_histories = {}
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md,
        object_tracker=mock_ot,
        cooldown_frames=5
    )
    
    # Prime into active
    orchestrator.state = "ACTIVE"
    
    # Set mock_ot to return an ENTER event
    mock_md.detect.return_value = True
    mock_ot.process_frame.return_value = [{
        "event_type": "ENTER",
        "tracker_id": 9,
        "confidence": 0.88,
        "snapshot_path": "/fake/path.jpg"
    }]
    
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    events = orchestrator.process_single_frame(frame)
    
    assert len(events) == 1
    assert events[0]["event_type"] == "ENTER"
    
    # Check DB
    state = db_manager.get_current_state()
    assert state["is_someone_home"] is True
    assert state["current_occupancy"] == 1
    
    recent_events = db_manager.get_recent_events(limit=1)
    assert recent_events[0]["event_type"] == "ENTER"
    assert recent_events[0]["tracker_id"] == 9
    assert recent_events[0]["confidence"] == 0.88
    assert recent_events[0]["snapshot_path"] == "/fake/path.jpg"

def test_pipeline_process_frame_exception(db_manager):
    """Verifies process_single_frame handles exceptions gracefully."""
    mock_md = MagicMock(spec=MotionDetector)
    mock_md.detect.side_effect = RuntimeError("Motion detector failure")
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # This should log but not raise the exception
    events = orchestrator.process_single_frame(frame)
    assert events == []

@patch('cv2.VideoCapture')
def test_threaded_video_reader_loop_and_reconnect(mock_video_capture):
    """Tests ThreadedVideoReader frame loop, timeout, and reconnection behavior."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Return a frame on first call, then trigger failure
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Read calls:
    # 1st call: (True, dummy_frame)
    # 2nd call: Raises an exception to test exception handling
    # 3rd call: (True, dummy_frame) for heartbeat log cover
    # 4th call: (True, dummy_frame) for heartbeat log cover
    # 5th call onwards: (False, None)
    def side_effect():
        if side_effect.count == 0:
            side_effect.count += 1
            return True, dummy_frame
        elif side_effect.count == 1:
            side_effect.count += 1
            raise RuntimeError("Capture read failure")
        elif side_effect.count in (2, 3, 4, 5):
            side_effect.count += 1
            return True, dummy_frame
        return False, None
    side_effect.count = 0
    mock_cap.read.side_effect = side_effect
    
    mock_video_capture.return_value = mock_cap
    
    reader = ThreadedVideoReader("rtsp://test-url")
    assert reader.start() == reader
    assert reader.start() == reader # Already running coverage
    
    # Force _last_read_time backwards to trigger timeout immediately on failure
    reader._last_read_time = time.time() - 20.0
    
    # Wait briefly for thread execution
    time.sleep(0.5)
    
    # Verify we successfully read at least once
    ret, frame = reader.read()
    assert ret is True
    assert frame is not None
    assert np.array_equal(frame, dummy_frame)
    
    # Test closed state behavior
    mock_cap.isOpened.return_value = False
    time.sleep(0.6)
    
    # Now set it open again but returning none so we timeout and reconnect
    mock_cap.isOpened.return_value = True
    reader._last_read_time = time.time() - 20.0
    # Make sure self.cap.read() returns False, None so we hit the timeout block
    mock_cap.read.side_effect = lambda: (False, None)
    time.sleep(1.2) # Allow more than 1.0 second sleep time of the reconnect routine to be executed
    
    reader.stop()
    assert reader.running is False

@patch('src.pipeline.ThreadedVideoReader')
def test_pipeline_orchestrator_run_on_stream(mock_reader_class, db_manager):
    """Verifies PipelineOrchestrator.run_on_stream loops and handles stops."""
    mock_reader = MagicMock()
    mock_reader_class.return_value = mock_reader
    mock_reader.start.return_value = mock_reader
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_reader.read.return_value = (True, dummy_frame)
    
    orchestrator = PipelineOrchestrator(db_manager=db_manager, fps_limit=100)
    
    # We will patch time.sleep inside run_on_stream to avoid delay
    with patch('time.sleep') as mock_sleep:
        # Instead of infinite loop, let's stop running inside sleep mock
        def sleep_side_effect(seconds):
            orchestrator.running = False
        mock_sleep.side_effect = sleep_side_effect
        
        orchestrator.run_on_stream("rtsp://dummy")
        
    assert orchestrator.running is False

@patch('src.pipeline.ThreadedVideoReader')
def test_pipeline_orchestrator_run_on_stream_exception(mock_reader_class, db_manager):
    """Verifies that unhandled exceptions in run_on_stream loop are caught gracefully."""
    orchestrator = PipelineOrchestrator(db_manager=db_manager)
    
    mock_reader = MagicMock()
    mock_reader_class.return_value = mock_reader
    mock_reader.start.return_value = mock_reader
    # Trigger an exception during the first iteration loop by raising it in read()
    mock_reader.read.side_effect = RuntimeError("Reader crash")
    
    # This should handle the crash inside try/except block of run_on_stream
    orchestrator.run_on_stream("rtsp://dummy")
    assert orchestrator.running is False

def test_pipeline_orchestrator_stop_log(db_manager):
    """Verifies PipelineOrchestrator.stop logging coverage."""
    orchestrator = PipelineOrchestrator(db_manager=db_manager)
    orchestrator.stop()
    assert orchestrator.running is False
