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

def test_pipeline_orchestrator_run_on_stream_event_trigger_mode(db_manager):
    """Verifies that run_on_stream behaves correctly in on-demand 'event' trigger mode."""
    from src.config import CONFIG
    original_mode = CONFIG.trigger_mode
    CONFIG.trigger_mode = "event"
    
    mock_md = MagicMock(spec=MotionDetector)
    mock_ot = MagicMock(spec=ObjectTracker)
    mock_ot.track_histories = {}
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md,
        object_tracker=mock_ot,
        fps_limit=100
    )
    
    # 1. Initially trigger_mode = event, inactive trigger window, no active tracks
    # We patch ThreadedVideoReader to verify it isn't started
    with patch('src.pipeline.ThreadedVideoReader') as mock_reader_class:
        mock_reader = MagicMock()
        mock_reader_class.return_value = mock_reader
        mock_reader.start.return_value = mock_reader
        
        # Patch time.sleep to terminate loop after first iteration
        with patch('time.sleep') as mock_sleep:
            def sleep_side_effect(seconds):
                orchestrator.running = False
            mock_sleep.side_effect = sleep_side_effect
            
            orchestrator.run_on_stream("rtsp://dummy")
            
        assert orchestrator.reader is None
        assert mock_reader_class.called is False

    # 2. Trigger window active: reader should be started lazily
    CONFIG.trigger_mode = "event"
    orchestrator.active_until = time.time() + 10.0
    with patch('src.pipeline.ThreadedVideoReader') as mock_reader_class:
        mock_reader = MagicMock()
        mock_reader_class.return_value = mock_reader
        mock_reader.start.return_value = mock_reader
        
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_reader.read.return_value = (True, dummy_frame)
        
        with patch('time.sleep') as mock_sleep:
            def sleep_side_effect(seconds):
                orchestrator.running = False
            mock_sleep.side_effect = sleep_side_effect
            
            orchestrator.run_on_stream("rtsp://dummy")
            
        assert mock_reader_class.called is True
        assert mock_reader.start.called is True
        assert mock_reader.stop.called is True

    # 3. Active tracks extend the trigger window even when active_until has passed
    CONFIG.trigger_mode = "event"
    orchestrator.active_until = time.time() - 10.0 # expired
    mock_ot.track_histories = {1: [(50, 50)]} # active track!
    
    with patch('src.pipeline.ThreadedVideoReader') as mock_reader_class:
        mock_reader = MagicMock()
        mock_reader_class.return_value = mock_reader
        mock_reader.start.return_value = mock_reader
        
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_reader.read.return_value = (True, dummy_frame)
        
        with patch('time.sleep') as mock_sleep:
            def sleep_side_effect(seconds):
                orchestrator.running = False
            mock_sleep.side_effect = sleep_side_effect
            
            orchestrator.run_on_stream("rtsp://dummy")
            
        assert orchestrator.active_until > time.time() # window should be extended
        assert mock_reader_class.called is True
        assert mock_reader.start.called is True
        assert mock_reader.stop.called is True

    # Restore original config
    CONFIG.trigger_mode = original_mode


def test_pipeline_orchestrator_event_mode_cleanup_on_expiry(db_manager):
    """Verifies that an active stream reader is stopped and cleared when active_until expires and there are no active tracks."""
    from src.config import CONFIG
    original_mode = CONFIG.trigger_mode
    CONFIG.trigger_mode = "event"
    
    mock_md = MagicMock(spec=MotionDetector)
    mock_ot = MagicMock(spec=ObjectTracker)
    mock_ot.track_histories = {}
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md,
        object_tracker=mock_ot,
        fps_limit=100
    )
    
    # Set the active window to be in the past
    orchestrator.active_until = time.time() - 10.0
    
    # We pre-populate the orchestrator with an active mock reader
    mock_reader = MagicMock()
    mock_reader.running = True
    orchestrator.reader = mock_reader
    
    with patch('time.sleep') as mock_sleep:
        def sleep_side_effect(seconds):
            orchestrator.running = False
        mock_sleep.side_effect = sleep_side_effect
        
        orchestrator.run_on_stream("rtsp://dummy")
        
    # Since active_until was in the past and track_histories is empty,
    # the reader should have been stopped and cleared.
    assert mock_reader.stop.called is True
    assert orchestrator.reader is None
    
    CONFIG.trigger_mode = original_mode


def test_pipeline_orchestrator_trigger_event_window(db_manager):
    """Verifies trigger_event_window increases active_until as expected."""
    orchestrator = PipelineOrchestrator(db_manager=db_manager)
    assert orchestrator.active_until == 0.0
    
    orchestrator.trigger_event_window(30)
    assert orchestrator.active_until > time.time() + 25.0
    assert orchestrator.active_until < time.time() + 35.0


@patch('cv2.VideoCapture')
def test_threaded_video_reader_heartbeat(mock_video_capture):
    """Tests that ThreadedVideoReader successfully triggers heartbeat log after 3 frames."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    def side_effect():
        return True, dummy_frame
    mock_cap.read.side_effect = side_effect
    mock_video_capture.return_value = mock_cap
    
    reader = ThreadedVideoReader("rtsp://test-url")
    reader.start()
    
    # Allow background thread to process several frames to trigger the heartbeat log
    time.sleep(0.5)
    reader.stop()
    
    assert reader._frame_count >= 3


def test_pipeline_webhook_success(db_manager, monkeypatch):
    """Verifies that webhook urls are successfully called when events are processed."""
    import httpx
    from src.config import CameraConfig
    
    # Configure mock object tracker to return an event
    mock_md = MagicMock(spec=MotionDetector)
    mock_md.detect.return_value = True
    
    mock_ot = MagicMock(spec=ObjectTracker)
    mock_ot.track_histories = {}
    mock_ot.process_frame.return_value = [{
        "event_type": "ENTER",
        "tracker_id": 1,
        "confidence": 0.85,
        "snapshot_path": "snapshots/test_snapshot.jpg"
    }]
    
    # Inject CONFIG overrides using monkeypatch
    monkeypatch.setattr("src.pipeline.CONFIG.webhook_urls", ["http://webhook1.test", "http://webhook2.test"])
    monkeypatch.setattr("src.pipeline.CONFIG.webhook_timeout", 3)
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md,
        object_tracker=mock_ot
    )
    
    # We patch httpx.post
    posted_payloads = []
    posted_urls = []
    posted_timeouts = []
    
    def mock_post(url, json, timeout):
        posted_urls.append(url)
        posted_payloads.append(json)
        posted_timeouts.append(timeout)
        mock_response = MagicMock()
        mock_response.status_code = 200
        return mock_response
        
    with patch("httpx.post", side_effect=mock_post):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        events = orchestrator.process_frame(frame) if hasattr(orchestrator, "process_frame") else orchestrator.process_single_frame(frame)
        
        # Give threads a short time to finish since they are executed in separate threads
        time.sleep(0.1)
        
    assert len(events) == 1
    assert len(posted_urls) == 2
    assert "http://webhook1.test" in posted_urls
    assert "http://webhook2.test" in posted_urls
    assert posted_timeouts == [3, 3]
    
    # Check payload contents
    payload = posted_payloads[0]
    assert payload["event_type"] == "ENTER"
    assert payload["tracker_id"] == 1
    assert payload["confidence"] == 0.85
    assert payload["snapshot_path"] == "snapshots/test_snapshot.jpg"
    assert payload["is_someone_home"] is True
    assert payload["current_occupancy"] == 1
    assert payload["timestamp"] is not None


def test_pipeline_webhook_failure(db_manager, monkeypatch, caplog):
    """Verifies that webhook errors (network error, timeout, non-200 status) are handled gracefully."""
    import httpx
    
    mock_md = MagicMock(spec=MotionDetector)
    mock_md.detect.return_value = True
    
    mock_ot = MagicMock(spec=ObjectTracker)
    mock_ot.track_histories = {}
    mock_ot.process_frame.return_value = [{
        "event_type": "LEAVE",
        "tracker_id": 2,
        "confidence": 0.90,
        "snapshot_path": "snapshots/test_snapshot_2.jpg"
    }]
    
    monkeypatch.setattr("src.pipeline.CONFIG.webhook_urls", ["http://webhook-fail-status.test", "http://webhook-fail-network.test", "http://webhook-fail-unexpected.test"])
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md,
        object_tracker=mock_ot
    )
    
    def mock_post(url, json, timeout):
        if "fail-status" in url:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            return mock_resp
        elif "fail-network" in url:
            raise httpx.RequestError("Network error", request=MagicMock())
        else:
            raise ValueError("Unexpected exception")
            
    with caplog.at_level(logging.WARNING), patch("httpx.post", side_effect=mock_post):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        orchestrator.process_single_frame(frame)
        time.sleep(0.1)
        
    # Check that appropriate warning/error messages are logged
    log_texts = [record.message for record in caplog.records]
    assert any("returned non-success status code" in text for text in log_texts)
    assert any("An error occurred while requesting" in text for text in log_texts)
    assert any("Unexpected error dispatching webhook" in text for text in log_texts)


def test_pipeline_webhook_db_failure(db_manager, monkeypatch):
    """Verifies that webhook payload is built with fallback values if database state retrieval fails."""
    mock_md = MagicMock(spec=MotionDetector)
    mock_md.detect.return_value = True
    
    mock_ot = MagicMock(spec=ObjectTracker)
    mock_ot.track_histories = {}
    mock_ot.process_frame.return_value = [{
        "event_type": "ENTER",
        "tracker_id": 1,
        "confidence": 0.85,
        "snapshot_path": "snapshots/test_snapshot.jpg"
    }]
    
    monkeypatch.setattr("src.pipeline.CONFIG.webhook_urls", ["http://webhook1.test"])
    
    orchestrator = PipelineOrchestrator(
        db_manager=db_manager,
        motion_detector=mock_md,
        object_tracker=mock_ot
    )
    
    # Mock self.db.get_current_state to raise an exception
    orchestrator.db.get_current_state = MagicMock(side_effect=Exception("DB error"))
    
    posted_payloads = []
    def mock_post(url, json, timeout):
        posted_payloads.append(json)
        mock_response = MagicMock()
        mock_response.status_code = 200
        return mock_response
        
    with patch("httpx.post", side_effect=mock_post):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        orchestrator.process_single_frame(frame)
        time.sleep(0.1)
        
    assert len(posted_payloads) == 1
    payload = posted_payloads[0]
    assert payload["event_type"] == "ENTER"
    assert payload["is_someone_home"] is None
    assert payload["current_occupancy"] is None
    assert payload["timestamp"] is None



