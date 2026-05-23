import numpy as np
import pytest
from unittest.mock import MagicMock
from src.database import DatabaseManager
from src.pipeline import PipelineOrchestrator
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
