import pytest
from src.types import CrossingEvent, PresenceState, DatabaseEvent

def test_crossing_event_attributes():
    event = CrossingEvent(
        event_type="ENTER",
        tracker_id=5,
        confidence=0.89,
        snapshot_path="/path/to/snap.jpg"
    )
    
    # Verify Attribute Access (Strong Typing)
    assert event.event_type == "ENTER"
    assert event.tracker_id == 5
    assert event.confidence == 0.89
    assert event.snapshot_path == "/path/to/snap.jpg"


def test_presence_state_attributes():
    state = PresenceState(
        is_someone_home=True,
        current_occupancy=2,
        last_updated="2026-06-02T03:00:00"
    )
    
    # Verify Attribute Access
    assert state.is_someone_home is True
    assert state.current_occupancy == 2
    assert state.last_updated == "2026-06-02T03:00:00"


def test_database_event_attributes():
    db_event = DatabaseEvent(
        id=101,
        event_type="LEAVE",
        tracker_id=12,
        confidence=0.94,
        timestamp="2026-06-02T03:05:00",
        snapshot_path=None,
        session_id="uuid-abc"
    )
    
    # Verify Attribute Access
    assert db_event.id == 101
    assert db_event.event_type == "LEAVE"
    assert db_event.tracker_id == 12
    assert db_event.confidence == 0.94
    assert db_event.timestamp == "2026-06-02T03:05:00"
    assert db_event.snapshot_path is None
    assert db_event.session_id == "uuid-abc"
